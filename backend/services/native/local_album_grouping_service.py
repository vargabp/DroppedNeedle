"""Apply durable local grouping contexts after target tag indexing commits."""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath
from typing import Any

from core.exceptions import StaleRevisionError, ValidationError

from infrastructure.persistence.native_library_store import NativeLibraryStore
from models.identification import (
    ExistingAlbumMembership,
    GroupingApplication,
    GroupingTrack,
)
from services.native.identification_queue_service import IdentificationQueueService
from services.native.identification_revisions import album_input_revisions
from services.native.local_album_grouper import (
    CONTINUITY_COMPONENT_EDGE_LIMIT,
    PROVISIONAL_PARSED_GROUP,
    LocalAlbumGrouper,
    _hungarian_min,
    grouping_directory,
    normalize_group_value,
)
from services.native.track_provenance import (
    derive_album_provenance,
    derive_title_artist_provenance,
    resolve_provenance,
    track_stem,
)

_GROUPING_NAMESPACE = uuid.UUID("296d6087-edf2-4861-8c72-7eae22654aef")
ARTIST_RESOLUTION_BATCH_SIZE = 256
QUEUE_BATCH_SIZE = 256
STAGING_BATCH_SIZE = 500
STAGED_GROUPING_THRESHOLD = 512
# CONTINUITY_COMPONENT_EDGE_LIMIT lives in local_album_grouper (imported
# above) so the small-path and staged-path continuity caps stay identical.


def grouping_track_from_row(row: dict) -> GroupingTrack:
    relative_path = str(row["relative_path"])
    stem = track_stem(relative_path)
    raw_album_title = str(row.get("tag_album_title") or "")
    raw_album_artist_name = str(row.get("tag_album_artist_name") or "")
    title = str(row["title"] or "")
    artist_name = str(row["artist_name"] or "")
    display_album_title = str(row.get("album_title") or "")
    display_album_artist_name = str(row.get("album_artist_name") or "")
    album_title_provenance = resolve_provenance(
        row.get("album_title_provenance"),
        derive_album_provenance(raw_album_title, display_album_title),
    )
    album_artist_provenance = resolve_provenance(
        row.get("album_artist_provenance"),
        derive_album_provenance(raw_album_artist_name, display_album_artist_name),
    )
    return GroupingTrack(
        local_track_id=str(row["id"]),
        root_id=str(row["root_id"]),
        relative_path=relative_path,
        title=title,
        artist_name=artist_name,
        album_title=(
            display_album_title
            if not raw_album_title and album_title_provenance == "parsed"
            else raw_album_title
        ),
        album_artist_name=(
            display_album_artist_name
            if not raw_album_artist_name and album_artist_provenance == "parsed"
            else raw_album_artist_name
        ),
        title_provenance=resolve_provenance(
            row.get("title_provenance"),
            derive_title_artist_provenance(title, artist_name, stem),
        ),
        album_title_provenance=album_title_provenance,
        album_artist_provenance=album_artist_provenance,
        artist_sort_name=row["artist_sort"],
        album_artist_sort_name=row["album_artist_sort"],
        track_number=int(row["track_number"] or 0),
        disc_number=int(row["disc_number"] or 1),
        duration_seconds=row["duration_seconds"],
        recording_mbid=row["embedded_recording_mbid"],
        release_mbid=row["embedded_release_mbid"],
        release_group_mbid=row["embedded_release_group_mbid"],
        is_compilation=bool(row["is_compilation"]),
        tags_readable=not bool(row["metadata_incomplete"]),
        membership_locked=bool(row["membership_locked"]),
        current_album_id=str(row["local_album_id"]),
    )


def grouping_artist_candidate_id(display_name: str) -> str:
    return str(uuid.uuid5(_GROUPING_NAMESPACE, f"artist:{display_name}:group"))


def grouping_album_id(grouping_key: str) -> str:
    return str(uuid.uuid5(_GROUPING_NAMESPACE, grouping_key))


class LocalAlbumGroupingService:
    def __init__(
        self,
        store: NativeLibraryStore,
        queue: IdentificationQueueService,
        grouper: LocalAlbumGrouper | None = None,
        *,
        _cohort_resolver_getter: Callable[[], Any] | None = None,
        _cohort_filesystem: Any = None,
        _cohort_max_total_bytes: int = 256 * 1024 * 1024,
        _cohort_timeout_seconds: float = 10,
        _cohort_tokens_candidate_enabled: bool = False,
        _cohort_summaries_candidate_enabled: bool = False,
        _cohort_continuity_candidate_enabled: bool = False,
        _cohort_provision_candidate_enabled: bool = False,
        _cohort_membership_candidate_enabled: bool = False,
        _cohort_retirement_candidate_enabled: bool = False,
        _cohort_queue_candidate_enabled: bool = False,
    ) -> None:
        self._store = store
        self._queue = queue
        self._grouper = grouper or LocalAlbumGrouper()
        # Private disposable integration only. No production constructor enables
        # this and candidate tables are deliberately not registered migrations.
        if _cohort_resolver_getter is not None and _cohort_filesystem is None:
            raise ValueError('Cohort grouping requires the native filesystem coordinator.')
        self._cohort_resolver_getter = _cohort_resolver_getter
        self._cohort_filesystem = _cohort_filesystem
        self._cohort_max_total_bytes = _cohort_max_total_bytes
        self._cohort_timeout_seconds = _cohort_timeout_seconds
        self._cohort_tokens_candidate_enabled = _cohort_tokens_candidate_enabled
        self._cohort_summaries_candidate_enabled = _cohort_summaries_candidate_enabled
        self._cohort_continuity_candidate_enabled = _cohort_continuity_candidate_enabled
        self._cohort_provision_candidate_enabled = _cohort_provision_candidate_enabled
        self._cohort_membership_candidate_enabled = _cohort_membership_candidate_enabled
        self._cohort_retirement_candidate_enabled = _cohort_retirement_candidate_enabled
        self._cohort_queue_candidate_enabled = _cohort_queue_candidate_enabled

    async def _cohort_binding_for_context(self, root_id: str, directory: str) -> str | None:
        # Discovery only; fresh byte and SQL proof are still required at both
        # writes. Never fall back to legacy grouping for a partial bound context.
        rows = await self._store.get_grouping_context_tracks(root_id, directory)
        ids = {str(row['id']) for row in rows}
        def discover(connection):
            bindings = {}
            for track_id in ids:
                for row in connection.execute('SELECT b.id,b.payload_json FROM library_import_cohort_bindings b '
                        'JOIN library_import_cohort_members m ON m.binding_id=b.id WHERE m.local_track_id=?', (track_id,)):
                    bindings[row['id']] = json.loads(row['payload_json'])
            if not bindings:
                return None
            if len(bindings) != 1:
                raise ValidationError('Multiple native cohorts share the grouping context.')
            binding_id, payload = next(iter(bindings.items()))
            if (payload['root_id'] != root_id or payload['album_parent'] != directory
                    or {member['track']['id'] for member in payload['members']} != ids):
                raise StaleRevisionError('Grouping context is not the exclusive complete native cohort.')
            return binding_id
        return await self._store._read(discover)

    async def regroup_run(
        self,
        run_id: str,
        *,
        now: float,
        checkpoint: Callable[[str, str], Awaitable[bool]] | None = None,
        frozen_policy_revision: str = "",
    ) -> int:
        enqueued = 0
        while contexts := await self._store.get_pending_grouping_contexts(run_id):
            for context in contexts:
                if checkpoint is not None and not await checkpoint(
                    run_id, frozen_policy_revision
                ):
                    return enqueued
                root_id = str(context["root_id"])
                relative_directory = str(context["relative_directory"])
                candidate_count = await self._store.count_grouping_context_candidates(
                    root_id,
                    relative_directory,
                    limit=STAGED_GROUPING_THRESHOLD + 1,
                )
                if self._cohort_resolver_getter is not None:
                    if await self._store.count_grouping_context_candidates(root_id, relative_directory, limit=501) > 500:
                        raise ValidationError('Private complete cohort exceeds observation member budget.')
                    binding_id = await self._cohort_binding_for_context(root_id, relative_directory)
                    if binding_id is None:
                        # This private slice only admits complete bound contexts.
                        # Falling through would permit an unfenced legacy write
                        # if a binding arrived after the discovery read.
                        raise ValidationError('Opt-in cohort grouping requires a complete bound context.')
                    if binding_id is not None:
                        from services.native.import_cohort_grouping_candidate import (
                            apply_cohort_grouping_candidate, complete_cohort_grouping_candidate,
                        )
                        resolver = self._cohort_resolver_getter()
                        if resolver.policy_revision != frozen_policy_revision:
                            raise StaleRevisionError('Frozen cohort scan policy changed.')
                        if candidate_count > STAGED_GROUPING_THRESHOLD:
                            enqueued += await self._regroup_large_context(run_id, root_id, relative_directory,
                                now=now, checkpoint=checkpoint, frozen_policy_revision=frozen_policy_revision,
                                _cohort_binding_candidate=binding_id)
                            continue
                        arguments = dict(store=self._store, filesystem=self._cohort_filesystem,
                            resolver=resolver, binding_id=binding_id, run_id=run_id, now=now,
                            max_total_bytes=self._cohort_max_total_bytes,
                            timeout_seconds=self._cohort_timeout_seconds)
                        await apply_cohort_grouping_candidate(**arguments, grouper=self._grouper)
                        dispositions = await complete_cohort_grouping_candidate(**arguments, checkpoint=checkpoint)
                        if dispositions is None:
                            return enqueued
                        enqueued += sum(created for _, created in dispositions)
                        continue
                if candidate_count > STAGED_GROUPING_THRESHOLD:
                    enqueued += await self._regroup_large_context(
                        run_id,
                        root_id,
                        relative_directory,
                        now=now,
                        checkpoint=checkpoint,
                        frozen_policy_revision=frozen_policy_revision,
                    )
                    continue
                rows = await self._store.get_grouping_context_tracks(
                    root_id, relative_directory
                )
                rows = [
                    row
                    for row in rows
                    if grouping_directory(str(row["relative_path"]))
                    == context["relative_directory"]
                ]
                memberships: dict[str, ExistingAlbumMembership] = {}
                for row in rows:
                    album_id = str(row["local_album_id"])
                    membership = memberships.setdefault(
                        album_id,
                        ExistingAlbumMembership(
                            local_album_id=album_id,
                            track_ids=[],
                            created_at=float(row["album_created_at"]),
                        ),
                    )
                    membership.track_ids.append(str(row["id"]))
                groups = self._grouper.group(
                    [grouping_track_from_row(row) for row in rows],
                    existing=list(memberships.values()),
                )
                artist_names = list(
                    dict.fromkeys(group.album_artist_name for group in groups)
                )
                artist_ids: dict[str, str] = {}
                for offset in range(0, len(artist_names), ARTIST_RESOLUTION_BATCH_SIZE):
                    if checkpoint is not None and not await checkpoint(
                        run_id, frozen_policy_revision
                    ):
                        return enqueued
                    names = artist_names[offset : offset + ARTIST_RESOLUTION_BATCH_SIZE]
                    candidates = [
                        (name, None, "group", grouping_artist_candidate_id(name))
                        for name in names
                    ]
                    resolved = await self._store.resolve_or_create_local_artists(
                        candidates, now=now, background=True
                    )
                    artist_ids.update(
                        {
                            name: resolved[grouping_artist_candidate_id(name)][0]
                            for name in names
                        }
                    )
                applications: list[GroupingApplication] = []
                for group in groups:
                    artist_id = artist_ids[group.album_artist_name]
                    album_id = group.retained_album_id or grouping_album_id(
                        group.grouping_key
                    )
                    applications.append(
                        GroupingApplication(
                            group=group,
                            local_album_id=album_id,
                            local_artist_id=artist_id,
                        )
                    )
                album_ids, _ = await self._store.apply_grouping_context(
                    run_id,
                    str(context["root_id"]),
                    str(context["relative_directory"]),
                    applications,
                    now=now,
                )
                refreshed_rows = await self._store.get_grouping_context_tracks(
                    str(context["root_id"]), str(context["relative_directory"])
                )
                rows_by_album: dict[str, list[dict]] = {}
                for row in refreshed_rows:
                    rows_by_album.setdefault(str(row["local_album_id"]), []).append(row)
                queue_candidates: list[tuple[str, str, str]] = []
                for album_id in album_ids:
                    album_rows = rows_by_album.get(album_id, [])
                    if not album_rows:
                        continue
                    policy = {str(row["applied_policy"]) for row in album_rows}
                    embedded = any(
                        row["embedded_release_group_mbid"]
                        or row["embedded_release_mbid"]
                        for row in album_rows
                    )
                    if policy == {"automatic"} or (
                        policy == {"local_metadata"} and embedded
                    ):
                        revisions = album_input_revisions(album_rows)
                        queue_candidates.append(
                            (
                                album_id,
                                ":".join(revisions),
                                "automatic"
                                if policy == {"automatic"}
                                else "post_processing",
                            )
                        )
                for offset in range(0, len(queue_candidates), QUEUE_BATCH_SIZE):
                    if checkpoint is not None and not await checkpoint(
                        run_id, frozen_policy_revision
                    ):
                        return enqueued
                    batch = queue_candidates[offset : offset + QUEUE_BATCH_SIZE]
                    dispositions = await self._queue.enqueue_albums_with_disposition(
                        batch, now=now, scan_run_id=run_id, background=True
                    )
                    enqueued += sum(created for _, created in dispositions)
                await self._store.complete_grouping_context(
                    run_id,
                    str(context["root_id"]),
                    str(context["relative_directory"]),
                )
        return enqueued

    async def _regroup_large_context(
        self,
        run_id: str,
        root_id: str,
        relative_directory: str,
        *,
        now: float,
        checkpoint: Callable[[str, str], Awaitable[bool]] | None,
        frozen_policy_revision: str,
        _cohort_binding_candidate: str | None = None,
    ) -> int:
        enqueued = 0

        async def continue_grouping() -> bool:
            return checkpoint is None or await checkpoint(
                run_id, frozen_policy_revision
            )

        state = await self._store.get_grouping_staging_state(
            run_id, root_id, relative_directory
        )
        if state in {"pending", "tracks"}:
            while True:
                if not await continue_grouping():
                    return enqueued
                if _cohort_binding_candidate is not None:
                    from services.native.import_cohort_grouping_candidate import stage_cohort_track_page_candidate
                    resolver = self._cohort_resolver_getter()
                    if resolver.policy_revision != frozen_policy_revision:
                        raise StaleRevisionError('Frozen cohort scan policy changed.')
                    if await stage_cohort_track_page_candidate(store=self._store, filesystem=self._cohort_filesystem,
                            resolver=resolver, binding_id=_cohort_binding_candidate, run_id=run_id,
                            root_id=root_id, directory=relative_directory, limit=STAGING_BATCH_SIZE,
                            max_total_bytes=self._cohort_max_total_bytes, timeout_seconds=self._cohort_timeout_seconds):
                        break
                    continue
                rows, cursor = await self._store.get_grouping_context_track_page(
                    run_id,
                    root_id,
                    relative_directory,
                    limit=STAGING_BATCH_SIZE,
                )
                evidence = [
                    self._grouping_evidence(row, relative_directory)
                    for row in rows
                    if grouping_directory(str(row["relative_path"]))
                    == relative_directory
                ]
                await self._store.stage_grouping_track_page(
                    run_id,
                    root_id,
                    relative_directory,
                    evidence,
                    next_cursor=cursor,
                    exhausted=not rows,
                )
                if not rows:
                    break
            state = "tokens"
        if _cohort_binding_candidate is not None and not self._cohort_tokens_candidate_enabled:
            # Never enter unfenced native token/catalog/queue writes. This
            # first paged slice durably stages exact evidence, nothing more.
            raise ValidationError('Native paged cohort grouping is not integrated beyond validated track staging.')
        if state == "tokens":
            while True:
                if not await continue_grouping():
                    return enqueued
                if _cohort_binding_candidate is not None:
                    from services.native.import_cohort_grouping_candidate import prepare_cohort_token_page_candidate
                    resolver = self._cohort_resolver_getter()
                    if resolver.policy_revision != frozen_policy_revision:
                        raise StaleRevisionError('Frozen cohort scan policy changed.')
                    if await prepare_cohort_token_page_candidate(store=self._store, filesystem=self._cohort_filesystem,
                            resolver=resolver, binding_id=_cohort_binding_candidate, run_id=run_id,
                            root_id=root_id, directory=relative_directory, limit=STAGING_BATCH_SIZE,
                            max_total_bytes=self._cohort_max_total_bytes, timeout_seconds=self._cohort_timeout_seconds):
                        break
                    continue
                if await self._store.prepare_staged_grouping_tokens(
                    run_id,
                    root_id,
                    relative_directory,
                    limit=STAGING_BATCH_SIZE,
                ):
                    break
            state = "groups"
        if _cohort_binding_candidate is not None and not self._cohort_summaries_candidate_enabled:
            raise ValidationError('Native paged cohort summaries are not integrated beyond validated tokens.')
        if state == "groups":
            while True:
                if not await continue_grouping():
                    return enqueued
                if _cohort_binding_candidate is not None:
                    from services.native.import_cohort_grouping_candidate import summarize_cohort_page_candidate
                    resolver = self._cohort_resolver_getter()
                    if resolver.policy_revision != frozen_policy_revision:
                        raise StaleRevisionError('Frozen cohort scan policy changed.')
                    if await summarize_cohort_page_candidate(store=self._store, filesystem=self._cohort_filesystem,
                            resolver=resolver, binding_id=_cohort_binding_candidate, run_id=run_id,
                            root_id=root_id, directory=relative_directory, limit=STAGING_BATCH_SIZE,
                            max_total_bytes=self._cohort_max_total_bytes, timeout_seconds=self._cohort_timeout_seconds):
                        break
                    continue
                if await self._store.stage_grouping_summary_page(
                    run_id,
                    root_id,
                    relative_directory,
                    limit=STAGING_BATCH_SIZE,
                ):
                    break
            state = "continuity"
        if _cohort_binding_candidate is not None and not self._cohort_continuity_candidate_enabled:
            raise ValidationError('Native paged cohort continuity is not integrated beyond validated summaries.')
        if _cohort_binding_candidate is not None:
            async def retain(*, finish=False):
                from services.native.import_cohort_grouping_candidate import retain_cohort_continuity_candidate
                resolver = self._cohort_resolver_getter()
                if resolver.policy_revision != frozen_policy_revision:
                    raise StaleRevisionError('Frozen cohort scan policy changed.')
                return await retain_cohort_continuity_candidate(store=self._store, filesystem=self._cohort_filesystem,
                    resolver=resolver, binding_id=_cohort_binding_candidate, run_id=run_id,
                    root_id=root_id, directory=relative_directory, limit=STAGING_BATCH_SIZE,
                    max_total_bytes=self._cohort_max_total_bytes, timeout_seconds=self._cohort_timeout_seconds,
                    finish=finish)
            if state == 'continuity':
                while True:
                    if not await continue_grouping():
                        return enqueued
                    if not await retain():
                        break
                if not await continue_grouping():
                    return enqueued
                await retain(finish=True)
                state = 'albums'
            # This isolated path rejects wider graphs; retained-catalog
            # provisioning requires a separate default-OFF fenced option.
            if not self._cohort_provision_candidate_enabled:
                raise ValidationError('Native paged cohort provisioning is not integrated beyond validated continuity.')
            async def provision(*, finish=False):
                from services.native.import_cohort_grouping_candidate import provision_cohort_album_candidate
                resolver = self._cohort_resolver_getter()
                if resolver.policy_revision != frozen_policy_revision:
                    raise StaleRevisionError('Frozen cohort scan policy changed.')
                return await provision_cohort_album_candidate(store=self._store, filesystem=self._cohort_filesystem,
                    resolver=resolver, binding_id=_cohort_binding_candidate, run_id=run_id,
                    root_id=root_id, directory=relative_directory, now=now,
                    max_total_bytes=self._cohort_max_total_bytes, timeout_seconds=self._cohort_timeout_seconds, finish=finish)
            if state == 'albums':
                while True:
                    if not await continue_grouping():
                        return enqueued
                    if not await provision():
                        break
                if not await continue_grouping():
                    return enqueued
                await provision(finish=True)
                state = 'memberships'
            if not self._cohort_membership_candidate_enabled:
                raise ValidationError('Native paged cohort membership is not integrated beyond validated provisioning.')
            if state == 'memberships':
                from services.native.import_cohort_grouping_candidate import apply_cohort_membership_candidate
                while True:
                    if not await continue_grouping():
                        return enqueued
                    resolver = self._cohort_resolver_getter()
                    if resolver.policy_revision != frozen_policy_revision:
                        raise StaleRevisionError('Frozen cohort scan policy changed.')
                    if await apply_cohort_membership_candidate(store=self._store, filesystem=self._cohort_filesystem,
                            resolver=resolver, binding_id=_cohort_binding_candidate, run_id=run_id,
                            root_id=root_id, directory=relative_directory, limit=STAGING_BATCH_SIZE,
                            max_total_bytes=self._cohort_max_total_bytes, timeout_seconds=self._cohort_timeout_seconds):
                        break
                state = 'retirement'
            if not self._cohort_retirement_candidate_enabled:
                raise ValidationError('Native paged cohort retirement is not integrated beyond validated membership.')
            if state == 'retirement':
                from services.native.import_cohort_grouping_candidate import retire_cohort_album_candidate
                while True:
                    if not await continue_grouping():
                        return enqueued
                    resolver = self._cohort_resolver_getter()
                    if resolver.policy_revision != frozen_policy_revision:
                        raise StaleRevisionError('Frozen cohort scan policy changed.')
                    if await retire_cohort_album_candidate(store=self._store, filesystem=self._cohort_filesystem,
                            resolver=resolver, binding_id=_cohort_binding_candidate, run_id=run_id,
                            root_id=root_id, directory=relative_directory, limit=STAGING_BATCH_SIZE, now=now,
                            max_total_bytes=self._cohort_max_total_bytes, timeout_seconds=self._cohort_timeout_seconds):
                        break
                state = 'queue'
            if not self._cohort_queue_candidate_enabled:
                raise ValidationError('Native paged cohort queue is not integrated beyond validated retirement.')
            if state != 'queue':
                raise StaleRevisionError('Native private cohort queue phase changed.')
            if not await continue_grouping():
                return enqueued
            from services.native.import_cohort_grouping_candidate import complete_paged_cohort_candidate
            resolver = self._cohort_resolver_getter()
            if resolver.policy_revision != frozen_policy_revision:
                raise StaleRevisionError('Frozen cohort scan policy changed.')
            results = await complete_paged_cohort_candidate(store=self._store, filesystem=self._cohort_filesystem,
                resolver=resolver, binding_id=_cohort_binding_candidate, run_id=run_id,
                root_id=root_id, directory=relative_directory, now=now,
                max_total_bytes=self._cohort_max_total_bytes, timeout_seconds=self._cohort_timeout_seconds)
            return enqueued + sum(created for _, created in results)
        if state == "continuity":
            while True:
                if not await continue_grouping():
                    return enqueued
                if not await self._store.assign_isolated_grouping_continuity(
                    run_id,
                    root_id,
                    relative_directory,
                    limit=STAGING_BATCH_SIZE,
                ):
                    break
            while True:
                if not await continue_grouping():
                    return enqueued
                while await self._store.discard_resolved_grouping_edges(
                    run_id,
                    root_id,
                    relative_directory,
                    limit=STAGING_BATCH_SIZE,
                ):
                    if not await continue_grouping():
                        return enqueued
                edge = await self._store.get_next_grouping_edge(
                    run_id, root_id, relative_directory
                )
                if edge is None:
                    break
                component = await self._load_continuity_component(
                    run_id, root_id, relative_directory, edge
                )
                if component is None:
                    await self._store.apply_next_sparse_grouping_continuity(
                        run_id, root_id, relative_directory
                    )
                    continue
                await self._store.apply_grouping_continuity_component(
                    run_id,
                    root_id,
                    relative_directory,
                    list(component.values()),
                    self._continuity_assignments(list(component.values())),
                )
            await self._store.finish_grouping_continuity(
                run_id, root_id, relative_directory
            )
            state = "albums"
        if state == "albums":
            while True:
                if not await continue_grouping():
                    return enqueued
                groups = await self._store.get_unprovisioned_grouping_groups(
                    run_id,
                    root_id,
                    relative_directory,
                    limit=ARTIST_RESOLUTION_BATCH_SIZE,
                )
                if not groups:
                    break
                candidates = [
                    (
                        str(group["album_artist_name"]),
                        None,
                        "group",
                        grouping_artist_candidate_id(
                            str(group["album_artist_name"])
                        ),
                    )
                    for group in groups
                ]
                resolved = await self._store.resolve_or_create_local_artists(
                    candidates, now=now, background=True
                )
                for group in groups:
                    candidate_id = grouping_artist_candidate_id(
                        str(group["album_artist_name"])
                    )
                    group["local_artist_id"] = resolved[candidate_id][0]
                    group["local_album_id"] = group["retained_album_id"] or (
                        grouping_album_id(str(group["grouping_key"]))
                    )
                await self._store.provision_staged_grouping_groups(
                    run_id,
                    root_id,
                    relative_directory,
                    groups,
                    now=now,
                )
            await self._store.finish_grouping_album_provisioning(
                run_id, root_id, relative_directory
            )
            state = "memberships"
        if state == "memberships":
            while True:
                if not await continue_grouping():
                    return enqueued
                if await self._store.apply_staged_grouping_membership_page(
                    run_id,
                    root_id,
                    relative_directory,
                    limit=STAGING_BATCH_SIZE,
                ):
                    break
            state = "retirement"
        if state == "retirement":
            while True:
                if not await continue_grouping():
                    return enqueued
                if await self._store.retire_staged_grouping_albums(
                    run_id,
                    root_id,
                    relative_directory,
                    now=now,
                    limit=STAGING_BATCH_SIZE,
                ):
                    break
            state = "queue"
        if state == "queue":
            while True:
                if not await continue_grouping():
                    return enqueued
                page = await self._store.get_staged_grouping_queue_page(
                    run_id,
                    root_id,
                    relative_directory,
                    limit=QUEUE_BATCH_SIZE,
                )
                if not page:
                    break
                candidates = [
                    (
                        str(row["local_album_id"]),
                        str(row["input_revision"]),
                        str(row["trigger"]),
                    )
                    for row in page
                    if row.get("local_album_id") is not None
                ]
                if candidates:
                    dispositions = await self._queue.enqueue_albums_with_disposition(
                        candidates,
                        now=now,
                        scan_run_id=run_id,
                        grouping_context=(root_id, relative_directory),
                        queue_cursor=str(page[-1]["grouping_token"]),
                        background=True,
                    )
                    enqueued += sum(created for _, created in dispositions)
                else:
                    await self._store.advance_staged_grouping_queue(
                        run_id,
                        root_id,
                        relative_directory,
                        str(page[-1]["grouping_token"]),
                    )
            await self._store.complete_grouping_context(
                run_id, root_id, relative_directory
            )
        return enqueued

    @staticmethod
    def _grouping_evidence(
        row: dict[str, Any], relative_directory: str
    ) -> dict[str, Any]:
        track_id = str(row["id"])
        album_title = str(row["tag_album_title"] or "")
        album_artist = str(row["tag_album_artist_name"] or "")
        display_album_title = str(row.get("album_title") or "")
        display_album_artist = str(row.get("album_artist_name") or "")
        # Values exposed to the store fold logic (title_normalized below):
        # raw tag values, except parsed-keyed rows expose their display
        # (parsed) values so they sit inside the fold they keyed into.
        key_title = album_title
        key_artist = album_artist
        if bool(row["membership_locked"]):
            preliminary_key = f"manual:{row['local_album_id']}"
            reason = "MANUAL_MEMBERSHIP_RESTORED"
        elif (
            not album_title.strip()
            and resolve_provenance(
                row.get("album_title_provenance"),
                derive_album_provenance(album_title, display_album_title),
            )
            == "parsed"
            and display_album_title.strip()
        ):
            # M-05 staged mirror of the small-path untagged-merge widening
            # (LocalAlbumGrouper expanded block): parsed rows keyed "missing"
            # here because only raw tag columns fed the key. Key them as
            # tagged-or-parsed groups off their display (parsed) values with
            # the provisional reason, so the unchanged store SQL merges
            # agreeing parses exactly like the small path does - no store,
            # SQL, or schema change. Absent/placeholder rows keep "missing"
            # (strict numbered/no-collision absorb, unchanged), and
            # conflicting parses land on different folds and stay split. A
            # mixed tag+parsed group still reports the tagged reason: the
            # summary picks the alphabetical-min reason value.
            artist_partition = (
                ""
                if bool(row["is_compilation"])
                else normalize_group_value(display_album_artist)
            )
            preliminary_key = (
                f"tagged:{normalize_group_value(display_album_title)}:"
                f"{artist_partition}"
            )
            reason = PROVISIONAL_PARSED_GROUP
            key_title = display_album_title
            key_artist = display_album_artist
        elif album_title.strip():
            artist_partition = "" if bool(row["is_compilation"]) else (
                normalize_group_value(album_artist)
            )
            preliminary_key = (
                f"tagged:{normalize_group_value(album_title)}:{artist_partition}"
            )
            parent = str(PurePosixPath(str(row["relative_path"])).parent)
            reason = (
                "COMPATIBLE_DISC_DIRECTORIES"
                if parent != relative_directory
                else "CONSISTENT_ALBUM_TAGS"
            )
        else:
            preliminary_key = "missing"
            reason = "MISSING_ALBUM_TAGS"
        return {
            "local_track_id": track_id,
            "preliminary_key": preliminary_key,
            "title": key_title,
            "title_normalized": normalize_group_value(key_title),
            "album_artist_name": key_artist,
            "album_artist_normalized": normalize_group_value(key_artist),
            "track_number": int(row["track_number"] or 0),
            "old_album_id": str(row["local_album_id"]),
            "album_created_at": float(row["album_created_at"]),
            "reason_code": reason,
        }

    async def _load_continuity_component(
        self,
        run_id: str,
        root_id: str,
        relative_directory: str,
        seed: dict[str, Any],
    ) -> dict[tuple[str, str], dict[str, Any]] | None:
        edges: dict[tuple[str, str], dict[str, Any]] = {}
        old_pending = {str(seed["old_album_id"])}
        token_pending: set[str] = set()
        old_seen: set[str] = set()
        token_seen: set[str] = set()
        while old_pending or token_pending:
            if old_pending:
                batch = sorted(old_pending)[:STAGING_BATCH_SIZE]
                old_pending.difference_update(batch)
                old_seen.update(batch)
                rows = await self._store.get_grouping_edges_for_old_albums(
                    run_id,
                    root_id,
                    relative_directory,
                    batch,
                    limit=CONTINUITY_COMPONENT_EDGE_LIMIT + 1,
                )
                if len(rows) > CONTINUITY_COMPONENT_EDGE_LIMIT:
                    return None
                for row in rows:
                    key = (str(row["old_album_id"]), str(row["grouping_token"]))
                    edges[key] = row
                    token = key[1]
                    if token not in token_seen:
                        token_pending.add(token)
                if len(edges) > CONTINUITY_COMPONENT_EDGE_LIMIT:
                    return None
            if token_pending:
                batch = sorted(token_pending)[:STAGING_BATCH_SIZE]
                token_pending.difference_update(batch)
                token_seen.update(batch)
                rows = await self._store.get_grouping_edges_for_tokens(
                    run_id,
                    root_id,
                    relative_directory,
                    batch,
                    limit=CONTINUITY_COMPONENT_EDGE_LIMIT + 1,
                )
                if len(rows) > CONTINUITY_COMPONENT_EDGE_LIMIT:
                    return None
                for row in rows:
                    key = (str(row["old_album_id"]), str(row["grouping_token"]))
                    edges[key] = row
                    album_id = key[0]
                    if album_id not in old_seen:
                        old_pending.add(album_id)
                if len(edges) > CONTINUITY_COMPONENT_EDGE_LIMIT:
                    return None
        return edges

    @staticmethod
    def _continuity_assignments(
        edges: list[dict[str, Any]],
    ) -> dict[str, tuple[str, str]]:
        old_created = {
            str(edge["old_album_id"]): float(edge["album_created_at"])
            for edge in edges
        }
        grouping_keys = {
            str(edge["grouping_token"]): str(edge["grouping_key"])
            for edge in edges
        }
        old_ids = sorted(old_created, key=lambda value: (old_created[value], value))
        tokens = sorted(grouping_keys, key=lambda value: grouping_keys[value])
        old_index = {value: index for index, value in enumerate(old_ids)}
        token_index = {value: index for index, value in enumerate(tokens)}
        size = max(len(old_ids), len(tokens))
        weights = [[0] * size for _ in range(size)]
        old_neighbors: dict[int, set[int]] = {}
        token_neighbors: dict[int, set[int]] = {}
        for edge in edges:
            row = old_index[str(edge["old_album_id"])]
            column = token_index[str(edge["grouping_token"])]
            weights[row][column] = int(edge["overlap_count"])
            old_neighbors.setdefault(row, set()).add(column)
            token_neighbors.setdefault(column, set()).add(row)
        maximum = max((int(edge["overlap_count"]) for edge in edges), default=0)
        assignment = _hungarian_min(
            [[maximum - value for value in row] for row in weights]
        )
        result: dict[str, tuple[str, str]] = {}
        for row_index, column_index in enumerate(assignment[: len(old_ids)]):
            if column_index >= len(tokens) or weights[row_index][column_index] == 0:
                continue
            value = weights[row_index][column_index]
            old_best = max(
                weights[row_index][candidate]
                for candidate in old_neighbors[row_index]
            )
            new_best = max(
                weights[candidate][column_index]
                for candidate in token_neighbors[column_index]
            )
            tied = (
                sum(
                    weights[row_index][candidate] == old_best
                    for candidate in old_neighbors[row_index]
                )
                > 1
                or sum(
                    weights[candidate][column_index]
                    == new_best
                    for candidate in token_neighbors[column_index]
                )
                > 1
            )
            result[tokens[column_index]] = (
                old_ids[row_index],
                "CONTINUITY_TIE_BROKEN" if tied else "MAXIMUM_TRACK_OVERLAP",
            )
        return result
