"""Download/import library contract backed only by the target catalog store."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sqlite3
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec

from models.local_catalog import (
    LocalAlbum,
    LocalArtist,
    LocalArtistCredit,
    LocalTrack,
    LocalTrackGenre,
)
from models.library_management import (
    BUNDLE_BLOCKED,
    FIELD_UNSUPPORTED_BY_FORMAT,
    PATH_COLLISION_DIFFERENT,
    POLICY_CHANGED,
    ROOT_UNAVAILABLE,
    LibraryManagementImportBundle,
    LibraryManagementImportResult,
    LibraryManagementPublishedImportFile,
)
from models.library_work import ScannedTrackWrite
from services.local_files_service import AUDIO_EXTENSIONS
from services.native.identification_revisions import album_input_revisions
from services.native.file_revision import revision_from_stat
from services.native.local_album_grouper import (
    grouping_directory,
    normalize_group_value,
)
from services.native.local_album_grouping_service import (
    grouping_album_id,
    grouping_artist_candidate_id,
)
from core.exceptions import (
    AudioWriteError,
    AutomaticManagementHoldError,
    ConflictError,
    ConfigurationError,
    LibraryManagementDestinationConflictError,
    LibraryManagementPolicyChangedError,
    StaleRevisionError,
    ValidationError,
)
from repositories.musicbrainz_base import MbSourceContext, capture_mb_source_context

if TYPE_CHECKING:
    from models.library_management_canonical import CanonicalReleaseDocument
    from infrastructure.persistence.native_library_store import NativeLibraryStore
    from models.audio import AudioInfo, AudioTag
    from services.native.identification_queue_service import IdentificationQueueService
    from services.native.automatic_import_management_service import (
        AutomaticImportManagementService,
    )
    from services.native.library_filesystem_coordinator import (
        LibraryFilesystemCoordinator,
    )
    from services.native.library_management_publisher import LibraryManagementPublisher
    from services.native.library_policy_resolver import LibraryPolicyResolver

_TRACK_NAMESPACE = uuid.UUID("1a8da1ca-9ca8-4bc5-bdb7-512fcf58ef67")
logger = logging.getLogger(__name__)


def _caused_by_os_error(error: BaseException) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, OSError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


class _CandidatePolicyScope:
    def __init__(self) -> None:
        self.task = asyncio.current_task()
        self.active = True


class TargetImportLibraryService:
    def __init__(
        self,
        store: "NativeLibraryStore",
        resolver_getter: "Callable[[], LibraryPolicyResolver]",
        identification_queue: "IdentificationQueueService",
        *,
        policy_transition_lock: asyncio.Lock | None = None,
        filesystem_coordinator: "LibraryFilesystemCoordinator | None" = None,
        management_publisher: "LibraryManagementPublisher | None" = None,
        automatic_management: "AutomaticImportManagementService | None" = None,
        _cohort_import_candidate_enabled: bool = False,
    ) -> None:
        self._store = store
        self._resolver_getter = resolver_getter
        self._queue = identification_queue
        self._policy_transition_lock = policy_transition_lock or asyncio.Lock()
        self._filesystem = filesystem_coordinator
        self._management_publisher = management_publisher
        self._automatic_management = automatic_management
        self._cohort_import_candidate_enabled = _cohort_import_candidate_enabled
        self._candidate_policy_scope: ContextVar[_CandidatePolicyScope | None] = ContextVar('cohort_policy_scope', default=None)

    @asynccontextmanager
    async def _policy_scope_candidate(self):
        """Only same-task lexical nesting borrows the actual held policy lock."""
        owner = self._candidate_policy_scope.get()
        if owner is not None and owner.active and owner.task is asyncio.current_task():
            yield
            return
        async with self._policy_transition_lock:
            owner = _CandidatePolicyScope()
            token = self._candidate_policy_scope.set(owner)
            try:
                yield
            finally:
                owner.active = False
                self._candidate_policy_scope.reset(token)

    def is_configured(self) -> bool:
        return bool(self._resolver_getter().settings.library_roots)

    async def publish_import_bundle(
        self, bundle: LibraryManagementImportBundle
    ) -> LibraryManagementImportResult:
        if self._management_publisher is None:
            raise RuntimeError("The staged import publisher is not configured.")

        source_context = capture_mb_source_context()
        if self._automatic_management is not None:
            bundle = await self._automatic_management.prepare(
                bundle,
                source_context=source_context,
            )

        async def commit(
            bundle_id: str,
            files: tuple[LibraryManagementPublishedImportFile, ...],
            commit_source_context: MbSourceContext | None,
        ) -> tuple[str, ...]:
            async with self._policy_scope_candidate():
                return await self._commit_published_import_bundle(
                    bundle_id,
                    files,
                    expected_policy_revision=bundle.policy_revision,
                    source_context=commit_source_context,
                )

        automatic = any(value.pinned_profile is not None for value in bundle.files)
        try:
            return await self._management_publisher.publish_import_bundle(
                bundle,
                commit,
                source_context=source_context if automatic else None,
            )
        except AutomaticManagementHoldError:
            raise
        except LibraryManagementPolicyChangedError as error:
            if not automatic:
                raise
            raise AutomaticManagementHoldError(
                POLICY_CHANGED,
                "The Library Management policy changed during publication. Retry this import.",
            ) from error
        except LibraryManagementDestinationConflictError as error:
            if not automatic:
                raise
            raise AutomaticManagementHoldError(
                PATH_COLLISION_DIFFERENT,
                "A planned destination is occupied by different content. Nothing was overwritten.",
            ) from error
        except StaleRevisionError as error:
            if not automatic:
                raise
            logger.warning(
                "Automatic import publication was blocked by stale durable evidence",
                exc_info=True,
            )
            raise AutomaticManagementHoldError(
                BUNDLE_BLOCKED,
                "The secured publication evidence changed while the organizer was verifying it. Nothing was overwritten.",
            ) from error
        except ConflictError as error:
            if not automatic:
                raise
            logger.warning(
                "Automatic import publication was blocked by conflicting durable evidence",
                exc_info=True,
            )
            raise AutomaticManagementHoldError(
                BUNDLE_BLOCKED,
                "The durable publication evidence no longer agrees with its journal. Nothing was overwritten.",
            ) from error
        except AudioWriteError as error:
            if not automatic:
                raise
            logger.warning(
                "Automatic import staged audio verification failed",
                exc_info=True,
            )
            if _caused_by_os_error(error):
                raise AutomaticManagementHoldError(
                    ROOT_UNAVAILABLE,
                    "The library destination interrupted the staged write. The secured files will be retried automatically.",
                ) from error
            raise AutomaticManagementHoldError(
                FIELD_UNSUPPORTED_BY_FORMAT,
                "The staged audio write did not match its pinned format contract.",
            ) from error
        except ValidationError as error:
            if not automatic:
                raise
            logger.warning(
                "Automatic import publication contract failed validation",
                exc_info=True,
            )
            raise AutomaticManagementHoldError(
                BUNDLE_BLOCKED,
                "The pinned publication contract no longer passes its safety checks. Nothing was overwritten.",
            ) from error
        except OSError as error:
            if not automatic:
                raise
            logger.warning(
                "Automatic import publication hit a filesystem I/O failure",
                exc_info=True,
            )
            raise AutomaticManagementHoldError(
                ROOT_UNAVAILABLE,
                "The library destination is unavailable. Restore it and retry.",
            ) from error
        except sqlite3.Error as error:
            if not automatic:
                raise
            raise AutomaticManagementHoldError(
                BUNDLE_BLOCKED,
                "The automatic import could not be committed. Retry it later.",
            ) from error

    @staticmethod
    def _root_for(path: Path, resolver: "LibraryPolicyResolver"):  # noqa: ANN205 - typed settings root variant
        resolved = path.resolve()
        for root in resolver.settings.library_roots:
            root_path = Path(root.path).resolve()
            try:
                relative = resolved.relative_to(root_path)
            except ValueError:
                continue
            return root, relative.as_posix()
        raise ValueError("Imported file is outside every configured library root.")

    async def upsert_file(
        self,
        audio_path: Path,
        tag: "AudioTag",
        info: "AudioInfo",
        *,
        release_group_mbid: str | None,
        release_mbid: str | None = None,
        recording_mbid: str | None = None,
        confidence: float = 1.0,
        source: str = "download",
        download_task_id: str | None = None,
        source_path: str | None = None,
        file_mtime: float | None = None,
    ) -> str:
        async with self._policy_transition_lock:
            for attempt in range(2):
                resolver = self._resolver_getter()
                try:
                    return await self._upsert_file_once(
                        audio_path,
                        tag,
                        info,
                        resolver=resolver,
                        release_group_mbid=release_group_mbid,
                        release_mbid=release_mbid,
                        recording_mbid=recording_mbid,
                        confidence=confidence,
                        source=source,
                        download_task_id=download_task_id,
                        source_path=source_path,
                        file_mtime=file_mtime,
                    )
                except StaleRevisionError:
                    if attempt == 1:
                        raise
            raise AssertionError("Policy retry loop did not return")

    async def _import_reuse_album_id(
        self, release_group_mbid: str | None
    ) -> str | None:
        """Issue #301: reuse the owning album for a provider release group.

        Returns the oldest active owner, or None when nothing owns the RG
        yet (caller falls back to the grouping-key album id). Never raises
        on multiple owners — ambiguity resolves to the oldest, never a mint.
        """
        if not release_group_mbid:
            return None
        return await self._store.find_import_reuse_album_id(release_group_mbid)

    async def _upsert_file_once(
        self,
        audio_path: Path,
        tag: "AudioTag",
        info: "AudioInfo",
        *,
        resolver: "LibraryPolicyResolver",
        release_group_mbid: str | None,
        release_mbid: str | None,
        recording_mbid: str | None,
        confidence: float,
        source: str,
        download_task_id: str | None,
        source_path: str | None,
        file_mtime: float | None,
    ) -> str:
        del confidence
        root, relative_path = self._root_for(audio_path, resolver)
        stat = await asyncio.to_thread(audio_path.stat)
        resolution = resolver.resolve(audio_path)
        if resolution is None:
            raise ValueError("Imported file is outside every configured library root.")
        policy = resolution.policy
        now = file_mtime if file_mtime is not None else stat.st_mtime
        album_title = tag.album.strip() or audio_path.parent.name or "Unknown Album"
        album_artist = (tag.album_artist or tag.artist or "Unknown Artist").strip()
        artist_id = await self._store.find_target_artist_by_name(album_artist)
        artist_id = artist_id or grouping_artist_candidate_id(album_artist)
        directory = grouping_directory(relative_path)
        grouping_key = (
            f"{root.id}:{directory}:{normalize_group_value(album_title)}:"
            f"{normalize_group_value(album_artist)}"
        )
        album_id = await self._import_reuse_album_id(
            release_group_mbid
        ) or grouping_album_id(grouping_key)
        existing = await self._store.get_target_track_by_path(str(audio_path))
        track_id = (
            str(existing["id"])
            if existing is not None
            else str(uuid.uuid5(_TRACK_NAMESPACE, f"{root.id}:{relative_path}"))
        )
        tag_revision = hashlib.sha256(msgspec.json.encode(tag)).hexdigest()
        artist = LocalArtist(
            id=artist_id,
            display_name=album_artist,
            folded_name=album_artist.casefold(),
            normalized_name=normalize_group_value(album_artist),
            sort_name=tag.album_artist_sort,
            kind="group",
            created_at=now,
            updated_at=now,
        )
        album = LocalAlbum(
            id=album_id,
            root_id=root.id,
            grouping_key=grouping_key,
            title=album_title,
            album_artist_id=artist_id,
            album_artist_name=album_artist,
            album_artist_sort_name=tag.album_artist_sort,
            year=tag.year,
            original_release_date=tag.original_release_date,
            primary_genre=tag.genre,
            is_compilation=tag.compilation,
            grouping_source="automatic",
            created_at=now,
            updated_at=now,
        )
        track = LocalTrack(
            id=track_id,
            local_album_id=album_id,
            root_id=root.id,
            file_path=str(audio_path),
            relative_path=relative_path,
            path_hash=hashlib.sha256(relative_path.encode()).hexdigest(),
            file_size_bytes=stat.st_size,
            file_mtime_ns=stat.st_mtime_ns,
            stat_revision=revision_from_stat(stat),
            tag_revision=tag_revision,
            tags_read_at=now,
            title=tag.title.strip() or audio_path.stem,
            artist_name=tag.artist or album_artist,
            album_title=album_title,
            album_artist_name=album_artist,
            tag_album_title=tag.album,
            tag_album_artist_name=tag.album_artist,
            disc_number=tag.disc_number,
            track_number=tag.track_number,
            year=tag.year,
            genre=tag.genre,
            title_sort=tag.title_sort,
            artist_sort=tag.artist_sort,
            album_sort=tag.album_sort,
            album_artist_sort=tag.album_artist_sort,
            disc_subtitle=tag.disc_subtitle,
            release_type=tag.release_type,
            is_compilation=tag.compilation,
            embedded_release_group_mbid=(
                release_group_mbid or tag.musicbrainz_release_group_id
            ),
            embedded_release_mbid=release_mbid or tag.musicbrainz_release_id,
            embedded_recording_mbid=recording_mbid or tag.musicbrainz_recording_id,
            embedded_release_track_mbid=tag.musicbrainz_release_track_id,
            embedded_artist_mbid=tag.musicbrainz_artist_id,
            embedded_album_artist_mbid=tag.musicbrainz_album_artist_id,
            duration_seconds=info.duration_seconds,
            file_format=info.file_format,
            bit_rate=info.bitrate,
            sample_rate=info.sample_rate,
            bit_depth=info.bit_depth,
            channels=info.channels,
            replaygain_track_gain=tag.replaygain_track_gain,
            replaygain_album_gain=tag.replaygain_album_gain,
            replaygain_track_peak=tag.replaygain_track_peak,
            replaygain_album_peak=tag.replaygain_album_peak,
            ingest_source=source,
            download_task_id=download_task_id,
            source_path=source_path,
            imported_at=now,
            desired_policy_revision=resolver.policy_revision,
            applied_policy_revision=resolver.policy_revision,
            applied_policy=policy,
            availability="excluded" if policy == "excluded" else "indexed",
            excluded_at=now if policy == "excluded" else None,
        )
        persisted_id, _ = await self._store.upsert_scanned_track(
            artist=artist,
            album=album,
            track=track,
            credit=LocalArtistCredit(
                local_artist_id=artist_id,
                position=0,
                credited_name=album_artist,
            ),
            genres=[
                LocalTrackGenre(
                    local_track_id=track_id,
                    position=position,
                    name=value,
                    folded_name=value.casefold(),
                    source="local",
                    source_document_revision=tag_revision,
                )
                for position, value in enumerate(
                    tag.genres or ([tag.genre] if tag.genre else [])
                )
            ],
            expected_policy_revision=resolver.policy_revision,
        )
        embedded_identity = bool(
            release_group_mbid
            or release_mbid
            or tag.musicbrainz_release_group_id
            or tag.musicbrainz_release_id
        )
        if policy == "automatic" or (policy == "local_metadata" and embedded_identity):
            rows = await self._store.get_target_album_tracks(album_id)
            revisions = album_input_revisions(rows)
            await self._queue.enqueue_album(
                album_id,
                input_revision=":".join(revisions),
                kind="automatic" if policy == "automatic" else "post_processing",
                now=now,
                expected_policy_revision=resolver.policy_revision,
            )
        return persisted_id

    async def _commit_published_import_bundle(
        self,
        bundle_id: str,
        files: tuple[LibraryManagementPublishedImportFile, ...],
        *,
        expected_policy_revision: str,
        source_context: MbSourceContext | None = None,
    ) -> tuple[str, ...]:
        if self._cohort_import_candidate_enabled:
            return await self._commit_cohort_import_candidate_callback(
                bundle_id, files, expected_policy_revision=expected_policy_revision, source_context=source_context)
        resolver = self._resolver_getter()
        if resolver.policy_revision != expected_policy_revision:
            raise LibraryManagementPolicyChangedError(
                "Library policy changed before import commit."
            )
        writes: list[tuple[int, ScannedTrackWrite]] = []
        replacements: dict[int, str] = {}
        recycle_track_ids: dict[int, str] = {}
        for value in files:
            request = value.request
            if request.conversion_recycle_only:
                if request.replacement_local_track_id is None:
                    raise ValidationError(
                        "A conversion recycle publication lost its catalog track."
                    )
                recycle_track_ids[request.ordinal] = request.replacement_local_track_id
                continue
            write = await self._build_published_import_write(
                Path(value.destination_path),
                value.tag,
                value.info,
                resolver=resolver,
                release_group_mbid=request.release_group_mbid,
                release_mbid=request.release_mbid,
                recording_mbid=request.recording_mbid,
                source=request.source,
                download_task_id=request.download_task_id,
                source_path=request.source_path,
                file_mtime=request.file_mtime,
            )
            if bundle_id and request.source == "edition_conversion":
                record = await self._store.get_library_management_import_bundle(
                    bundle_id
                )
                if record is None:
                    raise StaleRevisionError(
                        "The conversion publication bundle disappeared."
                    )
                sealed = msgspec.json.decode(
                    record.request_json.encode(), type=LibraryManagementImportBundle
                )
                if sealed.conversion_local_album_id is None:
                    raise ValidationError(
                        "The conversion publication lost its target album."
                    )
                write = msgspec.structs.replace(
                    write,
                    album=msgspec.structs.replace(
                        write.album, id=sealed.conversion_local_album_id
                    ),
                    track=msgspec.structs.replace(
                        write.track, local_album_id=sealed.conversion_local_album_id
                    ),
                )
            writes.append((request.ordinal, write))
            if request.replacement_local_track_id is not None:
                replacements[request.ordinal] = request.replacement_local_track_id
        track_ids = await self._store.commit_library_management_import_bundle(
            bundle_id,
            writes=writes,
            replacement_track_ids=replacements,
            recycle_track_ids=recycle_track_ids,
            requests_by_ordinal={
                value.request.ordinal: value.request for value in files
            },
            automatic_requests={
                value.request.ordinal: value.request
                for value in files
                if value.request.pinned_profile is not None
                and not value.request.conversion_recycle_only
            },
            expected_policy_revision=expected_policy_revision,
            result_paths_by_ordinal={
                value.request.ordinal: value.destination_path for value in files
            },
            updated_at=time.time(),
            source_context=source_context,
        )
        automatic_ordinals = {
            value.request.ordinal
            for value in files
            if value.request.pinned_profile is not None
        }
        queue_writes = {
            write.album.id: write
            for ordinal, write in writes
            if ordinal not in automatic_ordinals
        }
        for write in queue_writes.values():
            embedded_identity = bool(
                write.track.embedded_release_group_mbid
                or write.track.embedded_release_mbid
            )
            if write.track.applied_policy == "automatic" or (
                write.track.applied_policy == "local_metadata" and embedded_identity
            ):
                rows = await self._store.get_target_album_tracks(write.album.id)
                revisions = album_input_revisions(rows)
                await self._queue.enqueue_album(
                    write.album.id,
                    input_revision=":".join(revisions),
                    kind=(
                        "automatic"
                        if write.track.applied_policy == "automatic"
                        else "post_processing"
                    ),
                    now=write.track.imported_at,
                    expected_policy_revision=expected_policy_revision,
                )
        return track_ids

    async def _commit_cohort_import_candidate_callback(
        self, bundle_id: str, files: tuple[LibraryManagementPublishedImportFile, ...],
        *, expected_policy_revision: str, source_context: MbSourceContext | None,
    ) -> tuple[str, ...]:
        """Default-OFF candidate callback; all original automatic guards run.

        Must execute inside the real publisher's existing root write lease.
        There is no standalone lease-acquisition fallback or user bypass flag.
        """
        from infrastructure.persistence.import_cohort_binding import CohortSnapshotSeal, ImportCohortCommitCandidate
        from models.library_management_canonical import CanonicalReleaseDocument

        async with self._policy_scope_candidate():
            roots = {value.request.destination_root_id for value in files}
            if (source_context is None or self._filesystem is None or len(roots) != 1
                    or not self._filesystem._cohort_writer_is_held(next(iter(roots)))):
                raise ConfigurationError('Cohort commit requires publisher-owned root lease and native source context.')
            record = await self._store.get_library_management_import_bundle(bundle_id)
            if (record is None or record.state != 'publishing'
                    or hashlib.sha256(record.request_json.encode()).hexdigest() != record.request_hash):
                raise StaleRevisionError('Cohort publication bundle is not sealed and publishing.')
            bundle = msgspec.json.decode(record.request_json.encode(), type=LibraryManagementImportBundle)
            if bundle.policy_revision != expected_policy_revision:
                raise LibraryManagementPolicyChangedError('Cohort publication policy changed.')
            snapshot_ids = {value.metadata_snapshot_id for value in bundle.files}
            if (len(snapshot_ids) != 1 or None in snapshot_ids
                    or any(value.pinned_profile is None for value in bundle.files)):
                raise ValidationError('Cohort commit requires one complete pinned native canonical projection.')
            correlated = await self._store._get_cohort_source_snapshot_candidate(next(iter(snapshot_ids)), source_context)
            if correlated is None:
                raise StaleRevisionError('No matching native canonical response source observation.')
            snapshot, proof_id = correlated
            canonical = msgspec.json.decode(snapshot.canonical_payload_json.encode(), type=CanonicalReleaseDocument)
            writes, fingerprints, parent = await self._prepare_cohort_import_writes_candidate(
                bundle, files, canonical, max_total_bytes=64 * 1024**3, timeout_seconds=120)
            candidate = ImportCohortCommitCandidate(
                binding_id=f'import-cohort:{bundle_id}', root_id=next(iter(roots)),
                root_path=next(root.path for root in self._resolver_getter().settings.library_roots if root.id in roots),
                album_parent=parent,
                canonical_map=tuple((value.release_track_mbid, value.recording_mbid, value.medium_position,
                                     value.release_track_position) for value in bundle.files),
                fingerprints=tuple((int(ordinal), digest) for ordinal, digest in fingerprints.items()),
                snapshot=CohortSnapshotSeal(snapshot_id=snapshot.id, input_hash=snapshot.input_hash,
                    payload_sha256=snapshot.payload_sha256, source_proof_id=proof_id))
            return await self._store.commit_library_management_import_bundle(
                bundle_id, writes=writes, replacement_track_ids={}, recycle_track_ids={},
                requests_by_ordinal={value.ordinal: value for value in bundle.files},
                automatic_requests={value.ordinal: value for value in bundle.files},
                expected_policy_revision=expected_policy_revision,
                result_paths_by_ordinal={value.request.ordinal: value.destination_path for value in files},
                updated_at=time.time(), source_context=source_context, _cohort_candidate=candidate)

    async def _prepare_cohort_import_writes_candidate(
        self,
        bundle: LibraryManagementImportBundle,
        files: tuple[LibraryManagementPublishedImportFile, ...],
        canonical: "CanonicalReleaseDocument",
        *,
        max_total_bytes: int,
        timeout_seconds: float,
    ) -> tuple[list[tuple[int, ScannedTrackWrite]], dict[str, str], str]:
        """LOCAL candidate only: prepare one complete NEW album before commit.

        No publication caller uses this method. It never writes the catalog or
        invokes commit/cleanup. Canonical provenance must still be supplied by
        the native provider and pinned across the eventual atomic commit.
        Coordinator/policy leases end on return; this is NOT a commit token.
        """
        from pathlib import PurePosixPath

        from services.native.import_cohort_observation import observe_cohort_bytes
        from services.native.library_management_publisher import LibraryManagementPublisher

        LibraryManagementPublisher._validate_import_bundle(bundle)
        if self._filesystem is None or bundle.origin != "acquisition":
            raise ValidationError("Cohort preparation needs native coordination and acquisition scope.")
        requests = {value.ordinal: value for value in bundle.files}
        if len(files) != len(requests) or {value.request.ordinal for value in files} != set(requests):
            raise ValidationError("A complete published cohort is required.")
        if canonical.identity_kind != "exact_release" or not canonical.media or canonical.total_discs != len(canonical.media):
            raise ValidationError("A complete exact-release canonical document is required.")
        canonical_map = []
        for medium in canonical.media:
            if medium.track_count != len(medium.tracks) or medium.track_count < 1:
                raise ValidationError("Canonical projection omits release tracks.")
            for track in medium.tracks:
                identity = track.identifiers
                if (track.disc_number != medium.position or track.total_tracks != medium.track_count
                        or track.total_discs != canonical.total_discs
                        or identity.release_mbid != canonical.identifiers.release_mbid
                        or identity.release_group_mbid != canonical.identifiers.release_group_mbid):
                    raise ValidationError("Canonical medium or release identity disagrees.")
                canonical_map.append((identity.release_track_mbid, identity.recording_mbid,
                                      medium.position, track.track_number))
        wanted = [(v.release_track_mbid, v.recording_mbid, v.medium_position, v.release_track_position)
                  for v in bundle.files]
        if (any(not rt or not rec or type(d) is not int or type(p) is not int or d < 1 or p < 1 for rt, rec, d, p in wanted)
                or len({v[0] for v in wanted}) != len(wanted)
                or len({v[2:] for v in wanted}) != len(wanted)
                or sorted(wanted) != sorted(canonical_map)):
            raise ValidationError("Complete distinct canonical recording mapping required.")
        if any(not v.authoritative_mapping or v.reviewed_recording_identity
               or v.source != "download" or v.replacement_local_track_id is not None
               or v.conversion_recycle_only or not v.download_task_id
               or (v.release_mbid, v.release_group_mbid) != (
                   canonical.identifiers.release_mbid, canonical.identifiers.release_group_mbid)
               for v in bundle.files):
            raise ValidationError("Only a complete managed new acquisition is supported.")
        if len({v.download_task_id for v in bundle.files}) != 1:
            raise ValidationError("Mixed acquisition sources are not a cohort.")
        root_ids = {v.destination_root_id for v in bundle.files}
        if len(root_ids) != 1:
            raise ValidationError("One configured root is required.")
        root_id = next(iter(root_ids))
        async with self._policy_scope_candidate(), self._filesystem._read_for_cohort_candidate(root_id):
            resolver = self._resolver_getter()
            if resolver.policy_revision != bundle.policy_revision:
                raise LibraryManagementPolicyChangedError("Cohort policy changed.")
            roots = [root for root in resolver.settings.library_roots if root.id == root_id]
            if len(roots) != 1 or roots[0].policy != "automatic":
                raise ValidationError("Configured automatic root required.")
            root = Path(roots[0].path)
            relative_paths = [PurePosixPath(v.destination_relative_path) for v in bundle.files]
            common = os.path.commonpath([str(path.parent) for path in relative_paths])
            parent = PurePosixPath(common)
            if len(parent.parts) < 2 or str(parent) in (".", "/"):
                raise ValidationError("A specific artist/album parent is required.")
            contexts = set()
            disc_directories = {}
            observations = {}
            for value in files:
                request = requests[value.request.ordinal]
                if msgspec.json.encode(request) != msgspec.json.encode(value.request):
                    raise ValidationError("Published request differs from sealed request.")
                relative = PurePosixPath(request.destination_relative_path)
                tail = relative.relative_to(parent)
                if len(tail.parts) not in (1, 2):
                    raise ValidationError("Only flat or one-level medium layout is supported.")
                if len(tail.parts) == 2:
                    disc_directories.setdefault(tail.parts[0], set()).add(request.medium_position)
                audio_path = root / request.destination_relative_path
                if str(audio_path) != value.destination_path:
                    raise ValidationError("Published path escaped configured root scope.")
                resolution = resolver.resolve(audio_path)
                if resolution is None or resolution.root_id != root_id or resolution.policy != "automatic":
                    raise ValidationError("Cohort member has a different effective policy.")
                tag = value.tag
                if ((tag.musicbrainz_release_track_id, tag.musicbrainz_recording_id,
                     tag.disc_number, tag.track_number) != (
                         request.release_track_mbid, request.recording_mbid,
                         request.medium_position, request.release_track_position)
                        or (tag.musicbrainz_release_id, tag.musicbrainz_release_group_id) != (
                            request.release_mbid, request.release_group_mbid)):
                    raise ValidationError("Published tags disagree with canonical mapping.")
                context = (normalize_group_value(tag.album), normalize_group_value(tag.album_artist or ""))
                if not all(context):
                    raise ValidationError("Explicit album tags required.")
                contexts.add(context)
                observations[str(request.ordinal)] = (
                    request.destination_relative_path, revision_from_stat(audio_path.stat()))
            if len(contexts) != 1 or any(len(value) != 1 for value in disc_directories.values()):
                raise ValidationError("Conflicting album or medium contexts.")
            if len({next(iter(value)) for value in disc_directories.values()}) != len(disc_directories):
                raise ValidationError("A medium is spread across unrelated directories.")
            if await self._import_reuse_album_id(canonical.identifiers.release_group_mbid):
                raise ValidationError("Existing ownership requires separate native reuse validation.")
            fingerprints = await asyncio.to_thread(
                observe_cohort_bytes, root_path=str(root), members=observations,
                max_total_bytes=max_total_bytes, timeout_seconds=timeout_seconds)
            if any(fingerprints[str(v.request.ordinal)] != v.staged_fingerprint for v in files):
                raise ValidationError("Published bytes differ from staged journal evidence.")
            title, artist = next(iter(contexts))
            grouping_key = f"{root_id}:{parent}:{title}:{artist}"
            album_id = grouping_album_id(grouping_key)
            if await self._store.get_local_album(album_id) is not None:
                raise ValidationError("Candidate cannot adopt an existing album, including manual membership.")
            writes = []
            for value in sorted(files, key=lambda value: value.request.ordinal):
                request = value.request
                write = await self._build_published_import_write(
                    Path(value.destination_path), value.tag, value.info, resolver=resolver,
                    release_group_mbid=request.release_group_mbid, release_mbid=request.release_mbid,
                    recording_mbid=request.recording_mbid, source=request.source,
                    download_task_id=request.download_task_id, source_path=request.source_path,
                    file_mtime=request.file_mtime)
                if write.comparison_result != "new":
                    raise ValidationError("Candidate cannot replace an existing catalog member.")
                original_key = (
                    f"{root_id}:{grouping_directory(write.relative_path)}:"
                    f"{normalize_group_value(write.album.title)}:"
                    f"{normalize_group_value(write.album.album_artist_name)}"
                )
                if write.album.id != grouping_album_id(original_key):
                    raise ValidationError("Album ownership changed during candidate preparation.")
                write = msgspec.structs.replace(
                    write, album=msgspec.structs.replace(write.album, id=album_id, grouping_key=grouping_key),
                    track=msgspec.structs.replace(write.track, local_album_id=album_id), grouping_context=str(parent))
                writes.append((request.ordinal, write))
            final_fingerprints = await asyncio.to_thread(
                observe_cohort_bytes, root_path=str(root),
                members={str(ordinal): (write.relative_path, write.track.stat_revision)
                         for ordinal, write in writes},
                max_total_bytes=max_total_bytes, timeout_seconds=timeout_seconds)
            if final_fingerprints != fingerprints:
                raise ValidationError("Cohort bytes changed during native write preparation.")
            if (await self._store.get_local_album(album_id) is not None
                    or await self._import_reuse_album_id(canonical.identifiers.release_group_mbid)):
                raise ValidationError("Album ownership appeared during candidate preparation.")
            if self._resolver_getter().policy_revision != bundle.policy_revision:
                raise LibraryManagementPolicyChangedError("Cohort policy changed during preparation.")
            return writes, fingerprints, str(parent)

    async def _build_published_import_write(
        self,
        audio_path: Path,
        tag: "AudioTag",
        info: "AudioInfo",
        *,
        resolver: "LibraryPolicyResolver",
        release_group_mbid: str | None,
        release_mbid: str | None,
        recording_mbid: str | None,
        source: str,
        download_task_id: str | None,
        source_path: str | None,
        file_mtime: float | None,
    ) -> ScannedTrackWrite:
        root, relative_path = self._root_for(audio_path, resolver)
        stat_result = await asyncio.to_thread(audio_path.stat)
        resolution = resolver.resolve(audio_path)
        if resolution is None:
            raise ValueError("Imported file is outside every configured library root.")
        policy = resolution.policy
        now = file_mtime if file_mtime is not None else stat_result.st_mtime
        album_title = tag.album.strip() or audio_path.parent.name or "Unknown Album"
        album_artist = (tag.album_artist or tag.artist or "Unknown Artist").strip()
        artist_id = await self._store.find_target_artist_by_name(album_artist)
        artist_id = artist_id or grouping_artist_candidate_id(album_artist)
        directory = grouping_directory(relative_path)
        grouping_key = (
            f"{root.id}:{directory}:{normalize_group_value(album_title)}:"
            f"{normalize_group_value(album_artist)}"
        )
        album_id = await self._import_reuse_album_id(
            release_group_mbid
        ) or grouping_album_id(grouping_key)
        existing = await self._store.get_target_track_by_path(str(audio_path))
        track_id = (
            str(existing["id"])
            if existing is not None
            else str(uuid.uuid5(_TRACK_NAMESPACE, f"{root.id}:{relative_path}"))
        )
        tag_revision = hashlib.sha256(msgspec.json.encode(tag)).hexdigest()
        artist = LocalArtist(
            id=artist_id,
            display_name=album_artist,
            folded_name=album_artist.casefold(),
            normalized_name=normalize_group_value(album_artist),
            sort_name=tag.album_artist_sort,
            kind="group",
            created_at=now,
            updated_at=now,
        )
        album = LocalAlbum(
            id=album_id,
            root_id=root.id,
            grouping_key=grouping_key,
            title=album_title,
            album_artist_id=artist_id,
            album_artist_name=album_artist,
            album_artist_sort_name=tag.album_artist_sort,
            year=tag.year,
            original_release_date=tag.original_release_date,
            primary_genre=tag.genre,
            is_compilation=tag.compilation,
            grouping_source="automatic",
            created_at=now,
            updated_at=now,
        )
        track = LocalTrack(
            id=track_id,
            local_album_id=album_id,
            root_id=root.id,
            file_path=str(audio_path),
            relative_path=relative_path,
            path_hash=hashlib.sha256(relative_path.encode()).hexdigest(),
            file_size_bytes=stat_result.st_size,
            file_mtime_ns=stat_result.st_mtime_ns,
            stat_revision=revision_from_stat(stat_result),
            tag_revision=tag_revision,
            tags_read_at=now,
            title=tag.title.strip() or audio_path.stem,
            artist_name=tag.artist or album_artist,
            album_title=album_title,
            album_artist_name=album_artist,
            tag_album_title=tag.album,
            tag_album_artist_name=tag.album_artist,
            disc_number=tag.disc_number,
            track_number=tag.track_number,
            year=tag.year,
            genre=tag.genre,
            title_sort=tag.title_sort,
            artist_sort=tag.artist_sort,
            album_sort=tag.album_sort,
            album_artist_sort=tag.album_artist_sort,
            disc_subtitle=tag.disc_subtitle,
            release_type=tag.release_type,
            is_compilation=tag.compilation,
            embedded_release_group_mbid=(
                release_group_mbid or tag.musicbrainz_release_group_id
            ),
            embedded_release_mbid=release_mbid or tag.musicbrainz_release_id,
            embedded_recording_mbid=recording_mbid or tag.musicbrainz_recording_id,
            embedded_release_track_mbid=tag.musicbrainz_release_track_id,
            embedded_artist_mbid=tag.musicbrainz_artist_id,
            embedded_album_artist_mbid=tag.musicbrainz_album_artist_id,
            duration_seconds=info.duration_seconds,
            file_format=info.file_format,
            bit_rate=info.bitrate,
            sample_rate=info.sample_rate,
            bit_depth=info.bit_depth,
            channels=info.channels,
            replaygain_track_gain=tag.replaygain_track_gain,
            replaygain_album_gain=tag.replaygain_album_gain,
            replaygain_track_peak=tag.replaygain_track_peak,
            replaygain_album_peak=tag.replaygain_album_peak,
            ingest_source=source,
            download_task_id=download_task_id,
            source_path=source_path,
            imported_at=now,
            desired_policy_revision=resolver.policy_revision,
            applied_policy_revision=resolver.policy_revision,
            applied_policy=policy,
            availability="excluded" if policy == "excluded" else "indexed",
            excluded_at=now if policy == "excluded" else None,
        )
        credit = LocalArtistCredit(
            local_artist_id=artist_id,
            position=0,
            credited_name=album_artist,
        )
        genres = [
            LocalTrackGenre(
                local_track_id=track_id,
                position=position,
                name=value,
                folded_name=value.casefold(),
                source="local",
                source_document_revision=tag_revision,
            )
            for position, value in enumerate(
                tag.genres or ([tag.genre] if tag.genre else [])
            )
        ]
        return ScannedTrackWrite(
            artist=artist,
            album=album,
            track=track,
            credit=credit,
            root_id=root.id,
            relative_path=relative_path,
            comparison_result="changed" if existing is not None else "new",
            grouping_context=directory,
            artists=[artist],
            album_credits=[credit],
            track_credits=[credit],
            genres=genres,
        )

    async def soft_delete_file(self, file_path: str) -> None:
        row = await self._store.get_target_track_by_path(file_path)
        if row is not None:
            await self._store.mark_target_tracks_missing(
                [str(row["id"])],
                actor_user_id=None,
                reason_code="IMPORT_REPLACED",
                missing_at=time.time(),
            )

    async def reconcile_with_filesystem(self, targets: list[Path] | None = None) -> int:
        if not targets:
            return 0

        resolver = self._resolver_getter()
        total = 0
        for target in targets:
            resolved_target = target.resolve()
            resolved = resolver.resolve(resolved_target)
            if resolved is None:
                continue

            def walk() -> set[str]:
                return {
                    str(Path(directory) / filename)
                    for directory, _subdirs, filenames in os.walk(resolved_target)
                    for filename in filenames
                    if Path(filename).suffix.casefold() in AUDIO_EXTENSIONS
                }

            if self._filesystem is None:
                present = await asyncio.to_thread(walk)
                rows = await self._store.get_target_tracks_under_paths(
                    [str(resolved_target)]
                )
                missing = [
                    str(row["id"]) for row in rows if row["file_path"] not in present
                ]
                if missing:
                    await self._store.mark_target_tracks_missing(
                        missing,
                        actor_user_id=None,
                        reason_code="IMPORT_RECONCILE_MISSING",
                        missing_at=time.time(),
                    )
            else:
                async with self._filesystem.read(resolved.root_id):
                    present = await asyncio.to_thread(walk)
                    rows = await self._store.get_target_tracks_under_paths(
                        [str(resolved_target)]
                    )
                    missing = [
                        str(row["id"])
                        for row in rows
                        if row["file_path"] not in present
                    ]
                    if missing:
                        await self._store.mark_target_tracks_missing(
                            missing,
                            actor_user_id=None,
                            reason_code="IMPORT_RECONCILE_MISSING",
                            missing_at=time.time(),
                        )
            total += len(missing)
        return total

    async def get_attributions_for_paths(self, paths: list[str]) -> dict[str, dict]:
        rows = await self._store.get_target_attributions_for_paths(paths)
        for row in rows.values():
            row["release_group_mbid"] = row.get("provider_release_group_mbid")
            row["confidence"] = 1.0
            row["source"] = row.get("ingest_source")
        return rows

    async def get_file_at_position(
        self, album_id: str, disc_number: int, track_number: int
    ) -> dict | None:
        return next(
            (
                row
                for row in await self._store.get_target_album_tracks(album_id)
                if int(row["disc_number"] or 1) == disc_number
                and int(row["track_number"] or 0) == track_number
            ),
            None,
        )

    async def get_imported_file(
        self, download_task_id: str, filename: str
    ) -> dict | None:
        if not download_task_id:
            return None
        return await self._store.get_target_imported_track(download_task_id, filename)
