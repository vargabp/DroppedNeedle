"""Bounded, external-service-free indexing for discovered target inventory."""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
import unicodedata
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

import msgspec

from core.exceptions import AudioFormatError
from infrastructure.audio.metadata_engine import legacy_audio_projection
from infrastructure.persistence.native_library_store import NativeLibraryStore
from models.audio import AudioArtistCredit, AudioInfo, AudioTag
from models.library_work import ScanFailureRecord, ScanRun, ScannedTrackWrite
from models.local_catalog import (
    LocalAlbum,
    LocalArtist,
    LocalArtistCredit,
    LocalTrack,
    LocalTrackGenre,
)
from services.native.identification_queue_service import IdentificationQueueService
from services.native.library_filesystem_coordinator import LibraryFilesystemCoordinator
from services.native.local_album_grouper import grouping_directory
from services.native.local_album_grouping_service import LocalAlbumGroupingService
from services.native.file_revision import revision_from_stat
from services.native.track_provenance import UNKNOWN_ARTIST, parse_names_for_row

INDEX_FETCH_SIZE = 256
TAG_BATCH_SIZE = 64
INDEX_BATCH_SIZE = TAG_BATCH_SIZE
CHECKPOINT_INTERVAL_SECONDS = 0.25
TAG_READ_TIMEOUT_SECONDS = 30.0
MAX_DETACHED_TAG_READS = 4
# F-12: reap horizon for wedged detached tag reads as a multiple of the
# tag-read timeout (test-pinned at 3; see the hostile-filesystem
# qualification suite).
DETACHED_TAG_READ_REAP_MULTIPLIER = 3.0
_SCAN_NAMESPACE = uuid.UUID("c65f5557-43c6-4550-bd8a-ea7dcaac6411")


class _TagReadCapacityExhausted(Exception):
    pass


class _TagReadTimedOut(Exception):
    pass


def _fold_value(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.strip())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    ).casefold()
    return " ".join(without_marks.split())


class AudioTagReaderProtocol(Protocol):
    def read_tags(self, path: Path) -> tuple[AudioTag, AudioInfo]: ...


IndexProgressCallback = Callable[[ScanRun], Awaitable[None]]


class LibraryIndexer:
    def __init__(
        self,
        store: NativeLibraryStore,
        tag_reader: AudioTagReaderProtocol,
        grouping: LocalAlbumGroupingService | None = None,
        filesystem_coordinator: LibraryFilesystemCoordinator | None = None,
        *,
        tag_read_timeout_seconds: float = TAG_READ_TIMEOUT_SECONDS,
        max_detached_tag_reads: int = MAX_DETACHED_TAG_READS,
        clock: Callable[[], float] = time.time,
        detached_tag_read_reap_multiplier: float = DETACHED_TAG_READ_REAP_MULTIPLIER,
        monotonic_clock: Callable[[], float] = time.monotonic,
        _cohort_resolver_getter: Callable | None = None,
        _cohort_max_total_bytes: int = 256 * 1024 * 1024,
        _cohort_timeout_seconds: float = 10,
    ) -> None:
        if tag_read_timeout_seconds <= 0:
            raise ValueError("The tag-read timeout must be positive.")
        if max_detached_tag_reads < 1:
            raise ValueError("The detached tag-read limit must be positive.")
        if detached_tag_read_reap_multiplier <= 0:
            raise ValueError("The detached tag-read reap multiplier must be positive.")
        self._store = store
        self._tag_reader = tag_reader
        self._grouping = grouping or LocalAlbumGroupingService(
            store, IdentificationQueueService(store)
        )
        self._filesystem = filesystem_coordinator
        if _cohort_resolver_getter is not None and filesystem_coordinator is None:
            raise ValueError('Private cohort index requires native filesystem coordination.')
        self._cohort_resolver_getter = _cohort_resolver_getter
        self._cohort_max_total_bytes = _cohort_max_total_bytes
        self._cohort_timeout_seconds = _cohort_timeout_seconds
        self._tag_read_timeout_seconds = tag_read_timeout_seconds
        self._max_detached_tag_reads = max_detached_tag_reads
        self._detached_tag_read_reap_multiplier = detached_tag_read_reap_multiplier
        self._monotonic_clock = monotonic_clock
        # F-INDEXREC-05: catalog creation/import timestamps come from the scan
        # event (wall clock), never from the file's mtime.
        self._clock = clock
        self._detached_tag_reads: set[
            asyncio.Task[tuple[AudioTag, AudioInfo, os.stat_result]]
        ] = set()
        # F-12: detach timestamps for the in-flight set above (the set holds
        # no timestamps itself); entries older than the reap horizon are
        # forgotten lazily on refusal and on completion, never by a task.
        self._detached_tag_read_started: dict[
            asyncio.Task[tuple[AudioTag, AudioInfo, os.stat_result]], float
        ] = {}

    def _failure_record(
        self,
        run_id: str,
        position: tuple[str, str],
        failure_code: str,
        *,
        detail: str,
        error: BaseException | None = None,
    ) -> ScanFailureRecord:
        """Safe deterministic detail for an indexing failure: the exception
        CLASS is recorded, never str(error) or filesystem paths (NEW-SCAN-04)."""
        class_name = type(error).__name__ if error is not None else None
        detail_text = f"{detail} (exception class: {class_name})" if class_name else detail
        root_id, relative_path = position
        return ScanFailureRecord(
            root_id=root_id,
            relative_path=relative_path,
            failure_code=failure_code,
            recorded_at=self._clock(),
            failure_detail=detail_text,
            phase="indexing",
        )

    async def index(
        self,
        run: ScanRun,
        frozen_policy_revision: str = "",
        checkpoint: Callable[[str, str], Awaitable[bool]] | None = None,
        progress: IndexProgressCallback | None = None,
    ) -> dict[str, int]:
        counts = {
            "indexed": 0,
            "new": 0,
            "changed": 0,
            "unchanged": 0,
            "excluded": 0,
            "errored": 0,
            "tag_reads": 0,
            "identification_enqueued": 0,
        }
        while True:
            await self._store.normalize_pending_legacy_inventory(
                run.id, limit=INDEX_FETCH_SIZE
            )
            batch = await self._store.get_scan_inventory_batch(
                run.id, processing_state="pending", limit=INDEX_FETCH_SIZE
            )
            if not batch:
                counts["identification_enqueued"] += await self._grouping.regroup_run(
                    run.id,
                    now=run.updated_at or run.queued_at,
                    checkpoint=checkpoint,
                    frozen_policy_revision=frozen_policy_revision,
                )
                return counts
            if self._cohort_resolver_getter is not None:
                from services.native.import_cohort_scan_candidate import index_cohort_inventory_candidate
                updated = await index_cohort_inventory_candidate(self, run, batch, counts, frozen_policy_revision, checkpoint)
                if updated is None:
                    return counts
                run = updated
                await self._report_progress(run, progress)
                continue
            batch_counts = {
                "indexed": 0,
                "new": 0,
                "changed": 0,
                "unchanged": 0,
                "excluded": 0,
                "errored": 0,
            }
            writes: list[ScannedTrackWrite] = []
            # F-SCAN-04/NEW-SCAN-04: indexing failures carry a stable, safe
            # detail (no raw exception text or paths) through to the store.
            failures: list[ScanFailureRecord] = []
            states: dict[str, list[tuple[str, str]]] = {
                "unchanged": [],
                "excluded": [],
            }
            last_checkpoint = time.monotonic()
            for item in batch:
                if len(writes) >= TAG_BATCH_SIZE:
                    break
                if checkpoint is not None and (
                    time.monotonic() - last_checkpoint >= CHECKPOINT_INTERVAL_SECONDS
                ):
                    if not await checkpoint(run.id, frozen_policy_revision):
                        return counts
                    last_checkpoint = time.monotonic()
                key = [(str(item["root_id"]), str(item["relative_path"]))]
                # S-03 recovery: "excluded" rows always skip tag reads here, as
                # do "unchanged" rows on every kind except `rescan_files`. The
                # skip is safe because availability/applied-policy convergence
                # (exclude, restore, identification re-enqueue) still happens
                # afterwards in `NativeLibraryStore.reconcile_scan_scope_batch`
                # via `LibraryReconciler.reconcile` once the discovery fence
                # completes - the indexer never owns that convergence.
                if item["comparison_result"] == "excluded" or (
                    item["comparison_result"] == "unchanged"
                    and run.kind != "rescan_files"
                ):
                    state = (
                        "unchanged"
                        if item["comparison_result"] == "unchanged"
                        else "excluded"
                    )
                    states[state].extend(key)
                    batch_counts["unchanged"] += (
                        item["comparison_result"] == "unchanged"
                    )
                    batch_counts["excluded"] += item["comparison_result"] == "excluded"
                    continue
                try:
                    stable = await self._read_stable_tags(
                        item,
                        run_id=run.id,
                        frozen_policy_revision=frozen_policy_revision,
                        checkpoint=checkpoint,
                    )
                    counts["tag_reads"] += stable[3]
                    if stable[4]:
                        return counts
                    if stable[0] is None or stable[1] is None or stable[2] is None:
                        failures.append(
                            self._failure_record(
                                run.id, key[0], "FILE_CHANGED_DURING_READ",
                                detail=(
                                    "The file changed during the stable tag-read "
                                    "consistency check."
                                ),
                            )
                        )
                        batch_counts["errored"] += 1
                        continue
                    tag, info, stat = stable[:3]
                    tagged_item = {
                        **item,
                        "file_size_bytes": stat.st_size,
                        "file_mtime_ns": stat.st_mtime_ns,
                        "stat_revision": revision_from_stat(stat),
                    }
                    writes.append(
                        self._prepare_tagged(
                            run.id, tagged_item, tag, info, now=self._clock()
                        )
                    )
                    if checkpoint is not None and not await checkpoint(
                        run.id, frozen_policy_revision
                    ):
                        return counts
                    batch_counts["indexed"] += 1
                    if item["comparison_result"] == "new":
                        batch_counts["new"] += 1
                    elif item["comparison_result"] == "changed":
                        batch_counts["changed"] += 1
                except _TagReadCapacityExhausted:
                    failures.append(
                        self._failure_record(
                            run.id, key[0], "TAG_READ_DEFERRED",
                            detail=(
                                f"The detached tag-read capacity of "
                                f"{MAX_DETACHED_TAG_READS} was exhausted; the read "
                                "is deferred for this run."
                            ),
                        )
                    )
                    batch_counts["errored"] += 1
                except _TagReadTimedOut:
                    failures.append(
                        self._failure_record(
                            run.id, key[0], "TAG_READ_TIMEOUT",
                            detail=(
                                f"The tag read exceeded its "
                                f"{self._tag_read_timeout_seconds:.1f}s deadline. "
                                "A kernel-blocked read is bounded by the timeout but "
                                "the underlying syscall may still be running."
                            ),
                        )
                    )
                    batch_counts["errored"] += 1
                except AudioFormatError as error:
                    failures.append(
                        self._failure_record(
                            run.id, key[0], "TAG_READ_FAILED",
                            detail="AudioFormatError while reading tags.",
                            error=error,
                        )
                    )
                    batch_counts["errored"] += 1
                except OSError as error:
                    failures.append(
                        self._failure_record(
                            run.id, key[0], "TAG_READ_FAILED",
                            detail="OSError while reading tags.",
                            error=error,
                        )
                    )
                    batch_counts["errored"] += 1
                except ValueError as error:
                    failures.append(
                        self._failure_record(
                            run.id, key[0], "TAG_READ_FAILED",
                            detail="ValueError while reading tags.",
                            error=error,
                        )
                    )
                    batch_counts["errored"] += 1
            if checkpoint is not None and not await checkpoint(
                run.id, frozen_policy_revision
            ):
                return counts
            increments = {
                "inspected_count": sum(
                    batch_counts[name]
                    for name in ("indexed", "unchanged", "excluded", "errored")
                ),
                "new_count": batch_counts["new"],
                "changed_count": batch_counts["changed"],
                "indexed_count": batch_counts["indexed"],
                "unchanged_count": batch_counts["unchanged"],
                "excluded_count": batch_counts["excluded"],
                "errored_count": batch_counts["errored"],
            }
            run = await self._store.commit_scan_index_batch(
                run.id,
                writes=writes,
                states=states,
                failures=failures,
                increments=increments,
                updated_at=time.time(),
            )
            for name in (
                "indexed",
                "new",
                "changed",
                "unchanged",
                "excluded",
                "errored",
            ):
                counts[name] += batch_counts[name]
            await self._report_progress(run, progress)

    async def _read_stable_tags(
        self,
        item: dict[str, object],
        *,
        run_id: str,
        frozen_policy_revision: str,
        checkpoint: Callable[[str, str], Awaitable[bool]] | None,
    ) -> tuple[AudioTag | None, AudioInfo | None, os.stat_result | None, int, bool]:
        path = Path(str(item["absolute_path"]))
        expected_size = int(item["file_size_bytes"])
        expected_mtime_ns = int(item["file_mtime_ns"])
        root_id = str(item["root_id"])
        reads = 0
        for _attempt in range(2):
            tag, info, stat = await self._read_tags_and_stat(path, root_id)
            reads += 1
            if checkpoint is not None and not await checkpoint(
                run_id, frozen_policy_revision
            ):
                return None, None, None, reads, True
            if stat.st_size == expected_size and stat.st_mtime_ns == expected_mtime_ns:
                return tag, info, stat, reads, False
            expected_size = stat.st_size
            expected_mtime_ns = stat.st_mtime_ns
        return None, None, None, reads, False

    async def _read_tags_and_stat(
        self, path: Path, root_id: str
    ) -> tuple[AudioTag, AudioInfo, os.stat_result]:
        if len(self._detached_tag_reads) >= self._max_detached_tag_reads:
            # F-12: wedged entries past the horizon are forgotten (drained,
            # cap freed) by design so scans keep moving; concurrency is
            # bounded by max-detached plus refusal/completion reaps. Only a
            # still-fresh full set defers the read.
            self._reap_stale_detached_tag_reads()
            if len(self._detached_tag_reads) >= self._max_detached_tag_reads:
                raise _TagReadCapacityExhausted

        async def read() -> tuple[AudioTag, AudioInfo, os.stat_result]:
            task = asyncio.create_task(
                asyncio.to_thread(self._read_tags_and_stat_sync, path)
            )
            try:
                return await asyncio.wait_for(
                    asyncio.shield(task), timeout=self._tag_read_timeout_seconds
                )
            except asyncio.CancelledError:
                self._detach_tag_read(task)
                raise
            except TimeoutError as error:
                self._detach_tag_read(task)
                raise _TagReadTimedOut from error

        if self._filesystem is None:
            return await read()
        async with self._filesystem.read(root_id):
            return await read()

    def _read_tags_and_stat_sync(
        self, path: Path
    ) -> tuple[AudioTag, AudioInfo, os.stat_result]:
        tag, info = self._read_tags(path)
        return tag, info, path.stat()

    def _finish_detached_tag_read(
        self, task: asyncio.Task[tuple[AudioTag, AudioInfo, os.stat_result]]
    ) -> None:
        # A late finish of an already-evicted entry is a safe noop: the
        # discard/pop tolerate the missing entry and the exception is still
        # consumed so no "never retrieved" warning escapes.
        self._detached_tag_reads.discard(task)
        self._detached_tag_read_started.pop(task, None)
        if not task.cancelled():
            task.exception()
        # F-12: opportunistically forget other wedged entries on completion
        # (past the horizon by design so scans keep moving; concurrency is
        # bounded by max-detached plus refusal/completion reaps).
        self._reap_stale_detached_tag_reads()

    def _reap_stale_detached_tag_reads(self) -> int:
        """Forget detached tag reads older than the reap horizon.

        Evicted entries leave tracking entirely (cap freed, timestamp map
        cleaned): wedged entries are forgotten past the horizon by design so
        scans keep moving, and concurrency is bounded by max-detached plus
        refusal/completion reaps. A late finish of an evicted entry stays a
        safe noop via its still-attached done-callback. Reaped files
        re-attempt next run via the persisted ``TAG_READ_DEFERRED`` marker
        re-offer (no permanent skip).
        """
        horizon = (
            self._tag_read_timeout_seconds * self._detached_tag_read_reap_multiplier
        )
        now = self._monotonic_clock()
        stale = [
            task
            for task, started in self._detached_tag_read_started.items()
            if now - started > horizon
        ]
        for task in stale:
            self._detached_tag_reads.discard(task)
            self._detached_tag_read_started.pop(task, None)
        return len(stale)

    def _detach_tag_read(
        self, task: asyncio.Task[tuple[AudioTag, AudioInfo, os.stat_result]]
    ) -> None:
        if task.done():
            return
        self._detached_tag_reads.add(task)
        self._detached_tag_read_started[task] = self._monotonic_clock()
        task.add_done_callback(self._finish_detached_tag_read)

    def _read_tags(self, path: Path) -> tuple[AudioTag, AudioInfo]:
        richer_reader = getattr(self._tag_reader, "read_document", None)
        if callable(richer_reader):
            return legacy_audio_projection(richer_reader(path))
        return self._tag_reader.read_tags(path)

    @staticmethod
    async def _report_progress(
        run: ScanRun, progress: IndexProgressCallback | None
    ) -> None:
        if progress is None:
            return
        await progress(run)

    def _prepare_tagged(
        self,
        run_id: str,
        item: dict[str, object],
        tag: AudioTag,
        info: AudioInfo,
        *,
        now: float,
    ) -> ScannedTrackWrite:
        root_id = str(item["root_id"])
        relative_path = str(item["relative_path"])
        directory = grouping_directory(relative_path)
        raw_album_title = tag.album.strip()
        raw_album_artist = (tag.album_artist or tag.artist or "").strip()
        raw_title = tag.title.strip()
        # M-01: filename fallback when tags are missing (pure path parse, no
        # IO). Raw `tag_*` columns stay honest (empty when untagged); DISPLAY
        # columns use tags -> parsed -> stem/`"Unknown Artist"`, and every
        # row touched here gets explicit provenance (never `absent`).
        parsed = None
        if not raw_album_title or not raw_album_artist or not raw_title:
            parsed = parse_names_for_row(relative_path)
        file_stem = Path(relative_path).stem
        # A parse that strips nothing (title/album == stem) is not evidence -
        # it would disagree with the derivation heuristic forever, so it
        # falls through to the stem placeholder like any unparseable path.
        parsed_title = (
            parsed.title
            if parsed is not None and parsed.title and parsed.title != file_stem
            else None
        )
        if raw_album_title:
            album_title, album_title_provenance = raw_album_title, "tag"
        elif parsed is not None and parsed.album and parsed.album != file_stem:
            album_title, album_title_provenance = parsed.album, "parsed"
        else:
            album_title, album_title_provenance = (
                file_stem,
                "placeholder",
            )
        if raw_album_artist:
            album_artist, album_artist_provenance = raw_album_artist, "tag"
        elif parsed is not None and parsed.artist:
            album_artist, album_artist_provenance = parsed.artist, "parsed"
        else:
            album_artist, album_artist_provenance = UNKNOWN_ARTIST, "placeholder"
        album_audio_credits = tag.album_artists or [
            AudioArtistCredit(
                name=album_artist,
                credited_name=album_artist,
                sort_name=tag.album_artist_sort,
                musicbrainz_artist_id=tag.musicbrainz_album_artist_id,
            )
        ]
        track_artist = (tag.artist or album_artist).strip()
        track_audio_credits = tag.artists or [
            AudioArtistCredit(
                name=track_artist,
                credited_name=track_artist,
                sort_name=tag.artist_sort,
                musicbrainz_artist_id=tag.musicbrainz_artist_id,
            )
        ]

        def candidate_artist(credit: AudioArtistCredit) -> LocalArtist:
            candidate_id = str(
                uuid.uuid5(
                    _SCAN_NAMESPACE,
                    f"artist:{credit.name}:{credit.sort_name or ''}:group",
                )
            )
            return LocalArtist(
                id=candidate_id,
                display_name=credit.name,
                folded_name=_fold_value(credit.name),
                kind="group",
                normalized_name=_fold_value(credit.name),
                sort_name=credit.sort_name,
                created_at=now,
                updated_at=now,
            )

        album_artists = [candidate_artist(credit) for credit in album_audio_credits]

        def matching_album_artist(
            track_credit: AudioArtistCredit,
        ) -> LocalArtist | None:
            matches: list[LocalArtist] = []
            track_name = _fold_value(track_credit.credited_name or track_credit.name)
            for album_credit, album_artist_candidate in zip(
                album_audio_credits, album_artists, strict=True
            ):
                album_name = _fold_value(
                    album_credit.credited_name or album_credit.name
                )
                if track_name != album_name:
                    continue
                track_mbid = (track_credit.musicbrainz_artist_id or "").casefold()
                album_mbid = (album_credit.musicbrainz_artist_id or "").casefold()
                if track_mbid and album_mbid and track_mbid != album_mbid:
                    continue
                matches.append(album_artist_candidate)
            return matches[0] if len(matches) == 1 else None

        track_artists = [
            matching_album_artist(credit) or candidate_artist(credit)
            for credit in track_audio_credits
        ]
        artist_by_candidate_id = {
            artist.id: artist for artist in [*album_artists, *track_artists]
        }
        primary_album_artist = album_artists[0]
        artist_candidate_id = primary_album_artist.id
        grouping_key = (
            f"{directory}\x00{album_title.casefold()}\x00{album_artist.casefold()}"
        )
        album_id = str(uuid.uuid5(_SCAN_NAMESPACE, f"album:{root_id}:{grouping_key}"))
        track_id = str(
            item["local_track_id"]
            or uuid.uuid5(_SCAN_NAMESPACE, f"track:{root_id}:{relative_path}")
        )
        tag_revision = hashlib.sha256(msgspec.json.encode(tag)).hexdigest()
        artist = primary_album_artist
        album = LocalAlbum(
            id=album_id,
            root_id=root_id,
            grouping_key=grouping_key,
            title=album_title or "Unknown Album",
            album_artist_id=artist_candidate_id,
            album_artist_name=album_artist,
            album_artist_sort_name=tag.album_artist_sort,
            year=tag.year,
            original_release_date=tag.original_release_date,
            primary_genre=tag.genre,
            is_compilation=tag.compilation,
            created_at=now,
            updated_at=now,
        )
        track = LocalTrack(
            id=track_id,
            local_album_id=album_id,
            root_id=root_id,
            file_path=str(item["absolute_path"]),
            relative_path=relative_path,
            path_hash=hashlib.sha256(relative_path.encode()).hexdigest(),
            file_size_bytes=int(item["file_size_bytes"]),
            file_mtime_ns=int(item["file_mtime_ns"]),
            stat_revision=str(item["stat_revision"]),
            tag_revision=tag_revision,
            tags_read_at=now,
            title=(raw_title or parsed_title or file_stem),
            title_provenance=(
                "tag" if raw_title else ("parsed" if parsed_title else "placeholder")
            ),
            artist_name=tag.artist or album_artist,
            album_title=album.title,
            album_artist_name=album_artist,
            album_title_provenance=album_title_provenance,
            album_artist_provenance=album_artist_provenance,
            tag_album_title=tag.album.strip(),
            tag_album_artist_name=(tag.album_artist or "").strip(),
            disc_number=tag.disc_number,
            track_number=(
                tag.track_number
                or (parsed.track_number if parsed is not None else None)
                or 0
            ),
            year=tag.year,
            genre=tag.genre,
            title_sort=tag.title_sort,
            artist_sort=tag.artist_sort,
            album_sort=tag.album_sort,
            album_artist_sort=tag.album_artist_sort,
            disc_subtitle=tag.disc_subtitle,
            release_type=tag.release_type,
            is_compilation=tag.compilation,
            embedded_release_group_mbid=tag.musicbrainz_release_group_id,
            embedded_release_mbid=tag.musicbrainz_release_id,
            embedded_recording_mbid=tag.musicbrainz_recording_id,
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
            imported_at=now,
            desired_policy_revision=str(item.get("policy_revision", "")),
            applied_policy_revision=str(item.get("policy_revision", "")),
            applied_policy=item["effective_policy"],
        )
        album_credits = [
            LocalArtistCredit(
                local_artist_id=album_artists[position].id,
                position=position,
                credited_name=credit.credited_name or credit.name,
                join_phrase=credit.join_phrase,
            )
            for position, credit in enumerate(album_audio_credits)
        ]
        track_credits = [
            LocalArtistCredit(
                local_artist_id=track_artists[position].id,
                position=position,
                credited_name=credit.credited_name or credit.name,
                join_phrase=credit.join_phrase,
            )
            for position, credit in enumerate(track_audio_credits)
        ]
        genre_values = tag.genres or ([tag.genre] if tag.genre else [])
        genres = [
            LocalTrackGenre(
                local_track_id=track_id,
                position=position,
                name=value,
                folded_name=_fold_value(value),
                source="local",
                source_document_revision=tag_revision,
            )
            for position, value in enumerate(genre_values)
        ]
        return ScannedTrackWrite(
            artist=artist,
            album=album,
            track=track,
            credit=LocalArtistCredit(
                local_artist_id=artist_candidate_id,
                position=0,
                credited_name=album_artist,
            ),
            root_id=root_id,
            relative_path=relative_path,
            comparison_result=item["comparison_result"],
            grouping_context=directory,
            artists=list(artist_by_candidate_id.values()),
            album_credits=album_credits,
            track_credits=track_credits,
            genres=genres,
        )
