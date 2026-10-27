"""Bounded prospective primary-evidence cohort tests."""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from infrastructure.persistence.download_store import DownloadStore
from models.download_manifest import DownloadManifest, ExpectedFile, ExpectedTrack
from repositories.protocols.download_client import TaskHandle
from scripts.export_download_primary_evidence import export_cohort


GROUP = "11111111-1111-4111-8111-111111111111"
RELEASE = "22222222-2222-4222-8222-222222222222"
RECORDING = "33333333-3333-4333-8333-333333333333"
RELEASE_TRACK = "44444444-4444-4444-8444-444444444444"
COHORT = "prospective-test-2"


def _store(
    tmp_path: Path, *, enabled: bool, maximum: int = 2
) -> DownloadStore:
    db = tmp_path / "library.db"
    store = DownloadStore(
        db,
        threading.Lock(),
        primary_evidence_capture_enabled=enabled,
        primary_evidence_cohort_id=COHORT if enabled else "",
        primary_evidence_max_tasks=maximum if enabled else 0,
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS auth_users "
            "(id TEXT PRIMARY KEY, username TEXT, role TEXT)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO auth_users VALUES ('user-a','alice','user')"
        )
        conn.commit()
    return store


async def _task(store: DownloadStore, **overrides):
    values = {
        "user_id": "user-a",
        "download_type": "track",
        "release_group_mbid": GROUP,
        "release_mbid": RELEASE,
        "release_track_mbid": RELEASE_TRACK,
        "recording_mbid": RECORDING,
        "artist_name": "Artist",
        "album_title": "Album",
        "track_title": "Track",
        "track_number": 1,
        "disc_number": 1,
        "track_count": 1,
        "search_job_id": "search-1",
        "candidate_index": 0,
        "source_username": "peer-a",
        "source_directory": "Artist/Album",
    }
    values.update(overrides)
    return await store.create_task(**values)


def _manifest(task_id: str) -> DownloadManifest:
    return DownloadManifest(
        task_id=task_id,
        source_username="peer-a",
        handle=TaskHandle(
            source="soulseek", username="peer-a", filenames=["01 Track.flac"]
        ),
        origin="user",
        release_group_mbid=GROUP,
        release_mbid=RELEASE,
        artist_name="Artist",
        album_title="Album",
        naming_template="{track_number} {title}",
        target_files=[
            ExpectedFile(filename="01 Track.flac", size=1234, duration=180.0)
        ],
        expected_tracks=[
            ExpectedTrack(
                track_number=1,
                disc_number=1,
                duration_seconds=180.0,
                recording_mbid=RECORDING,
                title="Track",
                release_track_mbid=RELEASE_TRACK,
            )
        ],
        attempt_id="attempt-1",
    )


def _create_local_tracks(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE local_tracks (
            id TEXT PRIMARY KEY,file_path TEXT,path_hash TEXT,file_size_bytes INTEGER,
            file_mtime_ns INTEGER,stat_revision TEXT,stat_revision_kind TEXT,
            availability TEXT,embedded_release_group_mbid TEXT,
            embedded_release_mbid TEXT,embedded_recording_mbid TEXT,
            embedded_release_track_mbid TEXT)"""
    )


def test_enabled_configuration_requires_fixed_valid_bound(tmp_path: Path):
    with pytest.raises(ValueError, match="cohort id"):
        DownloadStore(
            tmp_path / "invalid.db",
            threading.Lock(),
            primary_evidence_capture_enabled=True,
            primary_evidence_cohort_id="",
            primary_evidence_max_tasks=40,
        )
    with pytest.raises(ValueError, match="max tasks"):
        DownloadStore(
            tmp_path / "invalid-bound.db",
            threading.Lock(),
            primary_evidence_capture_enabled=True,
            primary_evidence_cohort_id="valid",
            primary_evidence_max_tasks=0,
        )


@pytest.mark.asyncio
async def test_capture_disabled_by_default_and_no_retroactive_admission(tmp_path: Path):
    disabled = _store(tmp_path, enabled=False)
    old = await _task(disabled)
    enabled = _store(tmp_path, enabled=True)
    attempt = await enabled.create_download_attempt(
        task_id=old.id,
        source="soulseek",
        candidate_index=0,
        job_name="",
        handle=TaskHandle(source="soulseek", username="peer-a", filenames=[]),
    )
    assert attempt.task_id == old.id
    assert await enabled.list_primary_evidence_events(old.id) == []
    assert (await enabled.primary_evidence_cohort_status())["admitted_tasks"] == 0


@pytest.mark.asyncio
async def test_hard_bound_admits_only_first_new_tasks(tmp_path: Path):
    store = _store(tmp_path, enabled=True, maximum=2)
    first = await _task(store)
    second = await _task(store, recording_mbid="recording-2")
    third = await _task(store, recording_mbid="recording-3")

    assert len(await store.list_primary_evidence_events(first.id)) == 1
    assert len(await store.list_primary_evidence_events(second.id)) == 1
    assert await store.list_primary_evidence_events(third.id) == []
    status = await store.primary_evidence_cohort_status()
    assert status["max_tasks"] == 2
    assert status["admitted_tasks"] == 2
    assert status["admissions_closed_at"] is not None


@pytest.mark.asyncio
async def test_request_binds_exact_intent_and_original_ownership(tmp_path: Path):
    store = _store(tmp_path, enabled=True)
    audio = tmp_path / "owned.flac"
    audio.write_bytes(b"owned-audio")
    with sqlite3.connect(store.db_path) as conn:
        _create_local_tracks(conn)
        stat = audio.stat()
        conn.execute(
            "INSERT INTO local_tracks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "local-1",
                str(audio),
                "path-hash",
                stat.st_size,
                stat.st_mtime_ns,
                "stat-revision",
                "exact",
                "indexed",
                GROUP,
                RELEASE,
                RECORDING,
                RELEASE_TRACK,
            ),
        )
        conn.commit()

    task = await _task(store)
    event = (await store.list_primary_evidence_events(task.id))[0]
    payload = event["payload"]
    assert event["cohort_id"] == COHORT
    assert payload["request_intent"] == "exact_track"
    assert payload["request_identity"]["recording_mbid"] == RECORDING
    assert payload["ownership_snapshot"]["tracks"][0]["physical_exists"] is True
    assert payload["review_label"] is None
    assert set(payload["permissions"].values()) == {False}


@pytest.mark.asyncio
async def test_manifest_and_attempt_extend_verified_chain(tmp_path: Path):
    store = _store(tmp_path, enabled=True)
    task = await _task(store)
    await store.record_primary_evidence_manifest(_manifest(task.id))
    attempt = await store.create_download_attempt(
        task_id=task.id,
        source="soulseek",
        candidate_index=0,
        job_name="",
        handle=TaskHandle(
            source="soulseek", username="peer-a", filenames=["01 Track.flac"]
        ),
        attempt_id="attempt-1",
        now=100.0,
    )
    materialized = str(tmp_path / "01 Track.flac")
    digest = "a" * 64
    await store.transition_download_attempt(
        attempt.id,
        expected_row_revision=attempt.row_revision,
        new_state="preserved",
        now=101.0,
        materialized_paths_json=json.dumps([materialized]),
        materialized_fingerprints_json=json.dumps({materialized: digest}),
    )
    events = await store.list_primary_evidence_events(task.id)
    assert [event["event_kind"] for event in events] == [
        "request_created",
        "selection_manifest",
        "attempt_state",
        "attempt_state",
    ]
    assert events[1]["payload"]["recording_map_complete"] is True
    assert events[-1]["payload"]["materialization_hash_complete"] is True
    assert events[-1]["previous_event_sha256"] == events[-2]["event_sha256"]


@pytest.mark.asyncio
async def test_secured_artifact_hash_and_tampering_fail_closed(tmp_path: Path):
    store = _store(tmp_path, enabled=True)
    task = await _task(store)
    held = tmp_path / "held.flac"
    held.write_bytes(b"secured-artifact")
    await store.record_held_import(
        user_id="user-a",
        held_path=str(held),
        reason="tag_mismatch",
        source="soulseek",
        source_task_id=task.id,
        release_group_mbid=GROUP,
        release_mbid=RELEASE,
        release_track_mbid=RELEASE_TRACK,
        recording_mbid=RECORDING,
        track_number=1,
        disc_number=1,
        track_title="Track",
        artist_name="Artist",
        artist_mbid=None,
        album_title="Album",
        year=2026,
        original_filename="held.flac",
        file_format="flac",
        duration_seconds=180.0,
        evidence_title="Track",
        evidence_artist="Artist",
        evidence_score=1.0,
        naming_template="{track_number} {title}",
    )
    events = await store.list_primary_evidence_events(task.id)
    artifact = events[-1]["payload"]["artifact"]
    assert artifact["artifact_sha256"] == hashlib.sha256(
        b"secured-artifact"
    ).hexdigest()
    assert "held_path" not in artifact

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE download_primary_evidence_events SET payload_json='{}' "
            "WHERE task_id=? AND sequence=1",
            (task.id,),
        )
        conn.commit()
    with pytest.raises(ValueError, match="payload hash mismatch"):
        await store.list_primary_evidence_events(task.id)


@pytest.mark.asyncio
async def test_export_refuses_incomplete_then_seals_terminal_cohort(tmp_path: Path):
    store = _store(tmp_path, enabled=True, maximum=2)
    first = await _task(store)
    output = tmp_path / "reviewer"
    with pytest.raises(ValueError, match="admission is not complete"):
        export_cohort(store.db_path, COHORT, output)

    second = await _task(store, recording_mbid="recording-2")
    await store.update_status(first.id, "completed")
    with pytest.raises(ValueError, match="non-terminal"):
        export_cohort(store.db_path, COHORT, output)
    await store.update_status(second.id, "failed")

    seal = export_cohort(store.db_path, COHORT, output)
    assert seal["case_count"] == 2
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["controller_material_included"] is False
    assert manifest["review_labels_included"] is False
    assert [case["terminal_status"] for case in manifest["cases"]] == [
        "completed",
        "failed",
    ]
    assert (output / "case-001.json").is_file()
    assert (output / "seal.json").is_file()
