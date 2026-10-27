"""Bounded, append-only primary evidence for independent download review."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from infrastructure.serialization import to_jsonable


PRIMARY_EVIDENCE_DDL = """
CREATE TABLE IF NOT EXISTS download_primary_evidence_cohorts (
    cohort_id TEXT PRIMARY KEY,
    max_tasks INTEGER NOT NULL CHECK(max_tasks BETWEEN 1 AND 100),
    opened_at REAL NOT NULL,
    admissions_closed_at REAL
);
CREATE TABLE IF NOT EXISTS download_primary_evidence_admissions (
    cohort_id TEXT NOT NULL,
    task_id TEXT NOT NULL UNIQUE,
    ordinal INTEGER NOT NULL CHECK(ordinal >= 1),
    admitted_at REAL NOT NULL,
    PRIMARY KEY(cohort_id, ordinal),
    FOREIGN KEY(cohort_id) REFERENCES download_primary_evidence_cohorts(cohort_id)
);
CREATE TABLE IF NOT EXISTS download_primary_evidence_events (
    id TEXT PRIMARY KEY,
    cohort_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    sequence INTEGER NOT NULL CHECK(sequence >= 1),
    event_kind TEXT NOT NULL CHECK(event_kind IN (
        'request_created','selection_manifest','attempt_state','secured_artifact'
    )),
    captured_at REAL NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL CHECK(
        length(payload_sha256) = 64 AND payload_sha256 = lower(payload_sha256)
        AND payload_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    previous_event_sha256 TEXT,
    event_sha256 TEXT NOT NULL UNIQUE CHECK(
        length(event_sha256) = 64 AND event_sha256 = lower(event_sha256)
        AND event_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    execution_authorized INTEGER NOT NULL DEFAULT 0
        CHECK(execution_authorized = 0),
    UNIQUE(task_id, sequence),
    FOREIGN KEY(cohort_id) REFERENCES download_primary_evidence_cohorts(cohort_id)
);
CREATE INDEX IF NOT EXISTS idx_download_primary_evidence_task
    ON download_primary_evidence_events(task_id, sequence);
"""

_FORBIDDEN_KEYS = frozenset(
    {
        "controller_precheck",
        "controller_answer",
        "model_response",
        "gold",
        "expected_action",
        "expected_diagnosis",
        "safe_to_execute",
        "may_import",
        "may_retry",
        "may_delete",
        "may_acquire",
    }
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _scan(value: Any) -> None:
    if isinstance(value, dict):
        contaminated = set(value) & _FORBIDDEN_KEYS
        if contaminated:
            raise ValueError(
                "primary evidence contains forbidden keys: "
                + ",".join(sorted(contaminated))
            )
        for nested in value.values():
            _scan(nested)
    elif isinstance(value, list):
        for nested in value:
            _scan(nested)


def admit_task(
    conn: sqlite3.Connection,
    *,
    cohort_id: str,
    max_tasks: int,
    task_id: str,
    admitted_at: float,
) -> bool:
    """Admit only a newly created task, atomically, until the fixed bound."""
    conn.execute(
        "INSERT OR IGNORE INTO download_primary_evidence_cohorts "
        "(cohort_id,max_tasks,opened_at) VALUES (?,?,?)",
        (cohort_id, max_tasks, admitted_at),
    )
    cohort = conn.execute(
        "SELECT max_tasks,admissions_closed_at FROM "
        "download_primary_evidence_cohorts WHERE cohort_id=?",
        (cohort_id,),
    ).fetchone()
    if cohort is None or int(cohort["max_tasks"]) != max_tasks:
        raise ValueError("primary evidence cohort configuration changed")
    existing = conn.execute(
        "SELECT 1 FROM download_primary_evidence_admissions WHERE task_id=?",
        (task_id,),
    ).fetchone()
    if existing is not None:
        return True
    if cohort["admissions_closed_at"] is not None:
        return False
    count = int(
        conn.execute(
            "SELECT COUNT(*) FROM download_primary_evidence_admissions "
            "WHERE cohort_id=?",
            (cohort_id,),
        ).fetchone()[0]
    )
    if count >= max_tasks:
        return False
    ordinal = count + 1
    conn.execute(
        "INSERT INTO download_primary_evidence_admissions "
        "(cohort_id,task_id,ordinal,admitted_at) VALUES (?,?,?,?)",
        (cohort_id, task_id, ordinal, admitted_at),
    )
    if ordinal == max_tasks:
        conn.execute(
            "UPDATE download_primary_evidence_cohorts "
            "SET admissions_closed_at=? WHERE cohort_id=?",
            (admitted_at, cohort_id),
        )
    return True


def task_admitted(conn: sqlite3.Connection, task_id: str, cohort_id: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM download_primary_evidence_admissions "
            "WHERE task_id=? AND cohort_id=?",
            (task_id, cohort_id),
        ).fetchone()
        is not None
    )


def append_event(
    conn: sqlite3.Connection,
    *,
    cohort_id: str,
    task_id: str,
    event_kind: str,
    payload: dict[str, Any],
    captured_at: float | None = None,
) -> dict[str, Any] | None:
    """Append one event only for a task admitted to the configured cohort."""
    if not task_admitted(conn, task_id, cohort_id):
        return None
    _scan(payload)
    now = time.time() if captured_at is None else captured_at
    previous = conn.execute(
        "SELECT sequence,event_sha256 FROM download_primary_evidence_events "
        "WHERE task_id=? ORDER BY sequence DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    sequence = int(previous["sequence"]) + 1 if previous is not None else 1
    previous_sha = str(previous["event_sha256"]) if previous is not None else None
    payload_bytes = _canonical(payload)
    payload_sha = hashlib.sha256(payload_bytes).hexdigest()
    event_body = {
        "schema_version": 1,
        "cohort_id": cohort_id,
        "task_id": task_id,
        "sequence": sequence,
        "event_kind": event_kind,
        "captured_at": now,
        "payload_sha256": payload_sha,
        "previous_event_sha256": previous_sha,
        "execution_authorized": False,
    }
    event_sha = hashlib.sha256(_canonical(event_body)).hexdigest()
    event_id = uuid.uuid4().hex
    conn.execute(
        """INSERT INTO download_primary_evidence_events
           (id,cohort_id,task_id,sequence,event_kind,captured_at,payload_json,
            payload_sha256,previous_event_sha256,event_sha256,execution_authorized)
           VALUES (?,?,?,?,?,?,?,?,?,?,0)""",
        (
            event_id,
            cohort_id,
            task_id,
            sequence,
            event_kind,
            now,
            payload_bytes.decode("utf-8"),
            payload_sha,
            previous_sha,
            event_sha,
        ),
    )
    return {"id": event_id, **event_body, "event_sha256": event_sha, "payload": payload}


def verify_events(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    previous_sha = None
    for expected_sequence, row in enumerate(rows, 1):
        payload_text = str(row["payload_json"])
        payload = json.loads(payload_text)
        _scan(payload)
        if int(row["sequence"]) != expected_sequence:
            raise ValueError("primary evidence sequence gap")
        if row["previous_event_sha256"] != previous_sha:
            raise ValueError("primary evidence predecessor mismatch")
        payload_sha = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
        if payload_sha != row["payload_sha256"]:
            raise ValueError("primary evidence payload hash mismatch")
        body = {
            "schema_version": 1,
            "cohort_id": row["cohort_id"],
            "task_id": row["task_id"],
            "sequence": int(row["sequence"]),
            "event_kind": row["event_kind"],
            "captured_at": float(row["captured_at"]),
            "payload_sha256": payload_sha,
            "previous_event_sha256": previous_sha,
            "execution_authorized": False,
        }
        event_sha = hashlib.sha256(_canonical(body)).hexdigest()
        if event_sha != row["event_sha256"] or int(row["execution_authorized"]) != 0:
            raise ValueError("primary evidence event hash or authority mismatch")
        previous_sha = event_sha
        verified.append(
            {"id": row["id"], **body, "event_sha256": event_sha, "payload": payload}
        )
    return verified


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def _ownership_snapshot(conn: sqlite3.Connection, task: Any, captured_at: float) -> dict[str, Any]:
    if not _table_exists(conn, "local_tracks"):
        return {
            "captured_at": captured_at,
            "complete": False,
            "missing_evidence": ["native_local_tracks_table"],
            "tracks": [],
        }
    if task.download_type == "track" and task.recording_mbid:
        where = "LOWER(COALESCE(embedded_recording_mbid,''))=LOWER(?)"
        params: tuple[Any, ...] = (task.recording_mbid,)
    else:
        where = (
            "LOWER(COALESCE(embedded_release_group_mbid,''))=LOWER(?) "
            "OR LOWER(COALESCE(embedded_release_mbid,''))=LOWER(?)"
        )
        params = (task.release_group_mbid, task.release_mbid or "")
    rows = conn.execute(
        "SELECT id,file_path,path_hash,file_size_bytes,file_mtime_ns,stat_revision,"
        "stat_revision_kind,availability,embedded_release_group_mbid,"
        "embedded_release_mbid,embedded_recording_mbid,embedded_release_track_mbid "
        f"FROM local_tracks WHERE {where} ORDER BY id",
        params,
    ).fetchall()
    tracks = []
    stat_failures = 0
    for row in rows:
        path = str(row["file_path"])
        physical = Path(path).is_file()
        if not physical and row["availability"] == "indexed":
            stat_failures += 1
        tracks.append(
            {
                "local_track_id": row["id"],
                "path_hash": row["path_hash"],
                "physical_exists": physical,
                "file_size_bytes": row["file_size_bytes"],
                "file_mtime_ns": row["file_mtime_ns"],
                "stat_revision": row["stat_revision"],
                "stat_revision_kind": row["stat_revision_kind"],
                "availability": row["availability"],
                "release_group_mbid": row["embedded_release_group_mbid"],
                "release_mbid": row["embedded_release_mbid"],
                "recording_mbid": row["embedded_recording_mbid"],
                "release_track_mbid": row["embedded_release_track_mbid"],
            }
        )
    return {
        "captured_at": captured_at,
        "complete": stat_failures == 0,
        "missing_evidence": (
            [] if stat_failures == 0 else ["indexed_file_not_physically_present"]
        ),
        "tracks": tracks,
        "content_sha256_complete": False,
        "limitations": ["Ownership binds exact catalog and stat evidence, not audio bytes."],
    }


def request_payload(
    conn: sqlite3.Connection, task: Any, captured_at: float
) -> dict[str, Any]:
    intent = "exact_track" if task.download_type == "track" else "full_album"
    return {
        "schema_version": 1,
        "request_intent": intent,
        "request_identity": {
            "release_group_mbid": task.release_group_mbid,
            "release_mbid": task.release_mbid,
            "recording_mbid": task.recording_mbid if intent == "exact_track" else None,
            "release_track_mbid": (
                task.release_track_mbid if intent == "exact_track" else None
            ),
        },
        "request_context": {
            "origin": task.origin,
            "artist_name": task.artist_name,
            "album_title": task.album_title,
            "track_title": task.track_title,
            "track_count": task.track_count,
            "created_at": task.created_at,
        },
        "ownership_snapshot": _ownership_snapshot(conn, task, captured_at),
        "review_label": None,
        "independent_label_verified": False,
        "advisory_only": True,
        "execution_authorized": False,
        "permissions": {
            "import": False,
            "retry": False,
            "delete": False,
            "acquire": False,
        },
    }


def manifest_payload(
    conn: sqlite3.Connection, task: Any, manifest: Any, captured_at: float
) -> dict[str, Any]:
    tracks = [to_jsonable(value) for value in manifest.expected_tracks]
    recordings = [value.get("recording_mbid") for value in tracks]
    release_tracks = [value.get("release_track_mbid") for value in tracks]
    map_complete = (
        bool(tracks)
        and all(recordings)
        and all(release_tracks)
        and len(set(recordings)) == len(recordings)
        and len(set(release_tracks)) == len(release_tracks)
    )
    files = []
    for value in manifest.target_files:
        raw = to_jsonable(value)
        filename = str(raw.get("filename") or "")
        files.append(
            {
                "filename": filename,
                "filename_sha256": hashlib.sha256(filename.encode()).hexdigest(),
                "size": raw.get("size"),
                "duration": raw.get("duration"),
            }
        )
    payload = request_payload(conn, task, captured_at)
    payload.update(
        {
            "selected_candidate": {
                "search_job_id": task.search_job_id,
                "candidate_index": task.candidate_index,
                "source": task.source,
                "source_username": task.source_username,
                "source_directory": task.source_directory,
                "release_group_mbid": manifest.release_group_mbid,
                "release_mbid": manifest.release_mbid,
                "target_files": files,
            },
            "expected_recording_map": tracks,
            "expected_recording_mbids": recordings,
            "recording_map_complete": map_complete,
            "attempt_id": manifest.attempt_id,
        }
    )
    return payload


def attempt_payload(attempt: Any) -> dict[str, Any]:
    fingerprints = dict(attempt.materialized_fingerprints)
    return {
        "schema_version": 1,
        "attempt_id": attempt.id,
        "candidate_index": attempt.candidate_index,
        "source": attempt.source,
        "state": attempt.state,
        "disposition": attempt.disposition,
        "created_at": attempt.created_at,
        "updated_at": attempt.updated_at,
        "completed_at": attempt.completed_at,
        "row_revision": attempt.row_revision,
        "handle": to_jsonable(attempt.handle),
        "materialized_files": [
            {
                "path_sha256": hashlib.sha256(path.encode()).hexdigest(),
                "artifact_sha256": fingerprints.get(path),
            }
            for path in attempt.materialized_paths
        ],
        "materialization_hash_complete": bool(attempt.materialized_paths)
        and set(fingerprints) == set(attempt.materialized_paths)
        and all(fingerprints.values()),
        "error_code": attempt.error_code,
        "review_label": None,
        "execution_authorized": False,
        "permissions": {"import": False, "retry": False, "delete": False, "acquire": False},
    }


def secured_artifact_payload(
    values: dict[str, Any], artifact_sha256: str | None
) -> dict[str, Any]:
    held_path = str(values.get("held_path") or "")
    return {
        "schema_version": 1,
        "artifact": {
            "held_path_sha256": hashlib.sha256(held_path.encode()).hexdigest(),
            "artifact_sha256": artifact_sha256,
            "physical_exists": Path(held_path).is_file(),
            "release_group_mbid": values.get("release_group_mbid"),
            "release_mbid": values.get("release_mbid"),
            "release_track_mbid": values.get("release_track_mbid"),
            "recording_mbid": values.get("recording_mbid"),
            "disc_number": values.get("disc_number"),
            "track_number": values.get("track_number"),
            "original_filename": values.get("original_filename"),
            "reason": values.get("reason"),
        },
        "identity_complete": all(
            values.get(key)
            for key in (
                "release_group_mbid",
                "release_mbid",
                "release_track_mbid",
                "recording_mbid",
            )
        ),
        "artifact_hash_complete": artifact_sha256 is not None,
        "review_label": None,
        "execution_authorized": False,
        "permissions": {"import": False, "retry": False, "delete": False, "acquire": False},
    }


def hash_artifact(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, ValueError):
        return None
