"""Seal one complete primary-evidence cohort for a blind reviewer.

The export contains observation payloads and integrity metadata only. It never
includes controller/model answers, labels, source code, or execution authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from infrastructure.persistence.download_primary_evidence import verify_events


_TERMINAL = frozenset({"completed", "partial", "failed", "cancelled"})


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _binding_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bound(record: dict[str, Any]) -> dict[str, Any]:
    return {"record": record, "sha256": _binding_sha256(record)}


def _v2_host_evidence(
    *, case_id: str, terminal_status: str, events: list[dict[str, Any]]
) -> dict[str, Any]:
    manifests = [
        event["payload"]
        for event in events
        if event["event_kind"] == "selection_manifest"
        and event["payload"].get("schema_version") == 2
    ]
    artifacts = [
        event["payload"]
        for event in events
        if event["payload"].get("observation_kind")
        == "published_import_artifacts"
    ]
    manifest = manifests[-1] if manifests else None

    request: dict[str, Any] = {
        "intent": None,
        "release_mbid": None,
        "recording_mbid": None,
        "release_track_mbid": None,
        "expected_recording_map": [],
        "complete": False,
    }
    ownership: dict[str, Any] = {
        "complete": False,
        "missing_evidence": ["v2_manifest_missing"],
        "tracks": [],
    }
    if manifest is not None:
        identity = manifest.get("request_identity") or {}
        expected = [
            {
                "disc_number": value.get("disc_number"),
                "track_number": value.get("track_number"),
                "recording_mbid": value.get("recording_mbid"),
                "release_track_mbid": value.get("release_track_mbid"),
            }
            for value in manifest.get("expected_recording_map") or []
        ]
        request = {
            "intent": manifest.get("request_intent"),
            "release_mbid": identity.get("release_mbid"),
            "recording_mbid": identity.get("recording_mbid"),
            "release_track_mbid": identity.get("release_track_mbid"),
            "expected_recording_map": expected,
            "complete": bool(manifest.get("recording_map_complete")),
        }
        snapshot = manifest.get("ownership_snapshot") or {}
        ownership = {
            "complete": snapshot.get("complete") is True
            and snapshot.get("content_sha256_complete") is True,
            "missing_evidence": list(snapshot.get("missing_evidence") or []),
            "tracks": [
                {
                    "disc_number": value.get("disc_number"),
                    "track_number": value.get("track_number"),
                    "release_mbid": value.get("release_mbid"),
                    "recording_mbid": value.get("recording_mbid"),
                    "release_track_mbid": value.get("release_track_mbid"),
                    "physical": value.get("physical_exists"),
                    "indexed": value.get("availability") == "indexed",
                    "content_sha256": value.get("content_sha256"),
                }
                for value in snapshot.get("tracks") or []
            ],
            "captured_at": snapshot.get("captured_at"),
        }

    proof_files: list[dict[str, Any]] = []
    seen_proofs: set[str] = set()
    for observation in artifacts:
        for value in observation.get("artifacts") or []:
            if not isinstance(value, dict):
                continue
            binding = _binding_sha256(value)
            if binding in seen_proofs:
                continue
            seen_proofs.add(binding)
            proof_files.append(value)
    artifact_observation_complete = bool(artifacts) and all(
        value.get("observation_complete") is True for value in artifacts
    )
    selection = {
        "complete": bool(
            manifest
            and manifest.get("recording_map_complete")
            and artifact_observation_complete
        ),
        "release_mbid": request.get("release_mbid"),
        "recording_proofs": proof_files,
    }
    artifact = {
        "terminal": terminal_status in _TERMINAL,
        "terminal_status": terminal_status,
        "observation_complete": artifact_observation_complete,
        "files": proof_files,
    }
    return {
        "schema": "droppedneedle-host-evidence-v2",
        "case_id": case_id,
        "request": _bound(request),
        "ownership": _bound(ownership),
        "selection": _bound(selection),
        "artifact": _bound(artifact),
        "review_labels_included": False,
        "controller_material_included": False,
        "execution_authorized": False,
    }


def _write_new(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def export_cohort(db_path: Path, cohort_id: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("reviewer export destination already exists")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cohort = conn.execute(
            "SELECT * FROM download_primary_evidence_cohorts WHERE cohort_id=?",
            (cohort_id,),
        ).fetchone()
        if cohort is None:
            raise ValueError("primary evidence cohort not found")
        admissions = conn.execute(
            """SELECT a.ordinal,a.task_id,a.admitted_at,t.status,t.updated_at
               FROM download_primary_evidence_admissions a
               JOIN download_tasks t ON t.id=a.task_id
               WHERE a.cohort_id=? ORDER BY a.ordinal""",
            (cohort_id,),
        ).fetchall()
        maximum = int(cohort["max_tasks"])
        if len(admissions) != maximum or cohort["admissions_closed_at"] is None:
            raise ValueError("primary evidence cohort admission is not complete")
        nonterminal = [row["task_id"] for row in admissions if row["status"] not in _TERMINAL]
        if nonterminal:
            raise ValueError("primary evidence cohort contains non-terminal tasks")

        prepared: list[tuple[str, bytes, dict[str, Any]]] = []
        evaluation_cases: list[dict[str, Any]] = []
        for row in admissions:
            events = conn.execute(
                "SELECT * FROM download_primary_evidence_events "
                "WHERE task_id=? ORDER BY sequence",
                (row["task_id"],),
            ).fetchall()
            verified = verify_events(events)
            if not verified or verified[0]["event_kind"] != "request_created":
                raise ValueError("admitted task lacks a valid request event")
            if any(event["cohort_id"] != cohort_id for event in verified):
                raise ValueError("task evidence belongs to another cohort")
            case_id = f"prospective-{int(row['ordinal']):03d}"
            case = {
                "schema_version": 1,
                "cohort_id": cohort_id,
                "ordinal": int(row["ordinal"]),
                "case_id": case_id,
                "task_id": row["task_id"],
                "admitted_at": float(row["admitted_at"]),
                "terminal_status": row["status"],
                "terminal_updated_at": float(row["updated_at"]),
                "events": verified,
                "review_label": None,
                "independent_label_verified": False,
                "controller_material_included": False,
                "execution_authorized": False,
            }
            name = f"case-{int(row['ordinal']):03d}.json"
            prepared.append((name, _canonical(case), case))
            evaluation_cases.append(
                {
                    "case_id": case_id,
                    "host_evidence": _v2_host_evidence(
                        case_id=case_id,
                        terminal_status=str(row["status"]),
                        events=verified,
                    ),
                }
            )
    finally:
        conn.close()

    output.mkdir(parents=False)
    case_entries = []
    for name, content, case in prepared:
        digest = _write_new(output / name, content)
        case_entries.append(
            {
                "case_id": case["case_id"],
                "filename": name,
                "sha256": digest,
                "terminal_status": case["terminal_status"],
            }
        )
    evaluation_bundle = {
        "schema": "droppedneedle-evaluation-bundle-v2",
        "cohort_id": cohort_id,
        "sealed": True,
        "case_count": len(evaluation_cases),
        "cases": evaluation_cases,
        "review_labels_included": False,
        "controller_material_included": False,
        "execution_authorized": False,
    }
    evaluation_bundle_sha = _write_new(
        output / "evaluation-bundle.json", _canonical(evaluation_bundle)
    )
    manifest = {
        "schema_version": 2,
        "cohort_id": cohort_id,
        "max_tasks": maximum,
        "admitted_tasks": len(prepared),
        "terminal_tasks": len(prepared),
        "opened_at": float(cohort["opened_at"]),
        "admissions_closed_at": float(cohort["admissions_closed_at"]),
        "exported_at": time.time(),
        "cases": case_entries,
        "evaluation_bundle": {
            "filename": "evaluation-bundle.json",
            "sha256": evaluation_bundle_sha,
        },
        "review_labels_included": False,
        "controller_material_included": False,
        "execution_authorized": False,
    }
    manifest_bytes = _canonical(manifest)
    manifest_sha = _write_new(output / "manifest.json", manifest_bytes)
    seal = {
        "schema_version": 2,
        "cohort_id": cohort_id,
        "manifest_sha256": manifest_sha,
        "case_count": len(case_entries),
        "evaluation_bundle_sha256": evaluation_bundle_sha,
        "sealed_at": time.time(),
        "independent_review_required": True,
        "execution_authorized": False,
    }
    _write_new(output / "seal.json", _canonical(seal))
    return seal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seal = export_cohort(args.db, args.cohort_id, args.output)
    print(json.dumps(seal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
