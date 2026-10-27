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
            case = {
                "schema_version": 1,
                "cohort_id": cohort_id,
                "ordinal": int(row["ordinal"]),
                "case_id": f"prospective-{int(row['ordinal']):03d}",
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
    manifest = {
        "schema_version": 1,
        "cohort_id": cohort_id,
        "max_tasks": maximum,
        "admitted_tasks": len(prepared),
        "terminal_tasks": len(prepared),
        "opened_at": float(cohort["opened_at"]),
        "admissions_closed_at": float(cohort["admissions_closed_at"]),
        "exported_at": time.time(),
        "cases": case_entries,
        "review_labels_included": False,
        "controller_material_included": False,
        "execution_authorized": False,
    }
    manifest_bytes = _canonical(manifest)
    manifest_sha = _write_new(output / "manifest.json", manifest_bytes)
    seal = {
        "schema_version": 1,
        "cohort_id": cohort_id,
        "manifest_sha256": manifest_sha,
        "case_count": len(case_entries),
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
