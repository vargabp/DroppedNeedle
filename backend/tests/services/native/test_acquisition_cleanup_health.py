"""Health copy must distinguish safety preservation from retryable cleanup."""
import sqlite3
from types import SimpleNamespace

import pytest

from infrastructure.persistence.download_store import DownloadStore
from services.native import acquisition_cleanup_service as cleanup_module


@pytest.mark.parametrize("count,attention", [(1, 1), (2, 2), (3, 1)])
def test_protected_copies_request_review_without_blanket_retry_promise(count, attention):
    message = cleanup_module._cleanup_health_message(count, attention)
    assert "need review" in message
    assert "Automatic safety checks continue." in message
    assert "Retrying automatically." not in message
    assert ("Other temporary-file cleanup will retry" in message) == (count > attention)


def test_retryable_only_warning_retains_retry_message():
    assert cleanup_module._cleanup_health_message(1, 0) == (
        "Temporary files couldn't be removed for 1 download. "
        "Your library is safe. Retrying automatically."
    )


@pytest.mark.asyncio
async def test_count_query_deduplicates_tasks_and_excludes_completed_debt():
    with sqlite3.connect(":memory:") as db:
        db.row_factory = sqlite3.Row
        db.execute("CREATE TABLE download_attempts(task_id TEXT, state TEXT, cleanup_failures INTEGER)")
        db.executemany("INSERT INTO download_attempts VALUES(?,?,?)", [
            ("protected", "needs_attention", 0), ("protected", "cleanup_pending", 5),
            ("retry", "workspace_removed", 3), ("below", "cleanup_pending", 2),
            ("done", "complete", 99),
        ])

        async def read(operation):
            return operation(db)

        result = await DownloadStore.cleanup_warning_counts(SimpleNamespace(_read=read))
        assert result == {"total": 2, "needs_attention": 1}
