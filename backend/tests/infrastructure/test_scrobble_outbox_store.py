import sqlite3
import threading

import pytest

from infrastructure.persistence.auth_store import AuthStore
from infrastructure.persistence.native_library_store import NativeLibraryStore
from infrastructure.persistence.play_history_store import PlayHistoryStore
from infrastructure.persistence.scrobble_outbox_store import ScrobbleOutboxStore
from services.native.target_reference_adapters import TargetPlayHistoryStore


@pytest.mark.asyncio
async def test_outbox_persists_and_marks_completed_listen_delivered(tmp_path):
    store = ScrobbleOutboxStore(tmp_path / "library.db", threading.Lock())
    await store.enqueue(
        "history-1",
        user_id="user-1",
        artist_name="Artist",
        track_name="Track",
        listened_at=1700000000,
        album_name="Album",
        duration_ms=180000,
    )

    rows = await store.due(now=10**12)
    assert len(rows) == 1
    assert rows[0].play_history_id == "history-1"
    assert rows[0].user_id == "user-1"

    await store.mark_delivered("history-1")
    assert await store.count() == 1
    assert await store.count(status="pending") == 0
    assert await store.count(status="delivered") == 1


@pytest.mark.asyncio
async def test_outbox_enqueue_is_idempotent_and_defer_tracks_attempts(tmp_path):
    store = ScrobbleOutboxStore(tmp_path / "library.db", threading.Lock())
    kwargs = dict(
        user_id="user-1",
        artist_name="Artist",
        track_name="Track",
        listened_at=1700000000,
    )
    await store.enqueue("history-1", **kwargs)
    await store.enqueue("history-1", **kwargs)
    assert await store.count() == 1

    await store.defer("history-1", delay_seconds=60, increment_attempts=True)
    assert await store.due(now=0) == []
    rows = await store.due(now=10**12)
    assert rows[0].attempts == 1


@pytest.mark.asyncio
async def test_history_and_outbox_are_atomic_and_deduplicate_after_restart(tmp_path):
    db_path = tmp_path / "library.db"
    write_lock = threading.Lock()
    auth = AuthStore(db_path, write_lock)
    await auth.create_user(id="user-1", display_name="User", role="user")
    history = PlayHistoryStore(db_path, write_lock)
    store = ScrobbleOutboxStore(db_path, write_lock)

    first = await history.insert_with_listenbrainz_outbox(
        "user-1",
        delivery_key="stable-key",
        artist_name="Artist",
        track_name="Track",
        listened_at=1700000000,
        played_at="2023-11-14T22:13:20+00:00",
        album_name="Album",
        duration_ms=180000,
    )
    await store.mark_delivered(first.play_history_id)

    restarted = ScrobbleOutboxStore(db_path, write_lock)
    duplicate = await history.insert_with_listenbrainz_outbox(
        "user-1",
        delivery_key="stable-key",
        artist_name="Artist",
        track_name="Track",
        listened_at=1700000000,
        played_at="2023-11-14T22:13:20+00:00",
        album_name="Album",
        duration_ms=180000,
    )

    assert duplicate.created is False
    assert duplicate.play_history_id == first.play_history_id
    assert duplicate.status == "delivered"
    assert len(await history.recent("user-1")) == 1
    assert await restarted.count() == 1
    assert await restarted.due(now=10**12) == []


@pytest.mark.asyncio
async def test_outbox_insert_failure_rolls_back_local_history(tmp_path):
    db_path = tmp_path / "library.db"
    write_lock = threading.Lock()
    auth = AuthStore(db_path, write_lock)
    await auth.create_user(id="user-1", display_name="User", role="user")
    history = PlayHistoryStore(db_path, write_lock)
    store = ScrobbleOutboxStore(db_path, write_lock)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_lb_outbox BEFORE INSERT "
            "ON listenbrainz_scrobble_outbox BEGIN "
            "SELECT RAISE(ABORT, 'test outbox failure'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="test outbox failure"):
        await history.insert_with_listenbrainz_outbox(
            "user-1",
            delivery_key="will-fail",
            artist_name="Artist",
            track_name="Track",
            listened_at=1700000000,
            played_at="2023-11-14T22:13:20+00:00",
        )

    assert await history.recent("user-1") == []
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_target_history_and_outbox_share_the_atomic_transaction(tmp_path):
    db_path = tmp_path / "target.db"
    write_lock = threading.Lock()
    auth = AuthStore(db_path, write_lock)
    await auth.create_user(id="user-1", display_name="User", role="user")
    native = NativeLibraryStore(db_path, write_lock)
    history = TargetPlayHistoryStore(native)
    outbox = ScrobbleOutboxStore(db_path, write_lock)

    result = await history.insert_with_listenbrainz_outbox(
        "user-1",
        delivery_key="target-key",
        artist_name="Artist",
        track_name="Track",
        listened_at=1700000000,
        played_at="2023-11-14T22:13:20+00:00",
        album_name="Album",
        duration_ms=180000,
    )

    target_rows = await history.recent("user-1")
    with sqlite3.connect(db_path) as connection:
        legacy_count = connection.execute("SELECT COUNT(*) FROM play_history").fetchone()[0]

    assert result.created is True
    assert len(target_rows) == 1
    assert target_rows[0].id == result.play_history_id
    assert legacy_count == 0
    assert await outbox.count(status="pending") == 1
