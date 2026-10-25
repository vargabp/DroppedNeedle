import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from infrastructure.persistence._database import PersistenceBase, _safe_alter


@dataclass(frozen=True)
class ListenBrainzOutboxEnqueueResult:
    play_history_id: str
    created: bool
    status: str


@dataclass(frozen=True)
class PendingListenBrainzScrobble:
    play_history_id: str
    user_id: str
    artist_name: str
    track_name: str
    listened_at: int
    album_name: str
    duration_ms: int
    attempts: int
    delivery_key: str = ""


def find_listenbrainz_delivery(
    connection: sqlite3.Connection, delivery_key: str
) -> ListenBrainzOutboxEnqueueResult | None:
    row = connection.execute(
        "SELECT play_history_id, status FROM listenbrainz_scrobble_outbox "
        "WHERE delivery_key = ?",
        (delivery_key,),
    ).fetchone()
    if row is None:
        return None
    return ListenBrainzOutboxEnqueueResult(
        play_history_id=str(row["play_history_id"]),
        created=False,
        status=str(row["status"]),
    )


def insert_pending_listenbrainz_delivery(
    connection: sqlite3.Connection,
    *,
    play_history_id: str,
    delivery_key: str,
    user_id: str,
    artist_name: str,
    track_name: str,
    listened_at: int,
    album_name: str,
    duration_ms: int,
    now: float,
) -> ListenBrainzOutboxEnqueueResult:
    connection.execute(
        """
        INSERT INTO listenbrainz_scrobble_outbox (
            play_history_id, delivery_key, user_id, artist_name, track_name,
            listened_at, album_name, duration_ms, status, attempts,
            next_attempt_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
        """,
        (
            play_history_id,
            delivery_key,
            user_id,
            artist_name,
            track_name,
            listened_at,
            album_name,
            duration_ms,
            now,
            now,
            now,
        ),
    )
    return ListenBrainzOutboxEnqueueResult(
        play_history_id=play_history_id,
        created=True,
        status="pending",
    )


class ScrobbleOutboxStore(PersistenceBase):
    """Durable at-least-once delivery queue for completed ListenBrainz plays."""

    foreign_keys = True

    def __init__(self, db_path: Path, write_lock: threading.Lock | None = None):
        super().__init__(db_path, write_lock or threading.Lock())

    def _ensure_tables(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS listenbrainz_scrobble_outbox (
                    play_history_id TEXT PRIMARY KEY,
                    delivery_key TEXT,
                    user_id TEXT NOT NULL,
                    artist_name TEXT NOT NULL,
                    track_name TEXT NOT NULL,
                    listened_at INTEGER NOT NULL,
                    album_name TEXT NOT NULL DEFAULT '',
                    duration_ms INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL DEFAULT 0,
                    last_error TEXT
                )
                """
            )
            # Additive compatibility for databases created by an earlier
            # development build of the outbox.
            _safe_alter(
                connection,
                "ALTER TABLE listenbrainz_scrobble_outbox ADD COLUMN delivery_key TEXT",
            )
            _safe_alter(
                connection,
                "ALTER TABLE listenbrainz_scrobble_outbox "
                "ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'",
            )
            _safe_alter(
                connection,
                "ALTER TABLE listenbrainz_scrobble_outbox "
                "ADD COLUMN updated_at REAL NOT NULL DEFAULT 0",
            )
            _safe_alter(
                connection,
                "ALTER TABLE listenbrainz_scrobble_outbox ADD COLUMN last_error TEXT",
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_lb_scrobble_outbox_delivery_key "
                "ON listenbrainz_scrobble_outbox(delivery_key) "
                "WHERE delivery_key IS NOT NULL"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_lb_scrobble_outbox_due "
                "ON listenbrainz_scrobble_outbox(status, next_attempt_at, created_at)"
            )
            connection.commit()
        finally:
            connection.close()

    async def enqueue(
        self,
        play_history_id: str,
        *,
        user_id: str,
        artist_name: str,
        track_name: str,
        listened_at: int,
        album_name: str = "",
        duration_ms: int = 0,
        delivery_key: str | None = None,
    ) -> None:
        now = time.time()

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT OR IGNORE INTO listenbrainz_scrobble_outbox (
                    play_history_id, delivery_key, user_id, artist_name, track_name,
                    listened_at, album_name, duration_ms, status, attempts,
                    next_attempt_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
                """,
                (
                    play_history_id,
                    delivery_key,
                    user_id,
                    artist_name,
                    track_name,
                    listened_at,
                    album_name,
                    duration_ms,
                    now,
                    now,
                    now,
                ),
            )

        await self._write(operation)

    async def due(
        self, *, limit: int = 25, now: float | None = None
    ) -> list[PendingListenBrainzScrobble]:
        due_at = time.time() if now is None else now

        def operation(connection: sqlite3.Connection) -> list[sqlite3.Row]:
            return connection.execute(
                "SELECT * FROM listenbrainz_scrobble_outbox "
                "WHERE status = 'pending' AND next_attempt_at <= ? "
                "ORDER BY created_at, play_history_id LIMIT ?",
                (due_at, max(1, limit)),
            ).fetchall()

        rows = await self._read(operation)
        return [
            PendingListenBrainzScrobble(
                play_history_id=str(row["play_history_id"]),
                user_id=str(row["user_id"]),
                artist_name=str(row["artist_name"]),
                track_name=str(row["track_name"]),
                listened_at=int(row["listened_at"]),
                album_name=str(row["album_name"] or ""),
                duration_ms=int(row["duration_ms"] or 0),
                attempts=int(row["attempts"] or 0),
                delivery_key=str(row["delivery_key"] or ""),
            )
            for row in rows
        ]

    async def mark_delivered(self, play_history_id: str) -> None:
        now = time.time()
        await self._write(
            lambda connection: connection.execute(
                "UPDATE listenbrainz_scrobble_outbox SET status = 'delivered', "
                "updated_at = ?, last_error = NULL WHERE play_history_id = ?",
                (now, play_history_id),
            )
        )

    async def mark_failed(self, play_history_id: str, *, error: str) -> None:
        now = time.time()
        await self._write(
            lambda connection: connection.execute(
                "UPDATE listenbrainz_scrobble_outbox SET status = 'failed', "
                "updated_at = ?, last_error = ? WHERE play_history_id = ?",
                (now, error[:500], play_history_id),
            )
        )

    async def defer(
        self,
        play_history_id: str,
        *,
        delay_seconds: float,
        increment_attempts: bool,
    ) -> None:
        next_attempt_at = time.time() + max(1.0, delay_seconds)

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                "UPDATE listenbrainz_scrobble_outbox SET "
                "next_attempt_at = ?, attempts = attempts + ?, updated_at = ? "
                "WHERE play_history_id = ?",
                (
                    next_attempt_at,
                    1 if increment_attempts else 0,
                    time.time(),
                    play_history_id,
                ),
            )

        await self._write(operation)

    async def status_for_delivery_key(self, delivery_key: str) -> str | None:
        row = await self._read(
            lambda connection: connection.execute(
                "SELECT status FROM listenbrainz_scrobble_outbox "
                "WHERE delivery_key = ?",
                (delivery_key,),
            ).fetchone()
        )
        return str(row["status"]) if row is not None else None

    async def count(self, *, status: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM listenbrainz_scrobble_outbox"
        params: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        return int(
            await self._read(
                lambda connection: connection.execute(
                    query,
                    params,
                ).fetchone()[0]
            )
        )
