import asyncio
import hashlib
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from api.v1.schemas.scrobble import (
    NowPlayingRequest,
    ScrobbleRequest,
    ScrobbleResponse,
    ServiceResult,
)
from core.exceptions import (
    ExternalServiceError,
    NonRetriableExternalServiceError,
    RateLimitedError,
)
from infrastructure.persistence.play_history_store import PlayHistoryStore
from infrastructure.persistence.scrobble_outbox_store import (
    ListenBrainzOutboxEnqueueResult,
    PendingListenBrainzScrobble,
    ScrobbleOutboxStore,
)
from infrastructure.persistence.user_listening_prefs_store import UserListeningPrefsStore
from services.per_user_client_factory import PerUserClientFactory

logger = logging.getLogger(__name__)

DEDUP_TTL_SECONDS = 3600
DEDUP_MAX_ENTRIES = 200
MIN_TRACK_DURATION_MS = 30_000
LISTENBRAINZ_RETRY_BASE_SECONDS = 30.0
LISTENBRAINZ_RETRY_MAX_SECONDS = 3600.0
LISTENBRAINZ_UNLINKED_RETRY_SECONDS = 300.0


class ScrobbleService:
    """Per-user scrobbling + local play_history source of truth.

    Every accepted play is recorded regardless of external linkage; external
    forwarding goes only to the user's own linked Last.fm / ListenBrainz accounts.
    """

    def __init__(
        self,
        client_factory: PerUserClientFactory,
        listening_prefs_store: UserListeningPrefsStore,
        play_history_store: PlayHistoryStore,
        scrobble_outbox_store: ScrobbleOutboxStore | None = None,
        plugin_host=None,  # noqa: ANN001 - PluginHost, optional (01b)
    ):
        self._client_factory = client_factory
        self._listening_prefs_store = listening_prefs_store
        self._play_history_store = play_history_store
        self._scrobble_outbox_store = scrobble_outbox_store
        self._plugin_host = plugin_host
        self._dedup_cache: dict[str, float] = {}
        self._plugin_tasks: set[asyncio.Task] = set()
        self._outbox_retry_lock = asyncio.Lock()

    def _dispatch_to_plugins(self, request: ScrobbleRequest, user_id: str = "") -> None:
        """Fan the accepted play out to scrobbler + subscriber plugins, fire-and-forget so a
        slow plugin never adds latency to the client's scrobble call.

        Same gate as external forwarding: short tracks (<30s) and Navidrome-delegated
        plays never reach here (early returns above), so neither ``scrobble`` nor
        ``playback_started`` fires for them. The ``scrobble`` kind goes through
        ``dispatch_event`` ONLY: the host fans it to subscriber plugins and to v0
        ``scrobbler`` plugins exactly once (a direct ``dispatch_scrobble`` call here
        as well would fire ``on_scrobble`` twice)."""
        if self._plugin_host is None:
            return
        from infrastructure.plugins.protocols import PlaybackEvent, PluginEvent, ScrobbleEvent

        event = ScrobbleEvent(
            artist=request.artist_name,
            track=request.track_name,
            album=request.album_name,
            timestamp=request.timestamp,
            duration_ms=request.duration_ms or None,
            recording_mbid=request.mbid,
        )
        try:
            scrobble_event = PluginEvent(
                kind="scrobble", payload=event, causation_id=uuid.uuid4().hex
            )
            task = asyncio.create_task(self._plugin_host.dispatch_event(scrobble_event))
            self._plugin_tasks.add(task)
            task.add_done_callback(self._plugin_tasks.discard)
        except Exception:  # noqa: BLE001 - events never break scrobbling
            pass
        try:
            playback = PlaybackEvent(
                artist=request.artist_name,
                track=request.track_name,
                album=request.album_name,
                user_id=user_id,
            )
            playback_event = PluginEvent(
                kind="playback_started", payload=playback, causation_id=uuid.uuid4().hex
            )
            task = asyncio.create_task(self._plugin_host.dispatch_event(playback_event))
            self._plugin_tasks.add(task)
            task.add_done_callback(self._plugin_tasks.discard)
        except Exception:  # noqa: BLE001 - events never break scrobbling
            pass

    def _dedup_key(self, user_id: str, artist: str, track: str, timestamp: int) -> str:
        # user_id-scoped so two users scrobbling the same track aren't cross-deduped
        return f"{user_id}::{artist.lower()}::{track.lower()}::{timestamp}"

    def _is_duplicate(self, key: str) -> bool:
        entry_time = self._dedup_cache.get(key)
        if entry_time is None:
            return False
        return (time.time() - entry_time) < DEDUP_TTL_SECONDS

    def _record_dedup(self, key: str) -> None:
        self._dedup_cache[key] = time.time()
        if len(self._dedup_cache) > DEDUP_MAX_ENTRIES:
            now = time.time()
            expired = [
                k for k, v in self._dedup_cache.items()
                if (now - v) >= DEDUP_TTL_SECONDS
            ]
            for k in expired:
                del self._dedup_cache[k]
            if len(self._dedup_cache) > DEDUP_MAX_ENTRIES:
                oldest = sorted(self._dedup_cache, key=self._dedup_cache.get)  # type: ignore[arg-type]
                for k in oldest[: len(self._dedup_cache) - DEDUP_MAX_ENTRIES]:
                    del self._dedup_cache[k]

    async def report_now_playing(
        self, request: NowPlayingRequest, *, user_id: str
    ) -> ScrobbleResponse:
        prefs = await self._listening_prefs_store.get(user_id)
        if request.source == "navidrome" and prefs.navidrome_handles_external_scrobbles:
            return ScrobbleResponse(accepted=True, services={})

        tasks: dict[str, Any] = {}
        duration_sec = request.duration_ms // 1000 if request.duration_ms > 0 else 0

        if prefs.scrobble_to_lastfm:
            lastfm = await self._client_factory.resolve_lastfm(user_id)
            if lastfm is not None:
                tasks["lastfm"] = lastfm.update_now_playing(
                    artist=request.artist_name,
                    track=request.track_name,
                    album=request.album_name,
                    duration=duration_sec,
                    mbid=request.mbid,
                )

        if prefs.scrobble_to_listenbrainz:
            listenbrainz = await self._client_factory.resolve_listenbrainz(user_id)
            if listenbrainz is not None:
                tasks["listenbrainz"] = listenbrainz.submit_now_playing(
                    artist_name=request.artist_name,
                    track_name=request.track_name,
                    release_name=request.album_name,
                    duration_ms=request.duration_ms,
                )

        if not tasks:
            return ScrobbleResponse(accepted=False, services={})

        services, any_success = await self._gather_results(tasks, "Now playing report failed")
        return ScrobbleResponse(accepted=any_success, services=services)

    async def submit_scrobble(
        self, request: ScrobbleRequest, *, user_id: str
    ) -> ScrobbleResponse:
        dedup = self._dedup_key(
            user_id, request.artist_name, request.track_name, request.timestamp
        )
        if self._is_duplicate(dedup):
            logger.debug(
                "Duplicate scrobble skipped for %s: %s - %s at %d",
                user_id,
                request.artist_name,
                request.track_name,
                request.timestamp,
            )
            return ScrobbleResponse(accepted=True, services={})

        prefs = await self._listening_prefs_store.get(user_id)
        short_track = 0 < request.duration_ms < MIN_TRACK_DURATION_MS
        delegated = (
            request.source == "navidrome"
            and prefs.navidrome_handles_external_scrobbles
        )
        durable_listenbrainz = bool(
            self._scrobble_outbox_store is not None
            and prefs.scrobble_to_listenbrainz
            and not short_track
            and not delegated
        )

        outbox_result: ListenBrainzOutboxEnqueueResult | None = None
        if durable_listenbrainz:
            outbox_result = await self._record_play_and_enqueue_listenbrainz(
                request, user_id
            )
            play_history_id = outbox_result.play_history_id
        else:
            # Every accepted play is stored locally, including short tracks and
            # plays whose external forwarding is delegated to Navidrome.
            play_history_id = await self._record_play_history(request, user_id)
        self._record_dedup(dedup)

        # short-track gate blocks only external forwarding; history is already saved
        if short_track:
            logger.debug(
                "Short track (%dms) recorded locally, not forwarded: %s - %s",
                request.duration_ms,
                request.artist_name,
                request.track_name,
            )
            return ScrobbleResponse(accepted=True, services={})

        if delegated:
            logger.debug(
                "External scrobble forwarding delegated to Navidrome for user %s",
                user_id,
            )
            return ScrobbleResponse(accepted=True, services={})

        # The database-level delivery key survives restarts.  A replay of an
        # already accepted client scrobble must not append history or call any
        # external service again.
        if outbox_result is not None and not outbox_result.created:
            return ScrobbleResponse(
                accepted=True,
                services={
                    "listenbrainz": self._service_result_for_status(
                        outbox_result.status
                    )
                },
            )

        self._dispatch_to_plugins(request, user_id)

        tasks: dict[str, Any] = {}
        duration_sec = request.duration_ms // 1000 if request.duration_ms > 0 else 0

        if prefs.scrobble_to_lastfm:
            lastfm = await self._client_factory.resolve_lastfm(user_id)
            if lastfm is not None:
                tasks["lastfm"] = lastfm.scrobble(
                    artist=request.artist_name,
                    track=request.track_name,
                    timestamp=request.timestamp,
                    album=request.album_name,
                    duration=duration_sec,
                    mbid=request.mbid,
                )

        if prefs.scrobble_to_listenbrainz:
            listenbrainz = await self._client_factory.resolve_listenbrainz(user_id)
            if listenbrainz is not None:
                tasks["listenbrainz"] = self._deliver_listenbrainz_scrobble(
                    listenbrainz,
                    play_history_id=play_history_id,
                    artist_name=request.artist_name,
                    track_name=request.track_name,
                    listened_at=request.timestamp,
                    album_name=request.album_name,
                    duration_ms=request.duration_ms,
                    queued=outbox_result is not None,
                    attempts=0,
                )
            elif outbox_result is not None:
                tasks["listenbrainz"] = self._completed_service_result(
                    ServiceResult(
                        success=False,
                        status="pending",
                        error="ListenBrainz delivery queued until the account is linked",
                    )
                )

        if not tasks:
            # no linked/enabled external account - the play is still recorded
            return ScrobbleResponse(accepted=True, services={})

        # play is already recorded locally, so accepted=True regardless of the external
        # outcome; per-service results are reported in `services`
        services, _ = await self._gather_results(tasks, "Scrobble submission failed")
        return ScrobbleResponse(accepted=True, services=services)

    async def _record_play_history(self, request: ScrobbleRequest, user_id: str) -> str:
        played_at = datetime.fromtimestamp(request.timestamp, tz=timezone.utc).isoformat()
        return await self._play_history_store.insert(
            user_id,
            track_name=request.track_name,
            artist_name=request.artist_name,
            album_name=request.album_name or None,
            recording_mbid=request.mbid,
            release_group_mbid=request.release_group_mbid,
            duration_ms=request.duration_ms or None,
            source=request.source,
            played_at=played_at,
        )

    @staticmethod
    def _delivery_key(request: ScrobbleRequest, user_id: str) -> str:
        normalized = "\x1f".join(
            (
                user_id,
                request.artist_name.strip().casefold(),
                request.track_name.strip().casefold(),
                str(request.timestamp),
            )
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    async def _record_play_and_enqueue_listenbrainz(
        self, request: ScrobbleRequest, user_id: str
    ) -> ListenBrainzOutboxEnqueueResult:
        assert self._scrobble_outbox_store is not None
        played_at = datetime.fromtimestamp(request.timestamp, tz=timezone.utc).isoformat()
        return await self._play_history_store.insert_with_listenbrainz_outbox(
            user_id,
            delivery_key=self._delivery_key(request, user_id),
            track_name=request.track_name,
            artist_name=request.artist_name,
            album_name=request.album_name,
            recording_mbid=request.mbid,
            release_group_mbid=request.release_group_mbid,
            duration_ms=request.duration_ms,
            source=request.source,
            played_at=played_at,
            listened_at=request.timestamp,
        )

    @staticmethod
    def _listenbrainz_retry_delay(error: Exception, attempts: int) -> float | None:
        if isinstance(error, NonRetriableExternalServiceError):
            return None
        if not isinstance(error, ExternalServiceError):
            return None
        backoff = min(
            LISTENBRAINZ_RETRY_MAX_SECONDS,
            LISTENBRAINZ_RETRY_BASE_SECONDS * (2 ** min(attempts, 7)),
        )
        backoff = random.uniform(backoff * 0.8, backoff * 1.2)
        if isinstance(error, RateLimitedError) and error.retry_after_seconds is not None:
            backoff = max(backoff, float(error.retry_after_seconds))
        return backoff

    async def _deliver_listenbrainz_scrobble(
        self,
        listenbrainz,
        *,
        play_history_id: str,
        artist_name: str,
        track_name: str,
        listened_at: int,
        album_name: str,
        duration_ms: int,
        queued: bool,
        attempts: int,
    ) -> ServiceResult:
        try:
            await listenbrainz.submit_single_listen(
                artist_name=artist_name,
                track_name=track_name,
                listened_at=listened_at,
                release_name=album_name,
                duration_ms=duration_ms,
            )
        except Exception as error:  # noqa: BLE001 - classify provider and transport failures
            if queued and self._scrobble_outbox_store is not None:
                delay = self._listenbrainz_retry_delay(error, attempts)
                if delay is None:
                    await self._scrobble_outbox_store.mark_failed(
                        play_history_id, error=str(error)
                    )
                else:
                    await self._scrobble_outbox_store.defer(
                        play_history_id,
                        delay_seconds=delay,
                        increment_attempts=True,
                    )
                    return ServiceResult(
                        success=False, status="pending", error=str(error)
                    )
            return ServiceResult(success=False, status="failed", error=str(error))
        if queued and self._scrobble_outbox_store is not None:
            await self._scrobble_outbox_store.mark_delivered(play_history_id)
        return ServiceResult(success=True, status="delivered")

    @staticmethod
    async def _completed_service_result(result: ServiceResult) -> ServiceResult:
        return result

    @staticmethod
    def _service_result_for_status(status: str) -> ServiceResult:
        if status == "delivered":
            return ServiceResult(success=True, status="delivered")
        if status == "pending":
            return ServiceResult(success=False, status="pending")
        return ServiceResult(success=False, status="failed")

    async def retry_pending_listenbrainz(self, *, limit: int = 25) -> tuple[int, int]:
        """Retry due completed listens, resolving current per-user credentials."""
        if self._scrobble_outbox_store is None:
            return 0, 0
        delivered = 0
        deferred = 0
        async with self._outbox_retry_lock:
            rows = await self._scrobble_outbox_store.due(limit=limit)
            for row in rows:
                prefs = await self._listening_prefs_store.get(row.user_id)
                listenbrainz = (
                    await self._client_factory.resolve_listenbrainz(row.user_id)
                    if prefs.scrobble_to_listenbrainz
                    else None
                )
                if listenbrainz is None:
                    await self._scrobble_outbox_store.defer(
                        row.play_history_id,
                        delay_seconds=LISTENBRAINZ_UNLINKED_RETRY_SECONDS,
                        increment_attempts=False,
                    )
                    deferred += 1
                    continue
                try:
                    result = await self._deliver_pending_listenbrainz(listenbrainz, row)
                except Exception:  # noqa: BLE001 - one row cannot stop the queue
                    logger.exception("ListenBrainz queued scrobble persistence failed")
                    deferred += 1
                else:
                    if result.status == "delivered":
                        delivered += 1
                    elif result.status == "pending":
                        deferred += 1
                    else:
                        logger.warning(
                            "ListenBrainz queued scrobble permanently failed: %s",
                            result.error,
                        )
        return delivered, deferred

    async def _deliver_pending_listenbrainz(
        self, listenbrainz, row: PendingListenBrainzScrobble
    ) -> ServiceResult:
        return await self._deliver_listenbrainz_scrobble(
            listenbrainz,
            play_history_id=row.play_history_id,
            artist_name=row.artist_name,
            track_name=row.track_name,
            listened_at=row.listened_at,
            album_name=row.album_name,
            duration_ms=row.duration_ms,
            queued=True,
            attempts=row.attempts,
        )

    @staticmethod
    async def _gather_results(
        tasks: dict[str, Any], failure_msg: str
    ) -> tuple[dict[str, ServiceResult], bool]:
        results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)
        services: dict[str, ServiceResult] = {}
        any_success = False

        for service_name, result in zip(tasks.keys(), results_list):
            if isinstance(result, BaseException):
                logger.warning("%s for %s: %s", failure_msg, service_name, result)
                services[service_name] = ServiceResult(
                    success=False, status="failed", error=str(result)
                )
            elif isinstance(result, ServiceResult):
                services[service_name] = result
                any_success = any_success or result.success
            else:
                services[service_name] = ServiceResult(
                    success=True, status="delivered"
                )
                any_success = True

        return services, any_success
