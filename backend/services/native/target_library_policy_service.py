"""Target-only settings boundary with durable policy reconciliation state."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import msgspec

from api.v1.schemas.library_policies import (
    LibraryPolicyApplyPreviewResponse,
    LibraryPolicyApplyRequest,
    LibraryPolicyImpactRequest,
    LibraryPolicyImpactResponse,
    LibraryRestorableRoot,
    LibraryRestorableRootsResponse,
    LibraryRestoreRootsRequest,
    LibraryRootSettings,
    LibrarySettingsResponse,
    LibraryPolicyTreeResponse,
    TypedLibrarySettings,
)
from core.exceptions import StaleRevisionError, TargetStartupInvariantError, ValidationError
from infrastructure.persistence.native_library_store import NativeLibraryStore
from services.native.library_policy_reconciliation_service import (
    LibraryPolicyReconciliationService,
)
from services.native.library_policy_service import LibraryPolicyService
from services.native.library_policy_resolver import LibraryPolicyResolver


class TargetLibraryPolicyService:
    def __init__(
        self,
        settings: LibraryPolicyService,
        reconciliation: LibraryPolicyReconciliationService,
        store: NativeLibraryStore,
        *,
        on_settings_saved: Callable[[], None] | None = None,
        transition_lock: asyncio.Lock | None = None,
        _cohort_settings_candidate_enabled: bool = False,
        _cohort_filesystem=None,
        _cohort_recovery_candidate_enabled: bool = False,
    ) -> None:
        if _cohort_recovery_candidate_enabled and not _cohort_settings_candidate_enabled:
            raise ValidationError('Private cohort recovery requires settings candidate opt-in.')
        if _cohort_settings_candidate_enabled and (transition_lock is None or _cohort_filesystem is None):
            raise ValidationError('Private cohort settings require explicit shared lock and coordinator.')
        if _cohort_settings_candidate_enabled and reconciliation._store is not store:
            raise ValidationError('Private cohort settings require one native transaction owner.')
        self._settings = settings
        self._reconciliation = reconciliation
        self._store = store
        self._on_settings_saved = on_settings_saved
        self._save_lock = transition_lock or asyncio.Lock()
        self._cohort_settings_candidate_enabled = _cohort_settings_candidate_enabled
        self._cohort_filesystem = _cohort_filesystem
        self._cohort_recovery_candidate_enabled = _cohort_recovery_candidate_enabled

    async def _recover_cohort_transition_candidate(self):
        if not self._store._cohort_policy_transition_enabled or not self._store._cohort_policy_invalidation_enabled:
            raise ValidationError('Private cohort store permissions required for recovery.')
        transition = await self._store.get_policy_transition()
        if transition is None or transition['state'] != 'prepared':
            return False
        previous = LibraryPolicyResolver(msgspec.json.decode(transition['previous_settings_json'], type=TypedLibrarySettings))
        proposed = LibraryPolicyResolver(msgspec.json.decode(transition['proposed_settings_json'], type=TypedLibrarySettings))
        if (previous.policy_revision != transition['previous_policy_revision']
                or proposed.policy_revision != transition['proposed_policy_revision']):
            raise TargetStartupInvariantError('Private recovery settings and prepared revisions disagree.')
        roots = sorted({root.id for root in previous.settings.library_roots + proposed.settings.library_roots})
        async with self._cohort_filesystem.write_many(roots):
            async with self._save_lock:
                if await self._store.get_policy_transition() != transition:
                    raise StaleRevisionError('Prepared cohort recovery changed under root leases.')
                current = LibraryPolicyResolver(self._settings.current_settings()).policy_revision
                if current == previous.policy_revision:
                    import hashlib
                    import json
                    import time
                    def abort_seal(connection):
                        row = connection.execute('SELECT * FROM library_policy_transitions WHERE singleton=1').fetchone()
                        keys = ('previous_policy_revision', 'proposed_policy_revision', 'previous_settings_json',
                            'proposed_settings_json', 'state', 'prepared_at', 'completed_at')
                        if (row is None or any(row[key] != transition[key] for key in keys)
                                or json.loads(row['scopes_json']) != msgspec.to_builtins(transition['scopes'])):
                            raise StaleRevisionError('Private recovery abort journal changed.')
                        return (previous.policy_revision, hashlib.sha256(json.dumps(dict(row), sort_keys=True).encode()).hexdigest())
                    seal = await self._store._read(abort_seal)
                    from services.native.import_cohort_scan_candidate import settle_cohort_write_candidate
                    await settle_cohort_write_candidate(self._store.abort_policy_transition(
                        proposed_policy_revision=proposed.policy_revision, aborted_at=time.time(),
                        _cohort_recovery_abort_seal=seal))
                    return True
                if current != proposed.policy_revision:
                    raise TargetStartupInvariantError('Private recovery configuration matches neither prepared revision.')
                if self._on_settings_saved is not None:
                    self._on_settings_saved()
                from services.native.library_management_publisher import LibraryManagementPublisher
                _, cancelled = await LibraryManagementPublisher._finish_critical_task(asyncio.create_task(
                    self._commit_settings_boundary_candidate(proposed.policy_revision)))
                self._settings.mark_scan_scopes_dirty(LibraryPolicyService._scope_ids(transition['scopes']))
                if cancelled:
                    raise asyncio.CancelledError
                return True

    @asynccontextmanager
    async def _settings_save_scope_candidate(self, settings):
        if not self._cohort_settings_candidate_enabled:
            async with self._save_lock:
                yield
            return
        if not self._store._cohort_policy_transition_enabled or not self._store._cohort_policy_invalidation_enabled:
            raise ValidationError('Private cohort store permissions required for settings.')
        previous = LibraryPolicyResolver(self._settings.current_settings())
        proposed = LibraryPolicyResolver(settings)
        root_ids = sorted({root.id for root in previous.settings.library_roots + proposed.settings.library_roots})
        async with self._cohort_filesystem.write_many(root_ids):
            async with self._save_lock:
                if LibraryPolicyResolver(self._settings.current_settings()).policy_revision != previous.policy_revision:
                    raise StaleRevisionError('Settings changed while acquiring cohort root leases.')
                yield

    async def _commit_settings_boundary_candidate(self, proposed_revision):
        if not self._cohort_settings_candidate_enabled:
            return await self._reconciliation.commit_boundary(proposed_policy_revision=proposed_revision)
        import time
        from infrastructure.persistence.import_cohort_binding import capture_cohort_policy_transition_candidate
        current = LibraryPolicyResolver(self._settings.current_settings()).policy_revision
        if current != proposed_revision or not self._save_lock.locked():
            raise StaleRevisionError('Private settings configuration or policy scope changed.')
        def discover(connection):
            if not connection.in_transaction:
                connection.execute('BEGIN')
            return capture_cohort_policy_transition_candidate(connection, current)
        seal = await self._store._read(discover)
        if not all(self._cohort_filesystem._cohort_writer_is_held(root) for root in seal.root_ids):
            raise ValidationError('Private settings previous/proposed root leases required.')
        return await self._store.commit_policy_transition(proposed_policy_revision=proposed_revision,
            updated_at=time.time(), _cohort_policy_transition_candidate=seal)

    async def _abort_settings_boundary_candidate(self, proposed_revision):
        from services.native.library_management_publisher import LibraryManagementPublisher
        operation = self._reconciliation.abort_boundary(proposed_policy_revision=proposed_revision)
        result, cancelled = await LibraryManagementPublisher._finish_critical_task(asyncio.create_task(operation))
        if cancelled:
            raise asyncio.CancelledError
        return result

    @staticmethod
    def _settings_json(settings: TypedLibrarySettings) -> str:
        payload = msgspec.to_builtins(settings)
        payload["acoustid_api_key"] = ""
        return msgspec.json.encode(payload).decode()

    async def recover_pending_transition(self) -> bool:
        if self._cohort_settings_candidate_enabled and self._cohort_recovery_candidate_enabled:
            return await self._recover_cohort_transition_candidate()
        async with self._save_lock:
            transition = await self._store.get_policy_transition()
            if transition is None or transition["state"] != "prepared":
                return False
            if self._cohort_settings_candidate_enabled:
                raise TargetStartupInvariantError('Private cohort startup recovery is not implemented; prepared evidence retained.')
            current_revision = LibraryPolicyResolver(
                self._settings.current_settings()
            ).policy_revision
            proposed_revision = transition["proposed_policy_revision"]
            if current_revision == transition["previous_policy_revision"]:
                await self._abort_settings_boundary_candidate(proposed_revision)
                return True
            if current_revision != proposed_revision:
                raise TargetStartupInvariantError(
                    "The library policy transition does not match the saved configuration."
                )
            if self._on_settings_saved is not None:
                self._on_settings_saved()
            from services.native.library_management_publisher import LibraryManagementPublisher
            _, cancelled = await LibraryManagementPublisher._finish_critical_task(asyncio.create_task(
                self._reconciliation.commit_boundary(proposed_policy_revision=proposed_revision)))
            if cancelled:
                raise asyncio.CancelledError
            return True

    async def get_settings(self) -> LibrarySettingsResponse:
        response = self._settings.get_settings()
        pending = await self._store.get_pending_policy()
        if pending is None or not pending["pending_scope_ids"]:
            return response
        payload = msgspec.to_builtins(response)
        payload.update(
            {
                "reconciliation_required": True,
                "reconciliation_state": "awaiting_reconciliation",
                "pending_policy_revision": pending["desired_policy_revision"],
                "affected_scope_ids": pending["pending_scope_ids"],
            }
        )
        return LibrarySettingsResponse(**payload)

    async def save_settings(
        self,
        settings: TypedLibrarySettings,
        *,
        expected_policy_revision: str,
    ) -> LibrarySettingsResponse:
        async with self._settings_save_scope_candidate(settings):
            previous_settings = self._settings.current_settings()
            previous_settings_raw = self._settings.current_settings_raw()
            previous_revision = LibraryPolicyResolver(previous_settings).policy_revision
            proposed, changed_scopes = self._settings.prepare_change(
                settings,
                expected_policy_revision=expected_policy_revision,
            )
            if (
                not proposed.settings.library_roots
                and await self._store.catalog_has_tracks()
            ):
                raise ValidationError(
                    "Removing every library root would orphan the existing catalog. "
                    "Keep at least one root, or set its policy to Excluded instead."
                )
            previous_pending = await self._store.get_pending_policy()
            pending_scopes = (
                self._settings.rebase_scopes(
                    previous_pending["pending_scopes"], proposed
                )
                if previous_pending is not None
                else []
            )
            merged = {
                (scope.root_id, scope.relative_path): scope
                for scope in [*pending_scopes, *changed_scopes]
            }
            scopes = self._settings.collapse_scopes(list(merged.values()))
            prepare_task = asyncio.create_task(
                self._reconciliation.prepare_boundary(
                    previous_policy_revision=previous_revision,
                    proposed_policy_revision=proposed.policy_revision,
                    previous_settings_json=self._settings_json(previous_settings),
                    proposed_settings_json=self._settings_json(proposed.settings),
                    scopes=scopes,
                )
            )
            from services.native.library_management_publisher import LibraryManagementPublisher
            # The ordinary path owns the same durable boundary: repeated caller
            # cancellation must not abandon its prepare/abort/commit task either.
            _, cancelled = await LibraryManagementPublisher._finish_critical_task(prepare_task)
            if cancelled:
                await self._abort_settings_boundary_candidate(proposed.policy_revision)
                raise asyncio.CancelledError
            config_persisted = False
            try:
                self._settings.persist_settings(
                    proposed.settings,
                    expected_policy_revision=expected_policy_revision,
                )
                config_persisted = True
                if self._on_settings_saved is not None:
                    self._on_settings_saved()
            except Exception:
                if config_persisted:
                    self._settings.persist_settings(
                        previous_settings_raw,
                        expected_policy_revision=proposed.policy_revision,
                    )
                    if self._on_settings_saved is not None:
                        self._on_settings_saved()
                await self._abort_settings_boundary_candidate(proposed.policy_revision)
                raise
            commit_task = asyncio.create_task(
                self._commit_settings_boundary_candidate(proposed.policy_revision)
            )
            cancelled = False
            try:
                immediate, cancelled = await LibraryManagementPublisher._finish_critical_task(commit_task)
            except Exception:
                self._settings.persist_settings(
                    previous_settings_raw,
                    expected_policy_revision=proposed.policy_revision,
                )
                if self._on_settings_saved is not None:
                    self._on_settings_saved()
                await self._abort_settings_boundary_candidate(proposed.policy_revision)
                raise
            # S-01 Hook B writer: the restore path funnels through here too,
            # so both settings entry points mark affected scopes dirty. A durable
            # commit still needs its hint when caller cancellation is propagated.
            self._settings.mark_scan_scopes_dirty(
                LibraryPolicyService._scope_ids(scopes)
            )
            if cancelled:
                raise asyncio.CancelledError
            current = await self.get_settings()
            payload = msgspec.to_builtins(current)
            payload["actions_applied"] = [
                "Settings saved. No library work was started.",
                (
                    f"{immediate['cancelled']} queued identification "
                    f"job{'s were' if immediate['cancelled'] != 1 else ' was'} stopped "
                    "because the new policy no longer allows the work."
                ),
            ]
            return LibrarySettingsResponse(**payload)

    @staticmethod
    def _restored_root_label(path: str, used: set[str]) -> str:
        base = Path(path).name or Path(path).anchor or "Library"
        label = base
        number = 2
        while label.casefold() in used:
            label = f"{base} ({number})"
            number += 1
        used.add(label.casefold())
        return label

    async def _known_root_paths(self) -> dict[str, str]:
        # pending scopes freeze the paths from before the wipe
        pending = await self._store.get_pending_policy()
        if pending is None:
            return {}
        return {
            str(scope.root_id): str(scope.root_path)
            for scope in pending["pending_scopes"]
            if scope.relative_path == "." and scope.root_path
        }

    async def _restorable_paths(
        self, missing: list[str]
    ) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
        return (
            await self._known_root_paths(),
            await self._store.get_restorable_root_paths(missing),
        )

    async def restorable_roots(self) -> LibraryRestorableRootsResponse:
        migrated = await self._store.get_migrated_root_ids()
        configured = {
            root.id for root in self._settings.current_settings().library_roots
        }
        missing = sorted(migrated - configured)
        known, derived = await self._restorable_paths(missing)
        roots = []
        for root_id in missing:
            info = derived.get(root_id)
            path = known.get(root_id)
            if path is None and info is None:
                continue
            path = path if path is not None else str(info["path"])
            count = int(info["indexed_file_count"]) if info is not None else 0
            roots.append(
                LibraryRestorableRoot(
                    root_id=root_id,
                    path=path,
                    indexed_file_count=count,
                )
            )
        return LibraryRestorableRootsResponse(
            policy_revision=LibraryPolicyResolver(
                self._settings.current_settings()
            ).policy_revision,
            restorable_roots=roots,
        )

    async def restore_roots(
        self, request: LibraryRestoreRootsRequest
    ) -> LibrarySettingsResponse:
        current = self._settings.current_settings()
        migrated = await self._store.get_migrated_root_ids()
        configured = {root.id for root in current.library_roots}
        missing = sorted(migrated - configured)
        if not missing:
            raise ValidationError("There are no removed library roots to restore.")
        overrides = request.paths or {}
        known, derived = await self._restorable_paths(missing)
        used_labels = {root.label.casefold() for root in current.library_roots}
        roots = list(current.library_roots)
        for root_id in missing:
            info = derived.get(root_id)
            path = overrides.get(root_id) or known.get(root_id)
            if path is None and info is None:
                continue
            path = path if path is not None else str(info["path"])
            roots.append(
                LibraryRootSettings(
                    id=root_id,
                    path=path,
                    label=self._restored_root_label(path, used_labels),
                    policy="automatic",
                    rules=[],
                )
            )
        if len(roots) == len(current.library_roots):
            raise ValidationError(
                "The removed library roots have no catalog files to recover "
                "their path from."
            )
        return await self.save_settings(
            TypedLibrarySettings(
                library_roots=roots,
                staging_path=current.staging_path,
                naming_template=current.naming_template,
                acoustid_api_key=current.acoustid_api_key,
                enabled=current.enabled,
            ),
            expected_policy_revision=request.expected_policy_revision,
        )

    async def policy_tree(self) -> LibraryPolicyTreeResponse:
        tree = self._settings.policy_tree()
        scopes = [
            (root.id, "." if node.kind == "root" else node.path)
            for root in tree.roots
            for node in [root, *root.children]
        ]
        counts = await self._store.get_policy_scope_counts(scopes)
        payload = msgspec.to_builtins(tree)
        for root in payload["roots"]:
            indexed, on_disk = counts[(root["id"], ".")]
            root["indexed_file_count"] = indexed
            root["on_disk_file_count"] = on_disk
            for child in root["children"]:
                indexed, on_disk = counts[(root["id"], child["path"])]
                child["indexed_file_count"] = indexed
                child["on_disk_file_count"] = on_disk
        return msgspec.convert(payload, type=LibraryPolicyTreeResponse)

    async def preview_impact(
        self, request: LibraryPolicyImpactRequest
    ) -> LibraryPolicyImpactResponse:
        response = self._settings.preview_impact(request)
        scopes = self._settings.preview_scopes(request.settings)
        indexed, on_disk = await self._store.get_policy_scope_total_counts(scopes)
        payload = msgspec.to_builtins(response)
        payload["indexed_file_count"] = indexed
        payload["on_disk_file_count"] = on_disk
        return LibraryPolicyImpactResponse(**payload)

    async def preview_apply(
        self, request: LibraryPolicyApplyRequest
    ) -> LibraryPolicyApplyPreviewResponse:
        preview = await self._reconciliation.preview_apply(
            request.scope_ids,
            expected_policy_revision=request.expected_policy_revision,
        )
        pending = await self._store.get_pending_policy()
        return LibraryPolicyApplyPreviewResponse(
            policy_revision=str(preview["policy_revision"]),
            scope_ids=[str(value) for value in preview["scope_ids"]],
            estimated_file_count=int(preview["estimated_file_count"]),
            content_will_become_unavailable=any(
                scope.effective_policy == "excluded" for scope in preview["scopes"]
            ),
            queued_work_was_cancelled_on_save=bool(
                pending and pending["cancelled_work_count"]
            ),
        )
