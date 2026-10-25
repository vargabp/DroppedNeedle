"""Private actual settings-save path; simulated preferences, real native SQL."""
import asyncio
import importlib
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import msgspec
from core.exceptions import StaleRevisionError, TargetStartupInvariantError, ValidationError
from services.native.library_policy_resolver import LibraryPolicyResolver
from services.native.library_policy_service import LibraryPolicyService
from services.native.library_policy_reconciliation_service import LibraryPolicyReconciliationService
from services.native.target_library_policy_service import TargetLibraryPolicyService


class CohortSettingsSaveTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module = importlib.import_module('tests.services.native.test_import_cohort_invalidation')
        self.transition = module.CohortPolicyTransitionTests()
        await self.transition.asyncSetUp()
        self.addCleanup(self.transition.doCleanups)
        self.fixture = self.transition.base
        self.store, self.lock = self.transition.store, self.transition.lock
        self.filesystem = self.fixture.filesystem
        self.current = self.fixture.old.settings
        self.proposed = self.fixture.current.settings
        self.persisted, self.dirty = [], []
        preferences = SimpleNamespace(
            get_typed_library_settings=lambda: self.current,
            get_typed_library_settings_raw=lambda: self.current,
            save_typed_library_settings_if_current=self.persist,
            mark_library_scan_dirty_scopes=lambda scopes: self.dirty.append(scopes))
        self.settings = LibraryPolicyService(preferences, None, self.resolver, lambda: None)
        self.coordinator = SimpleNamespace(request_run=AsyncMock(side_effect=AssertionError('No scan authorized')))
        self.reconciliation = LibraryPolicyReconciliationService(self.store, self.resolver, self.coordinator)
        self.service = self.make_service()

    def resolver(self):
        return LibraryPolicyResolver(self.current)

    def persist(self, settings, *, expected_policy_revision):
        if self.resolver().policy_revision != expected_policy_revision:
            raise StaleRevisionError('Fixture preferences revision changed')
        self.persisted.append(settings)
        self.current = settings

    def make_service(self, *, store=None, enabled=True):
        reconciliation = self.reconciliation if store is None else LibraryPolicyReconciliationService(
            store, self.resolver, self.coordinator)
        return TargetLibraryPolicyService(self.settings, reconciliation, store or self.store,
            transition_lock=self.lock, _cohort_settings_candidate_enabled=enabled,
            _cohort_filesystem=self.filesystem)

    async def save(self, settings=None, revision=None):
        return await self.service.save_settings(settings or self.proposed,
            expected_policy_revision=revision or self.fixture.old.policy_revision)

    async def assert_aborted_without_effects(self, before):
        self.assertEqual(self.current, self.fixture.old.settings)
        self.assertEqual((await self.store.get_policy_transition())['state'], 'aborted')
        self.assertEqual((await self.store.get_pending_policy())['desired_policy_revision'], self.fixture.old.policy_revision)
        self.assertEqual(self.fixture.events(), [])
        self.assertEqual(self.fixture.rows(), before)
        self.assertEqual(self.dirty, [], 'Aborted saves must not emit successful-commit hints')
        self.coordinator.request_run.assert_not_awaited()

    async def test_actual_settings_save_commits_config_boundary_and_invalidation(self):
        before = self.fixture.rows()
        response = await self.save()
        self.assertEqual(response.policy_revision, self.fixture.current.policy_revision)
        self.assertEqual(self.current, self.proposed)
        self.assertEqual(len(self.fixture.events()), 1)
        self.assertEqual((await self.store.get_policy_transition())['state'], 'completed')
        self.assertEqual((await self.store.get_pending_policy())['desired_policy_revision'], response.policy_revision)
        self.assertEqual(self.fixture.rows(), before, 'Unrelated rule changes must not change owned catalog')
        self.assertEqual(len(self.persisted), 1)
        self.assertTrue(self.dirty)
        self.coordinator.request_run.assert_not_awaited()

    async def test_configuration_write_failure_aborts_prepared_boundary(self):
        before = self.fixture.rows()
        with patch.object(self.settings, 'persist_settings', side_effect=RuntimeError('Injected config failure')):
            with self.assertRaisesRegex(RuntimeError, 'Injected config failure'):
                await self.save()
        await self.assert_aborted_without_effects(before)

    async def test_settings_saved_callback_failure_restores_config_then_aborts(self):
        before = self.fixture.rows()
        def changed():
            if self.current == self.proposed:
                raise RuntimeError('Injected callback failure')
        self.service._on_settings_saved = changed
        with self.assertRaisesRegex(RuntimeError, 'Injected callback failure'):
            await self.save()
        self.assertEqual(len(self.persisted), 2)
        await self.assert_aborted_without_effects(before)

    async def test_native_commit_failure_restores_configuration_and_aborts(self):
        before = self.fixture.rows()
        with patch.object(self.store, 'commit_policy_transition', side_effect=RuntimeError('Injected native commit failure')):
            with self.assertRaisesRegex(RuntimeError, 'Injected native commit failure'):
                await self.save()
        self.assertEqual(len(self.persisted), 2)
        await self.assert_aborted_without_effects(before)

    async def test_post_invalidation_sql_failure_rolls_back_config_and_catalog(self):
        import sqlite3
        before = self.fixture.rows()
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER fail_settings_complete BEFORE UPDATE ON library_policy_transitions WHEN NEW.state='completed' BEGIN SELECT RAISE(ABORT,'Injected late settings failure'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'Injected late settings failure'):
            await self.save()
        await self.assert_aborted_without_effects(before)

    async def test_original_stale_revision_guard_is_preserved(self):
        with self.assertRaises(StaleRevisionError):
            await self.save(revision='stale')
        self.assertEqual(self.persisted, [])
        self.assertIsNone(await self.store.get_policy_transition())
        self.assertEqual(self.fixture.events(), [])

    async def test_original_orphan_root_guard_is_preserved(self):
        settings = msgspec.structs.replace(self.current, library_roots=[])
        with self.assertRaisesRegex(ValidationError, 'orphan'):
            await self.save(settings)
        self.assertEqual(self.persisted, [])
        self.assertIsNone(await self.store.get_policy_transition())

    async def test_constructor_requires_explicit_shared_lock_and_coordinator(self):
        with self.assertRaisesRegex(ValidationError, 'explicit shared lock'):
            TargetLibraryPolicyService(self.settings, self.reconciliation, self.store,
                _cohort_settings_candidate_enabled=True)

    async def test_constructor_refuses_mismatched_native_transaction_owners(self):
        with self.assertRaisesRegex(ValidationError, 'one native transaction owner'):
            TargetLibraryPolicyService(self.settings, self.reconciliation, self.fixture.store,
                transition_lock=self.lock, _cohort_filesystem=self.filesystem,
                _cohort_settings_candidate_enabled=True)

    async def test_settings_opt_in_does_not_grant_store_transition_permission(self):
        self.service = self.make_service(store=self.fixture.store)
        with self.assertRaisesRegex(ValidationError, 'store permissions'):
            await self.save()
        self.assertEqual(self.persisted, [])
        self.assertIsNone(await self.store.get_policy_transition())

    async def test_default_settings_path_does_not_enable_private_invalidation(self):
        self.service = self.make_service(store=self.fixture.store, enabled=False)
        await self.save()
        self.assertEqual(self.current, self.proposed)
        self.assertEqual((await self.store.get_policy_transition())['state'], 'completed')
        self.assertEqual(self.fixture.events(), [])

    async def test_unimplemented_private_recovery_fails_closed_without_mutation(self):
        await self.transition.prepare()
        before = self.transition.boundary_rows()
        with self.assertRaisesRegex(TargetStartupInvariantError, 'startup recovery is not implemented'):
            await self.service.recover_pending_transition()
        self.assertEqual(self.transition.boundary_rows(), before)
        self.assertEqual(self.persisted, [])
        self.assertEqual(self.fixture.events(), [])

    async def test_native_publisher_root_then_policy_scope_cannot_deadlock_save(self):
        publisher = self.fixture.source.staged.service
        publisher._policy_transition_lock = self.lock
        held, started, go, entered = [asyncio.Event() for _ in range(4)]
        async def publication():
            async with self.filesystem.write('root'):
                held.set()
                await go.wait()
                async with publisher._policy_scope_candidate():
                    entered.set()
        async def saving():
            started.set()
            return await self.save()
        publish = asyncio.create_task(publication())
        save = None
        try:
            await asyncio.wait_for(held.wait(), 2)
            save = asyncio.create_task(saving())
            await asyncio.wait_for(started.wait(), 2)
            self.assertFalse(self.lock.locked())
            go.set()
            await asyncio.wait_for(asyncio.gather(publish, save), 5)
            self.assertTrue(entered.is_set())
        finally:
            go.set()
            for task in (publish, save):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(publish, *([save] if save else []), return_exceptions=True)

    async def test_repeated_cancel_during_prepare_aborts_before_configuration_save(self):
        original = self.reconciliation.prepare_boundary
        entered, release = asyncio.Event(), asyncio.Event()
        async def gated(**kwargs):
            await original(**kwargs)
            entered.set()
            await asyncio.wait_for(release.wait(), 5)
        before = self.fixture.rows()
        with patch.object(self.reconciliation, 'prepare_boundary', side_effect=gated):
            owner = asyncio.create_task(self.save())
            try:
                await asyncio.wait_for(entered.wait(), 5)
                for _ in range(2):
                    owner.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(owner.done())
                    self.assertTrue(self.lock.locked())
                    self.assertTrue(self.filesystem._state('root').writer_active)
                    self.assertEqual(self.persisted, [])
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(owner, 5)
            finally:
                release.set()
                await asyncio.gather(owner, return_exceptions=True)
        await self.assert_aborted_without_effects(before)

    async def test_repeated_cancel_during_commit_retains_leases_through_real_sql(self):
        module = importlib.import_module('infrastructure.persistence.import_cohort_binding')
        original = module.invalidate_cohort_policy_candidate
        entered, release = threading.Event(), threading.Event()
        def gated(connection, *args, **kwargs):
            result = original(connection, *args, **kwargs)
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Settings SQL gate timed out')
            return result
        with patch.object(module, 'invalidate_cohort_policy_candidate', side_effect=gated):
            owner = asyncio.create_task(self.save())
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                for _ in range(2):
                    owner.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(owner.done())
                    self.assertTrue(self.lock.locked())
                    self.assertTrue(self.filesystem._state('root').writer_active)
                    self.assertEqual(self.fixture.events(), [])
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(owner, 5)
            finally:
                release.set()
                await asyncio.gather(owner, return_exceptions=True)
        self.assertEqual(self.current, self.proposed)
        self.assertEqual((await self.store.get_policy_transition())['state'], 'completed')
        self.assertEqual(len(self.fixture.events()), 1)
        transition = await self.store.get_policy_transition()
        self.assertEqual(self.dirty, [LibraryPolicyService._scope_ids(transition['scopes'])])
        self.coordinator.request_run.assert_not_awaited()

    async def test_actual_excluded_root_save_uses_native_suppression_not_reconciliation(self):
        root = msgspec.structs.replace(self.current.library_roots[0], policy='excluded')
        proposed = msgspec.structs.replace(self.current, library_roots=[root])
        with self.store._connect() as db:
            before = [dict(row) for row in db.execute('SELECT * FROM local_tracks ORDER BY id')]
        response = await self.save(proposed)
        self.assertEqual(response.policy_revision, LibraryPolicyResolver(proposed).policy_revision)
        self.assertEqual(len(self.fixture.events()), 1)
        with self.store._connect() as db:
            after = [dict(row) for row in db.execute('SELECT * FROM local_tracks ORDER BY id')]
        for old, new in zip(before, after, strict=True):
            self.assertEqual(new, dict(old, desired_policy_revision=response.policy_revision,
                row_revision=old['row_revision'] + 1))
        self.coordinator.request_run.assert_not_awaited()

    async def test_actual_replacement_root_save_leases_previous_and_proposed_ids(self):
        old = self.current.library_roots[0]
        root = msgspec.structs.replace(old, id='replacement', path=str(self.fixture.source.staged.fixture.root.parent / 'replacement'))
        proposed = msgspec.structs.replace(self.current, library_roots=[root])
        original = self.store.commit_policy_transition
        async def checked(**kwargs):
            self.assertTrue(self.lock.locked())
            self.assertTrue(self.filesystem._cohort_writer_is_held(old.id))
            self.assertTrue(self.filesystem._cohort_writer_is_held(root.id))
            return await original(**kwargs)
        with patch.object(self.store, 'commit_policy_transition', side_effect=checked):
            await self.save(proposed)
        self.assertEqual(self.current, proposed)
        self.assertEqual(len(self.fixture.events()), 1)
        self.coordinator.request_run.assert_not_awaited()

    async def test_late_configuration_drift_during_root_acquisition_refuses_save(self):
        with patch.object(self.settings, 'current_settings', side_effect=[self.current, self.proposed]):
            with self.assertRaisesRegex(StaleRevisionError, 'acquiring cohort root leases'):
                await self.save()
        self.assertEqual(self.persisted, [])
        self.assertIsNone(await self.store.get_policy_transition())
        self.assertEqual(self.fixture.events(), [])

    async def exercise_default_repeated_cancellation(self, phase):
        # Real default store/service, no cohort settings/transition permission.
        from infrastructure.persistence.native_library_store import NativeLibraryStore
        store = NativeLibraryStore(self.fixture.path, threading.Lock())
        self.service = self.make_service(store=store, enabled=False)
        self.assertFalse(self.service._cohort_settings_candidate_enabled)
        self.assertFalse(store._cohort_schema_candidate_enabled)
        self.assertFalse(store._cohort_policy_transition_enabled)
        self.assertFalse(store._cohort_policy_invalidation_enabled)
        before = self.fixture.rows()
        entered, release = asyncio.Event(), asyncio.Event()
        prepare_entered, prepare_release = asyncio.Event(), asyncio.Event()
        reconciliation = self.service._reconciliation
        name = {'prepare': 'prepare_boundary', 'commit': 'commit_boundary', 'abort': 'abort_boundary'}[phase]
        original = getattr(reconciliation, name)
        async def gated(**kwargs):
            result = await original(**kwargs)
            entered.set()
            await asyncio.wait_for(release.wait(), 5)
            return result
        prepare_original = reconciliation.prepare_boundary
        async def prepare_gated(**kwargs):
            result = await prepare_original(**kwargs)
            prepare_entered.set()
            await asyncio.wait_for(prepare_release.wait(), 5)
            return result
        with patch.object(reconciliation, name, side_effect=gated):
            prepare_patch = patch.object(reconciliation, 'prepare_boundary', side_effect=prepare_gated)
            if phase == 'abort': prepare_patch.start()
            owner = asyncio.create_task(self.save())
            try:
                if phase == 'abort':
                    await asyncio.wait_for(prepare_entered.wait(), 5)
                    owner.cancel()
                    prepare_release.set()
                await asyncio.wait_for(entered.wait(), 5)
                for _ in range(3):
                    owner.cancel()
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    self.assertFalse(owner.done(), 'Default boundary released before critical task settled')
                    self.assertTrue(self.lock.locked())
                    self.assertFalse(self.filesystem._state('root').writer_active)
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(owner, 5)
            finally:
                release.set(); prepare_release.set()
                await asyncio.gather(owner, return_exceptions=True)
                if phase == 'abort': prepare_patch.stop()
        transition = await store.get_policy_transition()
        self.assertEqual(transition['state'], 'completed' if phase == 'commit' else 'aborted')
        self.assertEqual(self.current, self.proposed if phase == 'commit' else self.fixture.old.settings)
        self.assertEqual(self.dirty, [LibraryPolicyService._scope_ids(transition['scopes'])] if phase == 'commit' else [])
        self.assertEqual(self.fixture.events(), [], 'Default cancellation must not enable cohort invalidation')
        self.assertEqual(self.fixture.rows(), before)
        self.assertFalse(self.lock.locked())
        self.coordinator.request_run.assert_not_awaited()

    async def test_default_repeated_cancellation_drains_prepare_then_aborts(self):
        await self.exercise_default_repeated_cancellation('prepare')

    async def test_default_repeated_cancellation_drains_commit_and_marks_hint(self):
        await self.exercise_default_repeated_cancellation('commit')

    async def test_default_repeated_cancellation_drains_abort_before_unlock(self):
        await self.exercise_default_repeated_cancellation('abort')
