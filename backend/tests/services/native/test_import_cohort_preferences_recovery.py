"""Real disposable preferences files and simulated durable phase interruptions."""
import asyncio
from contextlib import ExitStack
import importlib
import sqlite3
import threading
import unittest
from unittest.mock import Mock, patch

import msgspec
from core.config import Settings
from core.exceptions import ConfigurationError, StaleRevisionError, TargetStartupInvariantError
from infrastructure.persistence.native_library_store import NativeLibraryStore
from services.preferences_service import PreferencesService
from services.native.library_policy_resolver import LibraryPolicyResolver
from services.native.library_policy_service import LibraryPolicyService
from services.native.library_policy_reconciliation_service import LibraryPolicyReconciliationService
from services.native.target_library_policy_service import TargetLibraryPolicyService


class CohortPreferencesRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module = importlib.import_module('tests.services.native.test_import_cohort_settings_save')
        self.source = module.CohortSettingsSaveTests()
        await self.source.asyncSetUp()
        self.addCleanup(self.source.doCleanups)
        self.fixture = self.source.fixture
        self.config = self.fixture.path.parent / 'disposable-preferences.json'
        self.preferences = PreferencesService(Settings(config_file_path=self.config,
            root_app_dir=self.config.parent))
        self.preferences.save_typed_library_settings(self.fixture.old.settings)
        self.proposed = self.fixture.current.settings
        self.reopen()

    def reopen(self, *, recovery=True):
        self.preferences = PreferencesService(Settings(config_file_path=self.config,
            root_app_dir=self.config.parent))
        self.store = NativeLibraryStore(self.fixture.path, threading.Lock(),
            _cohort_schema_candidate_enabled=True, _cohort_policy_invalidation_enabled=True,
            _cohort_policy_transition_enabled=True)
        self.lock = asyncio.Lock()
        resolver = lambda: LibraryPolicyResolver(self.preferences.get_typed_library_settings_raw())
        self.settings = LibraryPolicyService(self.preferences, None, resolver, lambda: None)
        self.reconciliation = LibraryPolicyReconciliationService(self.store, resolver, self.source.coordinator)
        self.service = TargetLibraryPolicyService(self.settings, self.reconciliation, self.store,
            transition_lock=self.lock, _cohort_filesystem=self.fixture.filesystem,
            _cohort_settings_candidate_enabled=True, _cohort_recovery_candidate_enabled=recovery)

    def revision(self):
        return LibraryPolicyResolver(self.preferences.get_typed_library_settings_raw()).policy_revision

    async def prepare(self, *, persist=False):
        await self.source.transition.prepare()
        if persist:
            self.preferences.save_typed_library_settings_if_current(self.proposed,
                expected_policy_revision=self.fixture.old.policy_revision)

    def normal_factory(self):
        """Use actual cached providers, replacing only disposable leaf dependencies."""
        from core.dependencies import cache_providers, service_providers as providers
        factories = (providers.get_target_library_policy_service,
            providers.get_target_library_policy_reconciliation_service)
        for factory in factories:
            factory.cache_clear()
            self.addCleanup(factory.cache_clear)
        stack = self.enterContext(ExitStack())
        resolver = Mock(side_effect=lambda: LibraryPolicyResolver(
            self.preferences.get_typed_library_settings_raw()))
        stack.enter_context(patch.object(cache_providers, 'get_native_library_store', return_value=self.store))
        for name, value in {
            'get_preferences_service': Mock(return_value=self.preferences),
            'get_library_policy_resolver': resolver,
            'get_library_scan_wakeup': Mock(return_value=None),
            'get_target_library_scan_coordinator': Mock(return_value=self.source.coordinator),
            'get_library_policy_transition_lock': Mock(return_value=self.lock),
        }.items():
            stack.enter_context(patch.object(providers, name, value))
        return providers, providers.get_target_library_policy_service()

    async def test_actual_normal_factory_keeps_private_recovery_and_settings_off(self):
        providers, service = self.normal_factory()
        self.assertFalse(service._cohort_settings_candidate_enabled)
        self.assertFalse(service._cohort_recovery_candidate_enabled)
        self.assertIsNone(service._cohort_filesystem)
        self.assertIs(service._store, self.store)
        self.assertIs(service._reconciliation._store, self.store)
        self.assertIs(service._save_lock, self.lock)
        self.assertIs(service._settings._preferences, self.preferences)
        self.assertIs(service._reconciliation,
            providers.get_target_library_policy_reconciliation_service())

    async def test_actual_normal_factory_singleton_reuses_one_policy_boundary(self):
        providers, service = self.normal_factory()
        self.assertIs(providers.get_target_library_policy_service(), service)
        self.assertEqual(providers.get_target_library_policy_service.cache_info().misses, 1)
        self.assertEqual(providers.get_target_library_policy_service.cache_info().hits, 1)
        providers.get_target_library_policy_service.cache_clear()
        replacement = providers.get_target_library_policy_service()
        self.assertIsNot(replacement, service)
        self.assertIs(replacement._save_lock, self.lock)
        self.assertIs(replacement._store, self.store)
        self.assertIs(replacement._reconciliation, service._reconciliation)

    async def test_actual_normal_factory_empty_recovery_does_not_write(self):
        _, service = self.normal_factory()
        before = self.fixture.rows()
        config = self.config.read_bytes()
        self.assertFalse(await service.recover_pending_transition())
        self.assertIsNone(await self.store.get_policy_transition())
        self.assertEqual(self.fixture.rows(), before)
        self.assertEqual(self.fixture.events(), [])
        self.assertEqual(self.config.read_bytes(), config)
        self.source.coordinator.request_run.assert_not_awaited()

    async def test_real_settings_save_survives_new_preferences_and_store_instances(self):
        await self.service.save_settings(self.proposed, expected_policy_revision=self.fixture.old.policy_revision)
        events = self.fixture.events()
        self.reopen()
        self.assertEqual(self.revision(), self.fixture.current.policy_revision)
        self.assertEqual(len(events), 1)
        self.assertEqual(self.fixture.events(), events)
        self.assertFalse(await self.service.recover_pending_transition())

    async def exercise_production_lifespan_catalog_guard(self, *, expected_state=None):
        """Actual lifespan + recovery + validator; unrelated startup leaves isolated.

        This private catalog fixture deliberately has NO completed migration marker.
        It must reach the real catalog guard only after recovery, never admission.
        This is not successful full startup or normal factory activation evidence.
        """
        from unittest.mock import AsyncMock
        import target_application as target
        from maintenance import automatic_upgrade
        from types import SimpleNamespace
        import core.config as config_module

        original = self.store.get_target_startup_state
        async def observe_catalog_state():
            self.assertEqual((await self.store.get_policy_transition())['state'], expected_state)
            state = await original()
            self.assertIsNone(state['marker'], 'Fixture must exercise actual missing-marker guard')
            return state

        catalog = AsyncMock(side_effect=observe_catalog_state)
        admission = AsyncMock(side_effect=AssertionError('Admission must not run'))
        with ExitStack() as stack:
            for owner, name, value in (
                (target, 'get_settings', lambda: Settings(config_file_path=self.config,
                    root_app_dir=self.config.parent)),
                (target, '_warn_if_launcher_bypassed', AsyncMock(return_value=False)),
                (config_module, 'migrate_legacy_config', lambda: None),
                (target, 'init_app_state', AsyncMock()),
                (target, 'get_target_library_policy_service', lambda: self.service),
                (target, 'get_native_library_store', lambda: self.store),
                (target, 'get_preferences_service', lambda: self.preferences),
                (automatic_upgrade, 'target_startup_admission_pending', lambda: False),
                (automatic_upgrade, 'await_target_startup_admission', admission),
                (self.store, 'get_target_startup_state', catalog),
            ):
                stack.enter_context(patch.object(owner, name, value))
            message = ('completed legacy-catalog migration marker is missing'
                if expected_state is not None else 'startup recovery is not implemented')
            with self.assertRaisesRegex(TargetStartupInvariantError, message):
                async with target.production_target_lifespan(SimpleNamespace()):
                    self.fail('Startup must not yield or start operational workers')
        admission.assert_not_awaited()
        if expected_state is None:
            catalog.assert_not_awaited()
        else:
            catalog.assert_awaited_once()
        self.source.coordinator.request_run.assert_not_awaited()

    async def test_production_lifespan_aborts_prepared_old_config_before_real_catalog_guard(self):
        await self.prepare()
        self.reopen()
        before = self.fixture.rows()
        config = self.config.read_bytes()
        await self.exercise_production_lifespan_catalog_guard(expected_state='aborted')
        self.assertEqual(self.fixture.rows(), before)
        self.assertEqual(self.config.read_bytes(), config)
        self.assertEqual(self.fixture.events(), [])
        self.assertFalse(await self.service.recover_pending_transition())

    async def exercise_normal_recovery_cancellation(self, *, persist):
        await self.prepare(persist=persist)
        self.store = NativeLibraryStore(self.fixture.path, threading.Lock())
        _, self.service = self.normal_factory()
        for name in ('_cohort_schema_candidate_enabled', '_cohort_policy_invalidation_enabled',
                     '_cohort_policy_transition_enabled'):
            self.assertFalse(getattr(self.store, name))
        self.assertFalse(self.service._cohort_settings_candidate_enabled)
        self.assertFalse(self.service._cohort_recovery_candidate_enabled)
        self.assertIsNone(self.service._cohort_filesystem)
        before, config = self.fixture.rows(), self.config.read_bytes()
        expected = dict(before)
        if persist:
            with self.store._connect() as db:
                columns = [item[0] for item in db.execute('SELECT * FROM local_tracks LIMIT 0').description]
            expected_tracks = []
            for row in before['local_tracks']:
                track = dict(zip(columns, row, strict=True))
                self.assertEqual(track['root_id'], 'root')
                self.assertNotEqual(track['desired_policy_revision'], self.fixture.current.policy_revision)
                track.update(desired_policy_revision=self.fixture.current.policy_revision,
                             row_revision=track['row_revision'] + 1)
                expected_tracks.append(tuple(track[column] for column in columns))
            expected['local_tracks'] = expected_tracks
            self.assertEqual(len(before['library_catalog_revision']), 1)
            singleton, revision = before['library_catalog_revision'][0]
            expected['library_catalog_revision'] = [(singleton, revision + 1)]
        original_write = self.store._write
        entered, release = threading.Event(), threading.Event()
        async def gated_write(operation):
            def gated(connection):
                result = operation(connection)
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('Normal recovery SQL gate timed out')
                return result
            return await original_write(gated)
        with patch.object(self.store, '_write', side_effect=gated_write):
            owner = asyncio.create_task(self.service.recover_pending_transition())
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                for _ in range(3):
                    owner.cancel()
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    self.assertFalse(owner.done(), 'Normal recovery abandoned in-flight SQL')
                    self.assertTrue(self.lock.locked())
                    self.assertEqual((await self.store.get_policy_transition())['state'], 'prepared')
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(owner, 5)
            finally:
                release.set()
                await asyncio.gather(owner, return_exceptions=True)
                # Drain the same native writer lock even on a before-test failure.
                await original_write(lambda connection: None)
        self.assertEqual((await self.store.get_policy_transition())['state'], 'completed' if persist else 'aborted')
        self.assertEqual(self.fixture.rows(), expected)
        self.assertEqual(self.config.read_bytes(), config)
        self.assertEqual(self.fixture.events(), [])
        self.source.coordinator.request_run.assert_not_awaited()
        self.assertFalse(self.lock.locked())
        self.assertFalse(await self.service.recover_pending_transition(), 'Recovery must be idempotent')

    async def test_normal_recovery_repeated_cancellation_drains_commit_sql(self):
        await self.exercise_normal_recovery_cancellation(persist=True)

    async def test_normal_recovery_repeated_cancellation_drains_abort_sql(self):
        await self.exercise_normal_recovery_cancellation(persist=False)

    async def exercise_production_catalog_acceptance(self, *, persist,
            admission_phase=False, invalid_reference=False, mismatched_root=False):
        """Real steady-state validator, native-written synthetic migration marker.

        Stop at admission entry; no admission success or operational-startup claim.
        The fixture is not a replay of the complete legacy migration process.
        """
        from unittest.mock import AsyncMock
        from types import SimpleNamespace
        import target_application as target
        import core.config as config_module
        from maintenance import automatic_upgrade

        await self.store.save_migration_dry_run('fixture-startup',
            source_revision='fixture-source',root_revision='fixture-roots',
            report_json='{}',created_at=1)
        await self.store.finish_migration('fixture-startup',
            source_revision='fixture-source',report_json='{}',completed_at=2)
        if admission_phase:
            from models.library_work import MigrationProvenance
            with self.store._connect() as db:
                track_id = db.execute('SELECT id FROM local_tracks ORDER BY id LIMIT 1').fetchone()[0]
            for source_kind,source_key,target_kind,target_id in (
                ('root','fixture-root','library_root','other-root' if mismatched_root else 'root'),
                ('library_file','fixture-track','local_track','missing-fixture-track' if invalid_reference else track_id),
            ):
                await self.store.record_migration_provenance(MigrationProvenance(
                    source_kind=source_kind,source_key=source_key,target_kind=target_kind,
                    target_id=target_id,source_revision='fixture-source',imported_at=2),
                    migration_run_id='fixture-startup')
        await self.prepare(persist=persist)
        self.reopen()
        expected_state = 'completed' if persist else 'aborted'
        order = []
        original_state = self.store.get_target_startup_state
        original_integrity = self.store.validate_catalog_integrity
        original_references = self.store.validate_migration_references
        original_roots = self.store.get_migrated_root_ids

        async def read_state():
            self.assertEqual((await self.store.get_policy_transition())['state'], expected_state)
            order.append('catalog-state-after-recovery')
            return await original_state()

        async def validate_integrity():
            result = await original_integrity()
            self.assertTrue(all(count == 0 for count in result.values()), result)
            order.append('real-catalog-integrity')
            return result

        async def validate_references():
            result = await original_references()
            self.assertEqual(result, 1 if invalid_reference else 0)
            order.append('real-saved-references')
            return result

        async def migrated_roots():
            result = await original_roots()
            self.assertEqual(result, {'other-root'} if mismatched_root else {'root'})
            order.append('real-root-provenance')
            return result

        class AdmissionBoundaryReached(Exception):
            pass

        async def stop_at_admission(_settings):
            order.append('admission-entry')
            raise AdmissionBoundaryReached()

        admission = AsyncMock(side_effect=stop_at_admission)
        with ExitStack() as stack:
            for owner,name,value in (
                (target,'get_settings',lambda: Settings(config_file_path=self.config,
                    root_app_dir=self.config.parent)),
                (target,'_warn_if_launcher_bypassed',AsyncMock(return_value=False)),
                (config_module,'migrate_legacy_config',lambda: None),
                (target,'init_app_state',AsyncMock()),
                (target,'get_target_library_policy_service',lambda: self.service),
                (target,'get_native_library_store',lambda: self.store),
                (target,'get_preferences_service',lambda: self.preferences),
                (automatic_upgrade,'target_startup_admission_pending',lambda: admission_phase),
                (automatic_upgrade,'await_target_startup_admission',admission),
                (self.store,'get_target_startup_state',AsyncMock(side_effect=read_state)),
                (self.store,'validate_catalog_integrity',AsyncMock(side_effect=validate_integrity)),
                (self.store,'validate_migration_references',AsyncMock(side_effect=validate_references)),
                (self.store,'get_migrated_root_ids',AsyncMock(side_effect=migrated_roots)),
            ):
                stack.enter_context(patch.object(owner,name,value))
            error = TargetStartupInvariantError if invalid_reference or mismatched_root else AdmissionBoundaryReached
            message = ('failed startup integrity validation' if invalid_reference else
                'configured library roots do not match' if mismatched_root else '')
            with self.assertRaisesRegex(error,message):
                async with target.production_target_lifespan(SimpleNamespace()):
                    self.fail('No operational runtime permitted')
        expected_order = ['catalog-state-after-recovery','real-catalog-integrity']
        if admission_phase:
            expected_order.append('real-saved-references')
            if not invalid_reference:
                expected_order.append('real-root-provenance')
        if invalid_reference or mismatched_root:
            admission.assert_not_awaited()
        else:
            expected_order.append('admission-entry')
            admission.assert_awaited_once()
        self.assertEqual(order,expected_order)
        self.source.coordinator.request_run.assert_not_awaited()
        self.assertEqual(len(self.fixture.events()), 1 if persist else 0)
        self.assertFalse(await self.service.recover_pending_transition())

    async def test_production_lifespan_valid_marker_old_config_passes_real_catalog_before_admission(self):
        await self.exercise_production_catalog_acceptance(persist=False)

    async def test_production_lifespan_valid_marker_persisted_config_passes_real_catalog_before_admission(self):
        await self.exercise_production_catalog_acceptance(persist=True)

    async def test_production_lifespan_admission_phase_old_config_checks_real_references_and_roots(self):
        await self.exercise_production_catalog_acceptance(persist=False, admission_phase=True)

    async def test_production_lifespan_admission_phase_persisted_config_checks_real_references_and_roots(self):
        await self.exercise_production_catalog_acceptance(persist=True, admission_phase=True)

    async def test_production_lifespan_admission_phase_dangling_reference_refuses_entry(self):
        await self.exercise_production_catalog_acceptance(persist=True, admission_phase=True,
            invalid_reference=True)

    async def test_production_lifespan_admission_phase_mismatched_root_refuses_entry(self):
        await self.exercise_production_catalog_acceptance(persist=True, admission_phase=True,
            mismatched_root=True)

    async def test_production_lifespan_completes_persisted_config_before_real_catalog_guard(self):
        await self.prepare(persist=True)
        self.reopen()
        before = self.fixture.rows()
        config = self.config.read_bytes()
        await self.exercise_production_lifespan_catalog_guard(expected_state='completed')
        # Prepared fixture scope covers the root: native commit changes ONLY the
        # desired-policy/row revisions, plus one catalog revision if tracks changed.
        expected = dict(before)
        with self.store._connect() as db:
            columns = [row[1] for row in db.execute('PRAGMA table_info(local_tracks)')]
            revision_columns = [row[1] for row in db.execute('PRAGMA table_info(library_catalog_revision)')]
        desired, row_revision, root = [columns.index(name) for name in
            ('desired_policy_revision', 'row_revision', 'root_id')]
        changed = 0
        tracks = []
        for original in before['local_tracks']:
            row = list(original)
            if row[root] == 'root' and row[desired] != self.fixture.current.policy_revision:
                row[desired] = self.fixture.current.policy_revision
                row[row_revision] += 1
                changed += 1
            tracks.append(tuple(row))
        self.assertGreater(changed, 0)
        expected['local_tracks'] = tracks
        value = revision_columns.index('value')
        expected['library_catalog_revision'] = [tuple(
            item + 1 if index == value else item for index,item in enumerate(row))
            for row in before['library_catalog_revision']]
        self.assertEqual(self.fixture.rows(), expected)
        # Recovery also persists the exact native dirty scope, without changing
        # any other preference or starting the scan coordinator.
        import json
        expected_config = json.loads(config)
        expected_config['library_scan_dirty_scopes'] = {'scope_ids': ['root']}
        self.assertEqual(json.loads(self.config.read_bytes()), expected_config)
        self.assertEqual(len(self.fixture.events()), 1)
        self.assertFalse(await self.service.recover_pending_transition())

    async def test_production_lifespan_disabled_private_recovery_stops_before_catalog_and_admission(self):
        await self.prepare(persist=True)
        self.reopen(recovery=False)
        before = self.fixture.rows()
        config = self.config.read_bytes()
        await self.exercise_production_lifespan_catalog_guard()
        self.assertEqual((await self.store.get_policy_transition())['state'], 'prepared')
        self.assertEqual(self.fixture.rows(), before)
        self.assertEqual(self.config.read_bytes(), config)
        self.assertEqual(self.fixture.events(), [])

    async def test_real_configuration_revision_cas_refuses_stale_save_without_file_change(self):
        before = self.config.read_bytes()
        with self.assertRaises(StaleRevisionError):
            await self.service.save_settings(self.proposed, expected_policy_revision='stale')
        self.assertEqual(self.config.read_bytes(), before)
        self.assertIsNone(await self.store.get_policy_transition())

    async def test_real_preferences_root_move_guard_is_not_bypassed(self):
        old = self.fixture.old.settings.library_roots[0]
        moved = msgspec.structs.replace(old, path=str(self.config.parent / 'moved'))
        proposed = msgspec.structs.replace(self.fixture.old.settings, library_roots=[moved])
        before = self.config.read_bytes()
        with self.assertRaisesRegex(ConfigurationError, 'cannot be moved'):
            await self.service.save_settings(proposed, expected_policy_revision=self.fixture.old.policy_revision)
        self.assertEqual(self.config.read_bytes(), before)
        self.assertEqual((await self.store.get_policy_transition())['state'], 'aborted')
        self.assertEqual(self.fixture.events(), [])

    async def test_real_config_file_is_restored_after_late_native_save_failure(self):
        before = self.fixture.rows()
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER fail_real_save BEFORE UPDATE ON library_policy_transitions WHEN NEW.state='completed' BEGIN SELECT RAISE(ABORT,'Injected real save failure'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'Injected real save failure'):
            await self.service.save_settings(self.proposed, expected_policy_revision=self.fixture.old.policy_revision)
        self.reopen()
        self.assertEqual(self.revision(), self.fixture.old.policy_revision)
        self.assertEqual((await self.store.get_policy_transition())['state'], 'aborted')
        self.assertEqual(self.fixture.events(), [])
        self.assertEqual(self.fixture.rows(), before)

    async def test_prepared_only_interruption_aborts_after_reopen(self):
        await self.prepare()
        self.reopen()
        self.assertTrue(await self.service.recover_pending_transition())
        self.assertEqual(self.revision(), self.fixture.old.policy_revision)
        self.assertEqual((await self.store.get_policy_transition())['state'], 'aborted')
        self.assertEqual(self.fixture.events(), [])
        self.assertFalse(await self.service.recover_pending_transition())

    async def test_config_persisted_interruption_completes_once_after_reopen(self):
        await self.prepare(persist=True)
        self.reopen()
        self.assertTrue(await self.service.recover_pending_transition())
        events = self.fixture.events()
        self.assertEqual(len(events), 1)
        self.assertEqual(self.revision(), self.fixture.current.policy_revision)
        self.assertEqual((await self.store.get_policy_transition())['state'], 'completed')
        self.reopen()
        self.assertFalse(await self.service.recover_pending_transition())
        self.assertEqual(self.fixture.events(), events)
        self.source.coordinator.request_run.assert_not_awaited()

    async def test_settings_opt_in_alone_does_not_enable_startup_recovery(self):
        await self.prepare(persist=True)
        self.reopen(recovery=False)
        with self.assertRaisesRegex(TargetStartupInvariantError, 'startup recovery is not implemented'):
            await self.service.recover_pending_transition()
        self.assertEqual((await self.store.get_policy_transition())['state'], 'prepared')
        self.assertEqual(self.fixture.events(), [])

    async def test_unknown_saved_revision_preserves_prepared_journal_and_file(self):
        await self.prepare()
        root = msgspec.structs.replace(self.fixture.old.settings.library_roots[0], policy='excluded')
        unknown = msgspec.structs.replace(self.fixture.old.settings, library_roots=[root])
        self.preferences.save_typed_library_settings(unknown)
        self.reopen()
        before = self.config.read_bytes()
        with self.assertRaisesRegex(TargetStartupInvariantError, 'neither prepared revision'):
            await self.service.recover_pending_transition()
        self.assertEqual(self.config.read_bytes(), before)
        self.assertEqual((await self.store.get_policy_transition())['state'], 'prepared')
        self.assertEqual(self.fixture.events(), [])

    async def test_changed_prepared_settings_hash_is_not_repaired(self):
        await self.prepare(persist=True)
        with self.store._connect() as db:
            db.execute("UPDATE library_policy_transitions SET previous_policy_revision='changed'")
        with self.assertRaisesRegex(TargetStartupInvariantError, 'prepared revisions disagree'):
            await self.service.recover_pending_transition()
        self.assertEqual((await self.store.get_policy_transition())['state'], 'prepared')
        self.assertEqual(self.fixture.events(), [])

    async def test_late_journal_change_under_root_leases_refuses_recovery(self):
        await self.prepare(persist=True)
        original = self.store.get_policy_transition
        calls = 0
        async def changed():
            nonlocal calls
            calls += 1
            if calls == 2:
                with self.store._connect() as db:
                    db.execute('UPDATE library_policy_transitions SET prepared_at=999')
            return await original()
        with patch.object(self.store, 'get_policy_transition', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'changed under root leases'):
                await self.service.recover_pending_transition()
        self.assertEqual(self.fixture.events(), [])

    async def test_failed_recovery_keeps_persisted_config_and_prepared_evidence(self):
        await self.prepare(persist=True)
        before = self.config.read_bytes()
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER fail_real_recovery BEFORE UPDATE ON library_policy_transitions WHEN NEW.state='completed' BEGIN SELECT RAISE(ABORT,'Injected recovery failure'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'Injected recovery failure'):
            await self.service.recover_pending_transition()
        self.assertEqual(self.config.read_bytes(), before)
        self.assertEqual((await self.store.get_policy_transition())['state'], 'prepared')
        self.assertEqual(self.fixture.events(), [])

    async def test_recovery_obeys_actual_publisher_root_then_policy_order(self):
        await self.prepare(persist=True)
        publisher = self.fixture.source.staged.service
        publisher._policy_transition_lock = self.lock
        held, seen, go = [asyncio.Event() for _ in range(3)]
        original = self.store.get_policy_transition
        async def discovery():
            result = await original()
            seen.set()
            return result
        async def publication():
            async with self.fixture.filesystem.write('root'):
                held.set()
                await go.wait()
                async with publisher._policy_scope_candidate():
                    pass
        owner = asyncio.create_task(publication())
        recovery = None
        try:
            await asyncio.wait_for(held.wait(), 2)
            with patch.object(self.store, 'get_policy_transition', side_effect=discovery):
                recovery = asyncio.create_task(self.service.recover_pending_transition())
                await asyncio.wait_for(seen.wait(), 2)
                self.assertFalse(self.lock.locked())
                go.set()
                await asyncio.wait_for(asyncio.gather(owner, recovery), 5)
        finally:
            go.set()
            for task in (owner, recovery):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(owner, *([recovery] if recovery else []), return_exceptions=True)
        self.assertEqual(len(self.fixture.events()), 1)

    async def prepare_multiroot_recovery(self):
        """Establish coherent previous policy through real save, then interrupt next."""
        from models.library_work import ScanScope
        common = self.fixture.old.settings.library_roots[0]
        previous_path = self.config.parent / 'previous-only-root'
        proposed_path = self.config.parent / 'proposed-only-root'
        previous_path.mkdir()
        proposed_path.mkdir()
        previous_only = msgspec.structs.replace(common,id='a-previous',label='Previous only',path=str(previous_path))
        proposed_only = msgspec.structs.replace(common,id='z-proposed',label='Proposed only',path=str(proposed_path))
        previous_settings = msgspec.structs.replace(self.fixture.old.settings,
            library_roots=[common,previous_only])
        await self.service.save_settings(previous_settings,
            expected_policy_revision=self.fixture.old.policy_revision)
        previous = LibraryPolicyResolver(previous_settings)
        proposed_settings = msgspec.structs.replace(previous_settings,
            library_roots=[common,proposed_only])
        proposed = LibraryPolicyResolver(proposed_settings)
        await self.store.prepare_policy_transition(
            previous_policy_revision=previous.policy_revision,
            proposed_policy_revision=proposed.policy_revision,
            previous_settings_json=msgspec.json.encode(previous_settings).decode(),
            proposed_settings_json=msgspec.json.encode(proposed_settings).decode(),
            scopes=[ScanScope(root_id=root.id,root_path=root.path,scope_id=root.id,
                effective_policy='excluded' if root.id == previous_only.id else root.policy,
                policy_revision=proposed.policy_revision)
                for root in (common,previous_only,proposed_only)],prepared_at=20)
        self.preferences.save_typed_library_settings_if_current(proposed_settings,
            expected_policy_revision=previous.policy_revision)
        self.reopen()
        return proposed.policy_revision

    async def exercise_multiroot_recovery_wait(self, held_root, *, cancel=False):
        proposed_revision = await self.prepare_multiroot_recovery()
        filesystem = self.fixture.filesystem
        roots = ('a-previous','root','z-proposed')
        held,go = asyncio.Event(),asyncio.Event()
        before,config,events = self.fixture.rows(),self.config.read_bytes(),self.fixture.events()

        async def hold_root():
            async with filesystem.write(held_root):
                held.set()
                await go.wait()

        async def wait_registered():
            while filesystem._state(held_root).waiting_writers == 0:
                await asyncio.sleep(0.01)

        owner = asyncio.create_task(hold_root())
        recovery = None
        try:
            await asyncio.wait_for(held.wait(),5)
            recovery = asyncio.create_task(self.service.recover_pending_transition())
            await asyncio.wait_for(wait_registered(),5)
            self.assertFalse(recovery.done())
            self.assertFalse(self.lock.locked(), 'Root wait must precede policy lock')
            for root in roots:
                state = filesystem._state(root)
                self.assertTrue(state.writer_active or state.waiting_writers > 0)
            self.assertEqual((await self.store.get_policy_transition())['state'],'prepared')
            self.assertEqual(self.fixture.rows(),before)
            self.assertEqual(self.config.read_bytes(),config)
            self.assertEqual(self.fixture.events(),events)
            if cancel:
                recovery.cancel()
                recovery.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(recovery,5)
                self.assertFalse(self.lock.locked())
                for root in roots:
                    state = filesystem._state(root)
                    self.assertEqual(state.waiting_writers,0)
                    self.assertEqual(state.writer_active, root == held_root)
                self.assertEqual((await self.store.get_policy_transition())['state'],'prepared')
                self.assertEqual(self.fixture.rows(),before)
                self.assertEqual(self.config.read_bytes(),config)
                self.assertEqual(self.fixture.events(),events)
            go.set()
            await asyncio.wait_for(owner,5)
            if cancel:
                self.assertTrue(await self.service.recover_pending_transition())
            else:
                self.assertTrue(await asyncio.wait_for(recovery,5))
        finally:
            go.set()
            for task in (owner,recovery):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(owner,*([recovery] if recovery else []),return_exceptions=True)
        self.assertEqual(self.revision(),proposed_revision)
        self.assertEqual((await self.store.get_policy_transition())['state'],'completed')
        self.assertFalse(await self.service.recover_pending_transition())
        self.source.coordinator.request_run.assert_not_awaited()
        for root in roots:
            self.assertFalse(filesystem._state(root).writer_active)
            self.assertEqual(filesystem._state(root).waiting_writers,0)

    async def test_multiroot_recovery_waits_previous_only_root_before_policy_lock(self):
        await self.exercise_multiroot_recovery_wait('a-previous')

    async def test_multiroot_recovery_waits_common_root_before_policy_lock(self):
        await self.exercise_multiroot_recovery_wait('root')

    async def test_multiroot_recovery_waits_proposed_only_root_before_policy_lock(self):
        await self.exercise_multiroot_recovery_wait('z-proposed')

    async def test_multiroot_recovery_cancellation_releases_partial_leases_and_waiters(self):
        await self.exercise_multiroot_recovery_wait('z-proposed',cancel=True)

    async def test_repeated_recovery_cancellation_drains_real_sql_before_scope_release(self):
        await self.prepare(persist=True)
        transition = await self.store.get_policy_transition()
        dirty_scopes = []
        def mark_dirty(scopes):
            self.assertTrue(self.lock.locked())
            self.assertTrue(self.fixture.filesystem._state('root').writer_active)
            dirty_scopes.append(scopes)
        module = importlib.import_module('infrastructure.persistence.import_cohort_binding')
        original = module.invalidate_cohort_policy_candidate
        entered, release = threading.Event(), threading.Event()
        def gated(connection, *args, **kwargs):
            result = original(connection, *args, **kwargs)
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Recovery SQL gate timed out')
            return result
        with patch.object(module, 'invalidate_cohort_policy_candidate', side_effect=gated), patch.object(
                self.service._settings, 'mark_scan_scopes_dirty', side_effect=mark_dirty):
            owner = asyncio.create_task(self.service.recover_pending_transition())
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                for _ in range(2):
                    owner.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(owner.done())
                    self.assertTrue(self.lock.locked())
                    self.assertTrue(self.fixture.filesystem._state('root').writer_active)
                    self.assertEqual(self.fixture.events(), [])
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(owner, 5)
            finally:
                release.set()
                await asyncio.gather(owner, return_exceptions=True)
        self.assertEqual(dirty_scopes, [LibraryPolicyService._scope_ids(transition['scopes'])])
        self.reopen()
        self.assertEqual(self.revision(), self.fixture.current.policy_revision)
        self.assertEqual((await self.store.get_policy_transition())['state'], 'completed')
        self.assertEqual(len(self.fixture.events()), 1)

    async def test_previous_config_durable_boundary_disagreement_refuses_abort(self):
        await self.prepare()
        self.fixture.drift()
        before = self.config.read_bytes()
        with self.assertRaisesRegex(StaleRevisionError, 'journal or durable boundary changed'):
            await self.service.recover_pending_transition()
        self.assertEqual(self.config.read_bytes(), before)
        self.assertEqual((await self.store.get_policy_transition())['state'], 'prepared')
        self.assertEqual(self.fixture.events(), [])

    async def test_abort_completion_trigger_cannot_modify_catalog(self):
        await self.prepare()
        before = self.fixture.rows()
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_recovery_abort AFTER UPDATE ON library_policy_transitions WHEN NEW.state='aborted' BEGIN UPDATE local_tracks SET source_path='unexpected'; END")
        with self.assertRaisesRegex(StaleRevisionError, 'abort changed during write'):
            await self.service.recover_pending_transition()
        self.assertEqual(self.fixture.rows(), before)
        self.assertEqual((await self.store.get_policy_transition())['state'], 'prepared')
        self.assertEqual(self.fixture.events(), [])
