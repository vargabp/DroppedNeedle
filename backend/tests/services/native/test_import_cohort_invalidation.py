"""Durable policy-only disabling; no reactivation, cleanup or rebind authority."""
import asyncio
import hashlib
import importlib
import json
import sqlite3
import threading
import unittest
from unittest.mock import patch

import msgspec
from api.v1.schemas.library_policies import LibraryPathPolicyRule
from core.exceptions import StaleRevisionError, ValidationError
from infrastructure.persistence.native_library_store import NativeLibraryStore
from infrastructure.persistence.import_cohort_binding import read_consistent_cohort
from services.native.library_policy_resolver import LibraryPolicyResolver
from services.native.import_cohort_grouping_candidate import invalidate_policy_cohort_candidate


class CohortPolicyInvalidationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module = importlib.import_module('tests.services.native.test_import_cohort_scan')
        self.source = module.CohortScanContinuityTests()
        await self.source.asyncSetUp()
        self.addCleanup(self.source.doCleanups)
        self.path = self.source.staged.fixture.root.parent / 'catalog.db'
        self.store = NativeLibraryStore(self.path, threading.Lock(),
            _cohort_schema_candidate_enabled=True, _cohort_policy_invalidation_enabled=True)
        self.filesystem = self.source.staged.service._filesystem
        self.old = self.source.staged.fixture.resolver
        root = msgspec.structs.replace(self.old.settings.library_roots[0],
            rules=[LibraryPathPolicyRule(id='new-rule', relative_path='Unrelated', policy='excluded')])
        self.current = LibraryPolicyResolver(msgspec.structs.replace(self.old.settings, library_roots=[root]))
        self.assertNotEqual(self.old.policy_revision, self.current.policy_revision)
        with self.store._connect() as db:
            self.binding_id = db.execute('SELECT id FROM library_import_cohort_bindings').fetchone()[0]

    def drift(self):
        with self.store._connect() as db:
            db.execute('UPDATE library_policy_state SET desired_policy_revision=?', (self.current.policy_revision,))

    async def invalidate(self, *, store=None, resolver=None, at=10):
        return await invalidate_policy_cohort_candidate(store=store or self.store, filesystem=self.filesystem,
            resolver=resolver or self.current, binding_id=self.binding_id, invalidated_at=at)

    def rows(self):
        with self.store._connect() as db:
            return {table: [tuple(row) for row in db.execute(f'SELECT * FROM {table} ORDER BY rowid')]
                for table in ('local_tracks', 'local_albums', 'local_artists', 'local_album_artists',
                              'library_management_import_bundles', 'library_management_import_journal',
                              'library_management_metadata_snapshots', 'library_import_cohort_snapshot_sources',
                              'library_import_cohort_bindings', 'library_import_cohort_members', 'library_catalog_revision')}

    def events(self):
        with self.store._connect() as db:
            return [tuple(row) for row in db.execute('SELECT * FROM library_import_cohort_policy_invalidations')]

    def read(self):
        with self.store._connect() as db:
            db.execute('BEGIN')
            payload = json.loads(db.execute('SELECT payload_json FROM library_import_cohort_bindings').fetchone()[0])
            hashes = {member['track']['id']: hashlib.sha256((self.source.staged.fixture.root / member['track']['relative_path']).read_bytes()).hexdigest() for member in payload['members']}
            return read_consistent_cohort(db, binding_id=self.binding_id, root_id='root',
                root_path=str(self.source.staged.fixture.root), album_parent='Artist/Album', current_fingerprints=hashes)

    async def test_policy_event_disables_binding_without_changing_source_or_catalog(self):
        self.assertIsNotNone(self.read())
        self.drift()
        before = self.rows()
        self.assertTrue(await self.invalidate())
        self.assertEqual(self.rows(), before)
        event = self.events()[0]
        self.assertEqual(event[2:5], (self.old.policy_revision, self.current.policy_revision, 'POLICY_BOUNDARY_CHANGED'))
        with self.store._connect() as db:
            db.execute('UPDATE library_policy_state SET desired_policy_revision=?', (self.old.policy_revision,))
        self.assertIsNone(self.read(), 'Restoring old policy must not resurrect an invalidated binding')

    async def test_matching_policy_is_not_invalidated(self):
        with self.assertRaisesRegex(ValueError, 'Fresh durable policy disagreement'):
            await self.invalidate(resolver=self.old)
        self.assertEqual(self.events(), [])
        self.assertIsNotNone(self.read())

    async def test_private_schema_opt_in_does_not_enable_invalidation(self):
        self.drift()
        disabled = NativeLibraryStore(self.path, threading.Lock(), _cohort_schema_candidate_enabled=True)
        with self.assertRaisesRegex(ValueError, 'invalidation is disabled'):
            await self.invalidate(store=disabled)
        self.assertEqual(self.events(), [])

    async def test_configured_and_durable_policy_disagreement_refuses_event(self):
        with self.assertRaisesRegex(ValueError, 'Fresh durable policy disagreement'):
            await self.invalidate()
        self.assertEqual(self.events(), [])

    async def test_repeated_disabling_preserves_first_event_and_timestamp(self):
        self.drift()
        self.assertTrue(await self.invalidate())
        before = self.events()
        self.assertFalse(await self.invalidate(at=20))
        self.assertEqual(self.events(), before)

    async def test_events_cannot_be_updated_or_deleted(self):
        self.drift()
        await self.invalidate()
        before = self.events()
        with self.store._connect() as db:
            for sql in ("UPDATE library_import_cohort_policy_invalidations SET invalidated_at=20", 'DELETE FROM library_import_cohort_policy_invalidations'):
                with self.assertRaisesRegex(sqlite3.IntegrityError, 'invalidation is immutable'):
                    db.execute(sql)
                db.rollback()
        self.assertEqual(self.events(), before)

    async def test_native_restart_and_policy_restoration_do_not_reactivate(self):
        self.drift()
        await self.invalidate()
        event = self.events()
        with self.store._connect() as db:
            db.execute('UPDATE library_policy_state SET desired_policy_revision=?', (self.old.policy_revision,))
        NativeLibraryStore(self.path, threading.Lock(), _cohort_schema_candidate_enabled=True)
        self.assertEqual(self.events(), event)
        self.assertIsNone(self.read())

    async def test_late_durable_policy_change_refuses_event_under_root_lease(self):
        self.drift()
        original = self.store._invalidate_cohort_policy_candidate
        async def changed(*args, **kwargs):
            self.assertTrue(self.filesystem._cohort_writer_is_held('root'))
            with self.store._connect() as db:
                db.execute("UPDATE library_policy_state SET desired_policy_revision='different-late-policy'")
            return await original(*args, **kwargs)
        with patch.object(self.store, '_invalidate_cohort_policy_candidate', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'Fresh durable policy disagreement'):
                await self.invalidate()
        self.assertEqual(self.events(), [])

    async def test_late_binding_seal_change_refuses_event(self):
        self.drift()
        original = self.store._invalidate_cohort_policy_candidate
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute("UPDATE library_import_cohort_bindings SET payload_sha256=?", ('f' * 64,))
            return await original(*args, **kwargs)
        with patch.object(self.store, '_invalidate_cohort_policy_candidate', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'binding seal changed'):
                await self.invalidate()
        self.assertEqual(self.events(), [])

    async def test_postinsert_catalog_mutation_rolls_back_event_and_catalog(self):
        self.drift()
        before = self.rows()
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_invalidation AFTER INSERT ON library_import_cohort_policy_invalidations BEGIN UPDATE local_tracks SET source_path='changed-by-invalidation-fixture'; END")
        with self.assertRaisesRegex(ValueError, 'invalidation changed during write'):
            await self.invalidate()
        self.assertEqual(self.events(), [])
        self.assertEqual(self.rows(), before)

    async def test_wrong_configured_root_path_refuses_event(self):
        self.drift()
        root = msgspec.structs.replace(self.current.settings.library_roots[0], path=str(self.source.staged.fixture.root.parent / 'other'))
        wrong = LibraryPolicyResolver(msgspec.structs.replace(self.current.settings, library_roots=[root]))
        with self.assertRaisesRegex(ValidationError, 'Configured native automatic root'):
            await self.invalidate(resolver=wrong)
        self.assertEqual(self.events(), [])

    async def test_invalid_clock_refuses_event(self):
        self.drift()
        for at in (float('inf'), float('nan'), True, -1):
            with self.subTest(at=at), self.assertRaisesRegex(ValueError, 'Exact policy invalidation evidence'):
                await self.invalidate(at=at)
        self.assertEqual(self.events(), [])

    async def test_manual_exclusions_are_preserved_not_repaired(self):
        self.drift()
        with self.store._connect() as db:
            db.execute('UPDATE local_tracks SET manual_excluded=1,membership_locked=1')
        before = self.rows()
        self.assertTrue(await self.invalidate())
        self.assertEqual(self.rows(), before)

    async def test_missing_invalidation_table_cannot_resurrect_binding_or_be_auto_repaired(self):
        self.drift()
        await self.invalidate()
        with self.store._connect() as db:
            db.execute('UPDATE library_policy_state SET desired_policy_revision=?', (self.old.policy_revision,))
            db.execute('DROP TABLE library_import_cohort_policy_invalidations')
        self.assertIsNone(self.read())
        with self.assertRaisesRegex(ValueError, 'Partial or changed'):
            NativeLibraryStore(self.path, threading.Lock(), _cohort_schema_candidate_enabled=True)

    async def _inflight_invalidation(self, *, cancel=False, fail=False, reader=False):
        """Pause real SQL after INSERT but before native commit or rollback."""
        module = importlib.import_module('infrastructure.persistence.import_cohort_binding')
        original = module.invalidate_cohort_policy_candidate
        self.drift()
        before = self.rows()
        entered, release = threading.Event(), threading.Event()
        competing_started, competing_entered = asyncio.Event(), asyncio.Event()
        observed_events = []

        def gated(connection, *args, **kwargs):
            result = original(connection, *args, **kwargs)
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test SQL gate timed out')
            if fail:
                raise ValueError('Injected postinsert transaction failure')
            return result

        async def competitor():
            competing_started.set()
            lease = self.filesystem.read('root') if reader else self.filesystem.write('root')
            async with lease:
                competing_entered.set()
                observed_events.append(self.events())

        with patch.object(module, 'invalidate_cohort_policy_candidate', side_effect=gated):
            owner = asyncio.create_task(self.invalidate())
            competing = None
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                self.assertEqual(self.events(), [], 'Uncommitted event must not be visible')
                competing = asyncio.create_task(competitor())
                await asyncio.wait_for(competing_started.wait(), 2)
                self.assertFalse(competing_entered.is_set())
                if cancel:
                    for _ in range(2):
                        owner.cancel()
                        await asyncio.sleep(0)
                        self.assertFalse(owner.done(), 'Cancelled caller must drain native SQL')
                        self.assertFalse(competing_entered.is_set())
                release.set()
                if fail:
                    with self.assertRaisesRegex(ValueError, 'Injected postinsert'):
                        await asyncio.wait_for(owner, 5)
                elif cancel:
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(owner, 5)
                else:
                    self.assertTrue(await asyncio.wait_for(owner, 5))
                await asyncio.wait_for(competing, 5)
            finally:
                release.set()
                await asyncio.gather(owner, *([competing] if competing else []), return_exceptions=True)
        self.assertTrue(competing_entered.is_set())
        self.assertEqual(self.rows(), before)
        self.assertEqual(len(self.events()), 0 if fail else 1)
        self.assertEqual(observed_events, [self.events()], 'Competitor must see settled SQL only')

    async def test_competing_writer_waits_until_real_invalidation_commit(self):
        await self._inflight_invalidation()

    async def test_competing_reader_waits_until_real_invalidation_commit(self):
        await self._inflight_invalidation(reader=True)

    async def test_repeated_cancellation_keeps_writer_lease_until_commit(self):
        await self._inflight_invalidation(cancel=True)

    async def test_repeated_cancellation_keeps_writer_lease_until_rollback(self):
        await self._inflight_invalidation(cancel=True, fail=True)

    async def test_concurrent_duplicate_disabling_commits_exactly_one_event(self):
        self.drift()
        before = self.rows()
        results = await asyncio.wait_for(asyncio.gather(self.invalidate(), self.invalidate()), 5)
        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(len(self.events()), 1)
        self.assertEqual(self.rows(), before)

    async def test_cancellation_while_waiting_for_root_never_starts_sql(self):
        self.drift()
        before = self.rows()
        discovered = asyncio.Event()
        original = self.store._read
        async def discovery(operation):
            result = await original(operation)
            discovered.set()
            return result
        with patch.object(self.store, '_read', side_effect=discovery), patch.object(
                self.store, '_invalidate_cohort_policy_candidate', wraps=self.store._invalidate_cohort_policy_candidate) as sql:
            async with self.filesystem.write('root'):
                owner = asyncio.create_task(self.invalidate())
                try:
                    await asyncio.wait_for(discovered.wait(), 2)
                    owner.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(owner, 2)
                finally:
                    if not owner.done():
                        owner.cancel()
                    await asyncio.gather(owner, return_exceptions=True)
                sql.assert_not_called()
        self.assertEqual(self.events(), [])
        self.assertEqual(self.rows(), before)

class CohortPolicyTransitionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.base = CohortPolicyInvalidationTests()
        await self.base.asyncSetUp()
        self.addCleanup(self.base.doCleanups)
        self.store = NativeLibraryStore(self.base.path, threading.Lock(),
            _cohort_schema_candidate_enabled=True, _cohort_policy_invalidation_enabled=True,
            _cohort_policy_transition_enabled=True)
        self.lock = asyncio.Lock()
        self.resolver = self.base.current

    async def prepare(self, resolver=None):
        from models.library_work import ScanScope
        self.resolver = resolver or self.resolver
        old_root = self.base.old.settings.library_roots[0]
        new_roots = {root.id: root for root in self.resolver.settings.library_roots}
        new_root = new_roots.get(old_root.id)
        await self.store.prepare_policy_transition(
            previous_policy_revision=self.base.old.policy_revision,
            proposed_policy_revision=self.resolver.policy_revision,
            previous_settings_json=msgspec.json.encode(self.base.old.settings).decode(),
            proposed_settings_json=msgspec.json.encode(self.resolver.settings).decode(),
            scopes=[ScanScope(root_id=old_root.id, root_path=old_root.path, scope_id=old_root.id,
                effective_policy=new_root.policy if new_root else 'excluded',
                policy_revision=self.resolver.policy_revision)], prepared_at=20)

    async def commit(self, *, store=None, getter=None, at=30):
        from services.native.import_cohort_grouping_candidate import commit_policy_transition_cohort_candidate
        return await commit_policy_transition_cohort_candidate(store=store or self.store,
            filesystem=self.base.filesystem, resolver_getter=getter or (lambda: self.resolver),
            transition_lock=self.lock, updated_at=at)

    def boundary_rows(self):
        with self.store._connect() as db:
            return {table: [tuple(row) for row in db.execute(f'SELECT * FROM {table}')]
                for table in ('library_policy_state', 'library_policy_transitions')}

    async def test_native_boundary_and_event_commit_together(self):
        await self.prepare()
        result = await self.commit()
        self.assertEqual(set(result), {'changed', 'cancelled'})
        self.assertEqual(len(self.base.events()), 1)
        self.assertEqual((await self.store.get_policy_transition())['state'], 'completed')
        self.assertEqual((await self.store.get_pending_policy())['desired_policy_revision'], self.resolver.policy_revision)

    async def test_excluded_previous_root_is_leased_and_invalidated_not_reactivated(self):
        old_root = self.base.old.settings.library_roots[0]
        root = msgspec.structs.replace(old_root, policy='excluded')
        await self.prepare(LibraryPolicyResolver(msgspec.structs.replace(self.base.old.settings, library_roots=[root])))
        with self.store._connect() as db:
            before = [dict(row) for row in db.execute('SELECT * FROM local_tracks ORDER BY id')]
        original = self.store.commit_policy_transition
        async def checked(**kwargs):
            self.assertTrue(self.base.filesystem._cohort_writer_is_held(old_root.id))
            self.assertTrue(self.lock.locked())
            return await original(**kwargs)
        with patch.object(self.store, 'commit_policy_transition', side_effect=checked):
            await self.commit()
        self.assertEqual(len(self.base.events()), 1)
        with self.store._connect() as db:
            rows = [dict(row) for row in db.execute('SELECT * FROM local_tracks ORDER BY id')]
        self.assertTrue(rows)
        for previous, current in zip(before, rows, strict=True):
            expected = dict(previous, desired_policy_revision=self.resolver.policy_revision,
                row_revision=previous['row_revision'] + 1)
            self.assertEqual(current, expected, 'Desired boundary must not eagerly reconcile applied policy or ownership')
        self.assertIsNone(self.base.read())

    async def test_removed_root_and_replacement_root_both_leased(self):
        old = self.base.old.settings.library_roots[0]
        new = msgspec.structs.replace(old, id='replacement', path=str(self.base.source.staged.fixture.root.parent / 'replacement'))
        await self.prepare(LibraryPolicyResolver(msgspec.structs.replace(self.base.old.settings, library_roots=[new])))
        original = self.store.commit_policy_transition
        async def checked(**kwargs):
            self.assertTrue(self.base.filesystem._cohort_writer_is_held(old.id))
            self.assertTrue(self.base.filesystem._cohort_writer_is_held(new.id))
            return await original(**kwargs)
        with patch.object(self.store, 'commit_policy_transition', side_effect=checked):
            await self.commit()
        self.assertEqual(len(self.base.events()), 1)

    async def test_failure_after_event_rolls_back_boundary_catalog_and_event(self):
        await self.prepare()
        before, catalog = self.boundary_rows(), self.base.rows()
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER fail_transition BEFORE UPDATE ON library_policy_transitions WHEN NEW.state='completed' BEGIN SELECT RAISE(ABORT,'injected completion failure'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'injected completion'):
            await self.commit()
        self.assertEqual(self.boundary_rows(), before)
        self.assertEqual(self.base.rows(), catalog)
        self.assertEqual(self.base.events(), [])

    async def test_post_completion_transition_corruption_rolls_everything_back(self):
        await self.prepare()
        before, catalog = self.boundary_rows(), self.base.rows()
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_transition AFTER UPDATE ON library_policy_transitions WHEN NEW.state='completed' BEGIN UPDATE library_policy_transitions SET prepared_at=999; END")
        with self.assertRaisesRegex(ValueError, 'Completed cohort transition changed'):
            await self.commit()
        self.assertEqual(self.boundary_rows(), before)
        self.assertEqual(self.base.rows(), catalog)
        self.assertEqual(self.base.events(), [])

    async def test_invalidation_opt_in_does_not_grant_transition_permission(self):
        await self.prepare()
        before = self.boundary_rows()
        with self.assertRaisesRegex(ValueError, 'transition is disabled'):
            await self.commit(store=self.base.store)
        self.assertEqual(self.boundary_rows(), before)
        self.assertEqual(self.base.events(), [])

    async def test_configuration_change_while_acquiring_roots_refuses_commit(self):
        await self.prepare()
        values = iter([self.resolver, self.base.old])
        with self.assertRaisesRegex(StaleRevisionError, 'Configured cohort transition changed'):
            await self.commit(getter=lambda: next(values))
        self.assertEqual(self.base.events(), [])
        self.assertEqual((await self.store.get_policy_transition())['state'], 'prepared')

    async def test_late_prepared_transition_change_refuses_commit(self):
        await self.prepare()
        original = self.store.commit_policy_transition
        async def changed(**kwargs):
            with self.store._connect() as db:
                db.execute('UPDATE library_policy_transitions SET prepared_at=999')
            return await original(**kwargs)
        with patch.object(self.store, 'commit_policy_transition', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'transition seal changed'):
                await self.commit()
        self.assertEqual(self.base.events(), [])
        self.assertEqual((await self.store.get_pending_policy())['desired_policy_revision'], self.base.old.policy_revision)

    async def test_durable_configuration_disagreement_refuses_transition(self):
        await self.prepare()
        self.base.drift()
        before = self.boundary_rows()
        with self.assertRaisesRegex(ValueError, 'Durable cohort transition boundary disagrees'):
            await self.commit()
        self.assertEqual(self.boundary_rows(), before)
        self.assertEqual(self.base.events(), [])

    async def test_unchanged_revision_resave_does_not_invalidate(self):
        await self.prepare(self.base.old)
        await self.commit()
        self.assertEqual(self.base.events(), [])
        self.assertIsNotNone(self.base.read())

    async def test_invalid_clock_refuses_transition_without_changes(self):
        await self.prepare()
        before = self.boundary_rows()
        for at in (float('nan'), float('inf'), True, -1):
            with self.subTest(at=at), self.assertRaisesRegex(ValueError, 'transition clock'):
                await self.commit(at=at)
        self.assertEqual(self.boundary_rows(), before)
        self.assertEqual(self.base.events(), [])

    async def test_enabled_transition_cannot_silently_omit_seal(self):
        await self.prepare()
        before = self.boundary_rows()
        with self.assertRaisesRegex(ValueError, 'transition seal required'):
            await self.store.commit_policy_transition(proposed_policy_revision=self.resolver.policy_revision, updated_at=30)
        self.assertEqual(self.boundary_rows(), before)
        self.assertEqual(self.base.events(), [])

    async def test_default_native_transition_remains_unwired(self):
        await self.prepare()
        await self.base.store.commit_policy_transition(proposed_policy_revision=self.resolver.policy_revision, updated_at=30)
        self.assertEqual((await self.store.get_policy_transition())['state'], 'completed')
        self.assertEqual(self.base.events(), [])

    async def test_late_binding_checksum_change_refuses_transition(self):
        await self.prepare()
        original = self.store.commit_policy_transition
        async def changed(**kwargs):
            with self.store._connect() as db:
                db.execute("UPDATE library_import_cohort_bindings SET payload_sha256=?", ('f' * 64,))
            return await original(**kwargs)
        with patch.object(self.store, 'commit_policy_transition', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'binding checksum changed'):
                await self.commit()
        self.assertEqual(self.base.events(), [])
        self.assertEqual((await self.store.get_policy_transition())['state'], 'prepared')

    async def test_completion_trigger_cannot_sneak_in_catalog_writes(self):
        await self.prepare()
        before, catalog = self.boundary_rows(), self.base.rows()
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_catalog AFTER UPDATE ON library_policy_transitions WHEN NEW.state='completed' BEGIN UPDATE local_tracks SET source_path='unexpected'; END")
        with self.assertRaisesRegex(ValueError, 'Completed cohort transition changed'):
            await self.commit()
        self.assertEqual(self.boundary_rows(), before)
        self.assertEqual(self.base.rows(), catalog)
        self.assertEqual(self.base.events(), [])

    async def _cancel_transition(self, *, fail):
        await self.prepare()
        before, catalog = self.boundary_rows(), self.base.rows()
        module = importlib.import_module('infrastructure.persistence.import_cohort_binding')
        original = module.invalidate_cohort_policy_candidate
        entered, release = threading.Event(), threading.Event()
        acquired, started = asyncio.Event(), asyncio.Event()
        def gated(connection, *args, **kwargs):
            result = original(connection, *args, **kwargs)
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Transition test gate timed out')
            if fail:
                raise ValueError('Injected transition rollback')
            return result
        async def competing():
            started.set()
            async with self.lock:
                async with self.base.filesystem.write('root'):
                    acquired.set()
        with patch.object(module, 'invalidate_cohort_policy_candidate', side_effect=gated):
            owner = asyncio.create_task(self.commit())
            other = None
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                other = asyncio.create_task(competing())
                await asyncio.wait_for(started.wait(), 2)
                for _ in range(2):
                    owner.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(owner.done())
                    self.assertTrue(self.lock.locked())
                    self.assertFalse(acquired.is_set())
                self.assertEqual(self.boundary_rows(), before, 'Readers see old boundary until transaction settles')
                self.assertEqual(self.base.events(), [])
                release.set()
                with self.assertRaisesRegex(ValueError, 'Injected transition rollback') if fail else self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(owner, 5)
                await asyncio.wait_for(other, 5)
            finally:
                release.set()
                await asyncio.gather(owner, *([other] if other else []), return_exceptions=True)
        self.assertTrue(acquired.is_set())
        self.assertEqual(len(self.base.events()), 0 if fail else 1)
        if fail:
            self.assertEqual(self.boundary_rows(), before)
            self.assertEqual(self.base.rows(), catalog)
        else:
            self.assertEqual((await self.store.get_policy_transition())['state'], 'completed')

    async def test_repeated_cancellation_keeps_transition_and_root_leases_until_commit(self):
        await self._cancel_transition(fail=False)

    async def test_repeated_cancellation_keeps_transition_and_root_leases_until_rollback(self):
        await self._cancel_transition(fail=True)

    async def test_publisher_root_then_policy_order_cannot_deadlock_transition(self):
        await self.prepare()
        publisher = self.base.source.staged.service
        publisher._policy_transition_lock = self.lock
        held, discovered, go, policy_entered = [asyncio.Event() for _ in range(4)]
        original = self.store._read
        async def discovery(operation):
            result = await original(operation)
            discovered.set()
            return result
        async def publish_scope():
            async with self.base.filesystem.write('root'):
                held.set()
                await go.wait()
                async with publisher._policy_scope_candidate():
                    policy_entered.set()
        publication = asyncio.create_task(publish_scope())
        transition = None
        try:
            await asyncio.wait_for(held.wait(), 2)
            with patch.object(self.store, '_read', side_effect=discovery):
                transition = asyncio.create_task(self.commit())
                await asyncio.wait_for(discovered.wait(), 2)
                self.assertFalse(self.lock.locked(), 'Transition must not hold policy lock while waiting for publisher root')
                go.set()
                await asyncio.wait_for(asyncio.gather(publication, transition), 5)
            self.assertTrue(policy_entered.is_set())
            self.assertEqual(len(self.base.events()), 1)
        finally:
            go.set()
            for task in (publication, transition):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(publication, *([transition] if transition else []), return_exceptions=True)
