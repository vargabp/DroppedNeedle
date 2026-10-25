"""Policy ownership and default-OFF actual commit-callback integration."""
import asyncio
import importlib
import hashlib
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import msgspec

from core.exceptions import ConfigurationError, ValidationError, StaleRevisionError
from repositories import musicbrainz_base as mb
from models.library_management import LibraryManagementBlob


class CohortPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module = importlib.import_module('tests.services.native.test_import_cohort_provider')
        self.provider = module.CohortProviderTests()
        await self.provider.asyncSetUp()
        self.addCleanup(self.provider.doCleanups)
        self.atomic = self.provider.atomic
        self.service = self.atomic.fixture.service

    async def correlate(self):
        projection = await self.provider.project()
        await self.provider.correlate(projection)
        # Fixtures construct duration/confidence with integer literals; the
        # real sealed native DTO decoder normalizes declared float fields.
        self.atomic.fixture.bundle = msgspec.json.decode(
            msgspec.json.encode(self.atomic.fixture.bundle), type=type(self.atomic.fixture.bundle))
        requests = {value.ordinal: value for value in self.atomic.fixture.bundle.files}
        self.atomic.fixture.published = [msgspec.structs.replace(value, request=requests[value.request.ordinal])
                                        for value in self.atomic.fixture.published]
        self.service._cohort_import_candidate_enabled = True

    async def callback(self, **overrides):
        arguments = dict(expected_policy_revision=self.atomic.fixture.bundle.policy_revision,
                         source_context=self.provider.context)
        arguments.update(overrides)
        module = importlib.import_module('services.native.target_import_library_service')
        # Native snapshot fixture times are1/2; never alter global runtime clock.
        with patch.object(module, 'time', SimpleNamespace(time=lambda: 2)):
            return await self.service._commit_published_import_bundle(
                'bundle', tuple(self.atomic.fixture.published), **arguments)

    async def test_same_task_policy_nesting_uses_the_same_held_lock(self):
        async def nested():
            async with self.service._policy_scope_candidate():
                self.assertTrue(self.service._policy_transition_lock.locked())
                async with self.service._policy_scope_candidate():
                    self.assertTrue(self.service._policy_transition_lock.locked())
                self.assertTrue(self.service._policy_transition_lock.locked())
        await asyncio.wait_for(nested(), 1)
        self.assertFalse(self.service._policy_transition_lock.locked())

    async def test_child_task_cannot_inherit_policy_lock_ownership(self):
        started, entered = asyncio.Event(), asyncio.Event()
        async def child():
            started.set()
            async with self.service._policy_scope_candidate():
                entered.set()
        async with self.service._policy_scope_candidate():
            task = asyncio.create_task(child())
            await started.wait()
            self.assertFalse(entered.is_set())
        await asyncio.wait_for(task, 1)
        self.assertTrue(entered.is_set())

    async def test_policy_owner_revoked_after_exception(self):
        with self.assertRaisesRegex(ValueError, 'fixture'):
            async with self.service._policy_scope_candidate():
                raise ValueError('fixture')
        self.assertFalse(self.service._policy_transition_lock.locked())
        self.assertIsNone(self.service._candidate_policy_scope.get())

    async def test_opt_in_callback_requires_publisher_root_lease(self):
        await self.correlate()
        with self.assertRaisesRegex(ConfigurationError, 'publisher-owned'):
            await self.callback()
        self.atomic.assert_uncommitted()

    async def test_callback_keeps_full_automatic_baseline_guard_without_deadlock(self):
        await self.correlate()
        async def native_callback():
            # Same ownership pattern as the real publisher's commit closure.
            async with self.service._policy_scope_candidate():
                with self.assertRaisesRegex(ValidationError, 'baseline.*durably prepared'):
                    await self.callback()
        async with self.service._filesystem.write_many(['root']):
            await asyncio.wait_for(asyncio.create_task(native_callback()), 2)
        self.atomic.assert_uncommitted()

    async def test_missing_source_observation_refused_by_callback(self):
        await self.correlate()
        with self.atomic.store._connect() as db:
            db.execute('DELETE FROM library_import_cohort_snapshot_sources')
        async with self.service._filesystem.write_many(['root']):
            with self.assertRaisesRegex(StaleRevisionError, 'matching native'):
                await self.callback()
        self.atomic.assert_uncommitted()

    async def test_current_context_cannot_reuse_old_observation_in_callback(self):
        await self.correlate()
        with patch.object(mb, '_mb_source_generation', self.provider.context.generation + 1):
            current = mb.capture_mb_source_context()
            async with self.service._filesystem.write_many(['root']):
                with self.assertRaisesRegex(StaleRevisionError, 'matching native'):
                    await self.callback(source_context=current)
        self.atomic.assert_uncommitted()

    async def test_default_off_does_not_route_to_candidate(self):
        self.assertFalse(self.service._cohort_import_candidate_enabled)
        await self.correlate()
        self.service._cohort_import_candidate_enabled = False
        self.service._commit_cohort_import_candidate_callback = AsyncMock(side_effect=AssertionError('must stay disabled'))
        # Existing pinned requests still require the normal native source guard.
        with self.assertRaises(ConfigurationError):
            await self.callback(source_context=None)
        self.service._commit_cohort_import_candidate_callback.assert_not_awaited()

    async def prepare_synthetic_baselines(self, *, limit=2):
        # Exercise native durable baseline APIs, not audio parsing/restoration.
        # The fixture media are synthetic bytes, not playable FLAC files.
        for value in self.atomic.fixture.published[:limit]:
            ordinal = value.request.ordinal
            content = msgspec.json.encode({'fixture_semantics': ordinal})
            digest = hashlib.sha256(content).hexdigest()
            relative = f'fixture-blobs/{digest}.json'
            path = self.atomic.fixture.root.parent / relative
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(content)
            await self.atomic.store.register_management_blob(LibraryManagementBlob(
                sha256=digest, kind='tag_snapshot', byte_length=len(content),
                relative_path=relative, created_at=1))
            with self.atomic.store._connect() as db:
                revision = db.execute('SELECT row_revision FROM library_management_import_journal '
                                      'WHERE bundle_id=? AND ordinal=?', ('bundle', ordinal)).fetchone()[0]
            await self.atomic.store.set_library_management_import_baseline(
                'bundle', ordinal, expected_row_revision=revision,
                baseline_blob_sha256=digest, baseline_format='flac',
                baseline_adapter_version='fixture-only', baseline_stat_revision=f'stat-{ordinal}',
                baseline_tag_revision=f'tags-{ordinal}', baseline_image_snapshot_json='[]',
                baseline_ancillary_snapshot_json='[]', baseline_file_mtime_ns=1,
                baseline_file_mode=0o644, updated_at=1)

    async def callback_under_publisher_scopes(self):
        async def native_callback():
            async with self.service._policy_scope_candidate():
                return await self.callback()
        async with self.service._filesystem.write_many(['root']):
            return await asyncio.wait_for(asyncio.create_task(native_callback()), 3)

    def automatic_counts(self):
        tables = ('library_operation_jobs', 'library_operation_work',
                  'library_management_job_snapshots', 'library_management_job_metadata_snapshots',
                  'library_management_plan_items', 'library_management_baselines',
                  'library_management_operation_snapshots', 'library_management_blob_references',
                  'library_track_management_state', 'library_file_mutation_journal')
        with self.atomic.store._connect() as db:
            return {table: db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] for table in tables}

    async def test_actual_automatic_callback_commits_baselines_history_and_cohort(self):
        await self.correlate()
        await self.prepare_synthetic_baselines()
        await self.callback_under_publisher_scopes()
        counts = self.automatic_counts()
        self.assertEqual(counts, dict(library_operation_jobs=1, library_operation_work=1,
            library_management_job_snapshots=1, library_management_job_metadata_snapshots=1,
            library_management_plan_items=2, library_management_baselines=2,
            library_management_operation_snapshots=2, library_management_blob_references=4,
            library_track_management_state=2, library_file_mutation_journal=2))
        with self.atomic.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_import_cohort_bindings').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT COUNT(DISTINCT local_album_id) FROM local_tracks').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM local_tracks').fetchone()[0], 2)
            self.assertEqual(db.execute('SELECT state FROM library_operation_jobs').fetchone()[0], 'succeeded')
            self.assertEqual([row[0] for row in db.execute('SELECT status FROM held_imports')], ['imported', 'imported'])
            self.assertEqual(db.execute("SELECT state FROM library_management_import_bundles WHERE id='bundle'").fetchone()[0], 'catalog_committed')
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])

    async def test_late_bundle_failure_rolls_back_automatic_history_and_cohort(self):
        await self.correlate()
        await self.prepare_synthetic_baselines()
        before = self.automatic_counts()
        with self.atomic.store._connect() as db:
            revision = db.execute('SELECT value FROM library_catalog_revision').fetchone()[0]
            db.execute("CREATE TRIGGER late_failure BEFORE UPDATE OF state ON library_management_import_bundles WHEN NEW.state='catalog_committed' BEGIN SELECT RAISE(ABORT,'late automatic fixture failure'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'late automatic fixture failure'):
            await self.callback_under_publisher_scopes()
        self.atomic.assert_uncommitted()
        self.assertEqual(self.automatic_counts(), before)
        with self.atomic.store._connect() as db:
            self.assertEqual(db.execute('SELECT value FROM library_catalog_revision').fetchone()[0], revision)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_management_blobs').fetchone()[0], 2)
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])

    async def test_second_missing_baseline_rolls_back_first_automatic_track(self):
        await self.correlate()
        await self.prepare_synthetic_baselines(limit=1)
        before = self.automatic_counts()
        with self.assertRaisesRegex(ValidationError, 'baseline.*durably prepared'):
            await self.callback_under_publisher_scopes()
        self.atomic.assert_uncommitted()
        self.assertEqual(self.automatic_counts(), before)
