"""Atomic native queue/context completion candidate, disposable storage only."""
import asyncio
import importlib
import sqlite3
import unittest
from unittest.mock import AsyncMock, patch

import msgspec

from core.exceptions import StaleRevisionError, ValidationError
from services.native.import_cohort_grouping_candidate import complete_cohort_grouping_candidate


class CohortCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module = importlib.import_module('tests.services.native.test_import_cohort_grouping_candidate')
        self.fixture = module.CohortGroupingCandidateTests()
        await self.fixture.asyncSetUp()
        self.addCleanup(self.fixture.doCleanups)
        await self.fixture.apply()
        self.store = self.fixture.store

    async def complete(self, **overrides):
        arguments = dict(store=self.store, filesystem=self.fixture.fixture.fixture.staged.service._filesystem,
            resolver=self.fixture.fixture.fixture.staged.fixture.resolver,
            binding_id=self.fixture.fixture.binding_id, run_id='continuity-scan', now=5,
            max_total_bytes=10000000, timeout_seconds=5)
        arguments.update(overrides)
        return await complete_cohort_grouping_candidate(**arguments)

    def assert_pending(self):
        self.fixture.fixture.assert_preserved()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT state FROM library_scan_grouping_contexts').fetchone()[0], 'pending')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_identification_jobs').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT identification_enqueued_count FROM library_scan_runs').fetchone()[0], 0)

    async def test_queue_and_completion_commit_once_and_replay_cannot_duplicate(self):
        result = await self.complete()
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0][1])
        self.fixture.fixture.assert_preserved()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT state FROM library_scan_grouping_contexts').fetchone()[0], 'completed')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_identification_jobs').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT identification_enqueued_count FROM library_scan_runs').fetchone()[0], 1)
        with self.assertRaisesRegex(StaleRevisionError, 'not pending/running'):
            await self.complete()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_identification_jobs').fetchone()[0], 1)

    async def test_checkpoint_pause_does_not_enqueue_or_complete(self):
        self.assertIsNone(await self.complete(checkpoint=AsyncMock(return_value=False)))
        self.assert_pending()

    async def test_existing_native_job_is_reused_without_duplicate_or_counter_credit(self):
        from services.native.identification_queue_service import IdentificationQueueService
        from services.native.identification_revisions import album_input_revisions
        album = self.fixture.fixture.fixture.original[0][1]
        with self.store._connect() as db:
            rows = [dict(row) for row in db.execute('SELECT * FROM local_tracks WHERE local_album_id=?', (album,))]
        job_id = await IdentificationQueueService(self.store).enqueue_album(album,
            input_revision=':'.join(album_input_revisions(rows)), now=4)
        self.assertEqual(await self.complete(), [(job_id, False)])
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_identification_jobs').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT identification_enqueued_count FROM library_scan_runs').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT state FROM library_scan_grouping_contexts').fetchone()[0], 'completed')

    async def test_late_policy_change_after_read_is_refused_at_completion_fence(self):
        original = self.store.enqueue_identification_job_results
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute("UPDATE library_policy_state SET desired_policy_revision='late-completion-policy'")
            return await original(*args, **kwargs)
        with patch.object(self.store, 'enqueue_identification_job_results', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'cohort changed'):
                await self.complete()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_identification_jobs').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT state FROM library_scan_grouping_contexts').fetchone()[0], 'pending')

    async def test_native_scan_pause_control_is_checked_at_sql_fence(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_runs SET requested_control='pause'")
        with self.assertRaisesRegex(StaleRevisionError, 'not pending/running'):
            await self.complete()
        self.assert_pending()

    async def test_late_context_completion_failure_rolls_back_queue_and_counters(self):
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER completion_failure BEFORE UPDATE OF state ON library_scan_grouping_contexts WHEN NEW.state='completed' BEGIN SELECT RAISE(ABORT,'late completion fixture failure'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'late completion fixture failure'):
            await self.complete()
        self.assert_pending()

    async def test_queued_job_mutation_during_completion_rolls_back_everything(self):
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_queued BEFORE UPDATE OF state ON library_scan_grouping_contexts WHEN NEW.state='completed' BEGIN UPDATE library_identification_jobs SET input_revision='changed-by-fixture'; END")
        with self.assertRaisesRegex(StaleRevisionError, 'queued identification changed'):
            await self.complete()
        self.assert_pending()

    async def test_changed_job_revision_is_refused_by_native_queue_transaction(self):
        original = self.store.enqueue_identification_job_results
        async def changed(jobs, **kwargs):
            return await original([msgspec.structs.replace(jobs[0], input_revision='changed')], **kwargs)
        with patch.object(self.store, 'enqueue_identification_job_results', side_effect=changed):
            with self.assertRaisesRegex(ValidationError, 'Exact fresh native'):
                await self.complete()
        self.assert_pending()

    async def test_ungrouped_context_cannot_be_marked_complete(self):
        with self.store._connect() as db:
            db.execute("UPDATE local_albums SET grouping_key='not-grouped'")
        with self.assertRaisesRegex(StaleRevisionError, 'not passed native grouping'):
            await self.complete()
        self.assert_pending()

    async def test_deliberate_exclusion_remains_excluded_without_false_completion(self):
        album = self.fixture.fixture.fixture.original[0][1]
        with self.store._connect() as db:
            db.execute("INSERT INTO library_identification_reviews(id,local_album_id,state,reason_code,input_revision,created_at,updated_at) VALUES('excluded-fixture',?,'excluded','USER_EXCLUDED','older',1,1)", (album,))
        with self.assertRaisesRegex(StaleRevisionError, 'identification admission refused'):
            await self.complete()
        self.assert_pending()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT state FROM library_identification_reviews').fetchone()[0], 'excluded')

    async def test_manual_track_exclusion_prevents_completion(self):
        with self.store._connect() as db:
            db.execute('UPDATE local_tracks SET manual_excluded=1')
        with self.assertRaisesRegex(StaleRevisionError, 'membership or applied policy'):
            await self.complete()
        self.assert_pending()

    async def test_cancelled_caller_keeps_native_lease_until_queue_write_settles(self):
        entered, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original = self.store.enqueue_identification_job_results
        filesystem = self.fixture.fixture.fixture.staged.service._filesystem
        async def held(*args, **kwargs):
            entered.set()
            await release.wait()
            self.assertTrue(filesystem._cohort_writer_is_held('root'))
            result = await original(*args, **kwargs)
            settled.set()
            return result
        with patch.object(self.store, 'enqueue_identification_job_results', side_effect=held):
            task = asyncio.create_task(self.complete())
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertFalse(settled.is_set())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(settled.is_set())
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT state FROM library_scan_grouping_contexts').fetchone()[0], 'completed')
