"""Native scan continuity acceptance tests; currently expose integration gaps.

Run separately from passing candidate suite. No expectedFailure/skip decoration:
these must go green through real whole-cohort admission, not directory regexes
or manual membership locks. All storage/media is disposable.
"""
import importlib
import hashlib
import json
from unittest.mock import AsyncMock, patch
import unittest

from core.exceptions import StaleRevisionError, ValidationError

from models.library_work import ScanInventoryItem, ScanRun
from infrastructure.audio.metadata_engine import legacy_audio_projection
from infrastructure.persistence.import_cohort_binding import read_consistent_cohort
from services.native.library_indexer import LibraryIndexer
from services.native.identification_queue_service import IdentificationQueueService
from services.native.local_album_grouping_service import LocalAlbumGroupingService
from services.native.local_album_grouper import grouping_directory


class CohortScanContinuityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module = importlib.import_module('tests.services.native.test_import_cohort_staged')
        self.staged = module.CohortStagedTests()
        await self.staged.asyncSetUp()
        self.addCleanup(self.staged.doCleanups)
        self.result = await self.staged.publish()
        self.store = self.staged.store
        self.assert_current_binding('Fresh native publication must have a valid whole binding')
        with self.store._connect() as db:
            self.original = [(row['id'], row['local_album_id']) for row in db.execute(
                'SELECT id,local_album_id FROM local_tracks ORDER BY id')]
        self.grouping = LocalAlbumGroupingService(self.store, IdentificationQueueService(self.store))
        self.cohort_grouping = LocalAlbumGroupingService(self.store, IdentificationQueueService(self.store),
            _cohort_resolver_getter=lambda: self.staged.fixture.resolver,
            _cohort_filesystem=self.staged.service._filesystem)
        await self.store.create_scan_run(ScanRun(id='continuity-scan', kind='rescan_files', trigger='manual',
                                               state='indexing', phase='indexing'))

    def assert_current_binding(self, message):
        with self.store._connect() as db:
            db.execute('BEGIN')
            row = db.execute('SELECT id,payload_json FROM library_import_cohort_bindings').fetchone()
            payload = json.loads(row['payload_json'])
            hashes = {member['track']['id']: hashlib.sha256((self.staged.fixture.root /
                member['track']['relative_path']).read_bytes()).hexdigest() for member in payload['members']}
            self.assertIsNotNone(read_consistent_cohort(db, binding_id=row['id'], root_id='root',
                root_path=str(self.staged.fixture.root), album_parent='Artist/Album',
                current_fingerprints=hashes), message)

    async def seed_contexts(self, directories):
        with self.store._connect() as db:
            db.executemany('INSERT INTO library_scan_grouping_contexts '
                '(run_id,root_id,relative_directory) VALUES(?,?,?)',
                [('continuity-scan', 'root', value) for value in directories])

    def native_scan_writes(self):
        indexer = LibraryIndexer(self.store, self.staged.audio, self.grouping,
                                 self.staged.service._filesystem)
        writes = []
        for track_id, request in zip(self.result.local_track_ids, self.staged.bundle.files):
            path = self.staged.fixture.root / request.destination_relative_path
            stat = path.stat()
            tag, info = legacy_audio_projection(self.staged.audio.read(path))
            with self.store._connect() as db:
                revision = db.execute('SELECT stat_revision FROM local_tracks WHERE id=?', (track_id,)).fetchone()[0]
            item = dict(root_id='root', relative_path=request.destination_relative_path,
                absolute_path=str(path), local_track_id=track_id, file_size_bytes=stat.st_size,
                file_mtime_ns=stat.st_mtime_ns, stat_revision=revision,
                policy_revision=self.staged.fixture.bundle.policy_revision,
                effective_policy='automatic', comparison_result='changed')
            writes.append(indexer._prepare_tagged('continuity-scan', item, tag, info, now=3))
        return writes

    async def regroup_and_assert_whole_identity(self, threshold):
        await self.seed_inventory()
        module = importlib.import_module('services.native.local_album_grouping_service')
        with patch.object(module, 'STAGED_GROUPING_THRESHOLD', threshold), patch.object(module, 'STAGING_BATCH_SIZE', 1):
            counts = await self.index_pending()
        self.assertEqual(counts['indexed'], 2)
        self.assertEqual(counts['tag_reads'], 2)
        with self.store._connect() as db:
            current = [(row['id'], row['local_album_id']) for row in db.execute(
                'SELECT id,local_album_id FROM local_tracks ORDER BY id')]
            self.assertEqual(current, self.original, 'Native scan split or reassigned the complete imported cohort')
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])
        self.assert_current_binding('Native regroup invalidated the complete published binding')

    def indexer(self):
        audio = self.staged.audio
        class Reader:
            def read_tags(self, path):
                return legacy_audio_projection(audio.read(path))
        return LibraryIndexer(self.store, Reader(), self.cohort_grouping, self.staged.service._filesystem,
            _cohort_resolver_getter=lambda: self.staged.fixture.resolver)

    async def index_pending(self, **kwargs):
        run, _, _ = await self.store.get_scan_run('continuity-scan')
        return await self.indexer().index(run, frozen_policy_revision=self.staged.bundle.policy_revision, **kwargs)

    async def seed_inventory(self):
        # Fixture discovery boundary only. Actual native inventory admission,
        # tag reads, index commit, counters, grouping and queue are exercised.
        with self.store._connect() as db:
            db.execute('INSERT INTO library_scan_run_scopes (run_id,scope_sequence,root_id,relative_path,root_path,'
                'effective_policy,policy_revision,discovery_state) VALUES(?,?,?,?,?,?,?,?)',
                ('continuity-scan', 0, 'root', '.', str(self.staged.fixture.root), 'automatic',
                 self.staged.bundle.policy_revision, 'completed'))
            rows = [dict(row) for row in db.execute('SELECT * FROM local_tracks ORDER BY id')]
        items = []
        for row in rows:
            path = self.staged.fixture.root / row['relative_path']
            stat = path.stat()
            items.append(ScanInventoryItem(root_id='root', relative_path=row['relative_path'], absolute_path=str(path),
                file_size_bytes=stat.st_size, file_mtime_ns=stat.st_mtime_ns, stat_revision=row['stat_revision'],
                effective_policy='automatic', comparison_result='changed',
                policy_revision=self.staged.bundle.policy_revision, local_track_id=row['id']))
        run, _, _ = await self.store.get_scan_run('continuity-scan')
        await self.store.add_scan_inventory_batch('continuity-scan', items,
            expected_run_revision=run.row_revision, updated_at=3)

    async def test_small_native_grouping_preserves_complete_imported_cohort(self):
        await self.regroup_and_assert_whole_identity(512)

    async def test_paged_native_grouping_preserves_complete_imported_cohort(self):
        # Explicit full PRIVATE profile; production defaults remain OFF.
        for phase in ('tokens', 'summaries', 'continuity', 'provision', 'membership', 'retirement', 'queue'):
            setattr(self.cohort_grouping, f'_cohort_{phase}_candidate_enabled', True)
        await self.regroup_and_assert_whole_identity(0)
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT state,staging_state FROM library_scan_grouping_contexts').fetchone()[:], ('completed', 'completed'))
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_identification_jobs').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT identification_enqueued_count FROM library_scan_runs').fetchone()[0], 1)

    async def test_parent_context_does_not_silently_drop_bound_medium_tracks(self):
        await self.seed_contexts(['Artist/Album'])
        # Observe the actual native grouper input; catalog staying unchanged
        # after receiving zero rows is not a continuity success.
        native = self.cohort_grouping._grouper.group
        with patch.object(self.cohort_grouping._grouper, 'group', wraps=native) as called:
            await self.cohort_grouping.regroup_run('continuity-scan', now=3,
                frozen_policy_revision=self.staged.bundle.policy_revision)
        self.assertEqual({value.local_track_id for value in called.call_args.args[0]},
                         set(self.result.local_track_ids), 'Bound parent context discarded its medium tracks')


class CohortIndexInventoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = CohortScanContinuityTests()
        await self.fixture.asyncSetUp()
        self.addCleanup(self.fixture.doCleanups)
        await self.fixture.seed_inventory()
        self.store = self.fixture.store

    def assert_not_indexed(self):
        self.fixture.assert_current_binding('Refused inventory must not corrupt binding')
        with self.store._connect() as db:
            self.assertTrue(all(row[0] == 'pending' for row in db.execute('SELECT processing_state FROM library_scan_inventory')))
            row = db.execute('SELECT indexed_count,inspected_count,identification_enqueued_count FROM library_scan_runs').fetchone()
            self.assertEqual(tuple(row), (0, 0, 0))
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_scan_grouping_contexts').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_identification_jobs').fetchone()[0], 0)

    async def test_whole_cohort_crosses_fetch_and_tag_batch_boundaries_atomically(self):
        module = importlib.import_module('services.native.library_indexer')
        with patch.object(module, 'INDEX_FETCH_SIZE', 1), patch.object(module, 'TAG_BATCH_SIZE', 1):
            counts = await self.fixture.index_pending()
        self.assertEqual((counts['indexed'], counts['tag_reads'], counts['changed'], counts['identification_enqueued']), (2, 2, 2, 1))
        self.fixture.assert_current_binding('Whole public index must retain binding')
        with self.store._connect() as db:
            self.assertEqual([row[0] for row in db.execute('SELECT processing_state FROM library_scan_inventory')], ['indexed', 'indexed'])
            row = db.execute('SELECT indexed_count,inspected_count,changed_count,identification_enqueued_count FROM library_scan_runs').fetchone()
            self.assertEqual(tuple(row), (2, 2, 2, 1))
            self.assertEqual(db.execute('SELECT state FROM library_scan_grouping_contexts').fetchone()[0], 'completed')
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])

    async def test_partial_inventory_is_not_admitted_or_fallen_back(self):
        with self.store._connect() as db:
            db.execute('DELETE FROM library_scan_inventory WHERE relative_path=(SELECT MAX(relative_path) FROM library_scan_inventory)')
        with self.assertRaisesRegex(StaleRevisionError, 'whole pending cohort inventory'):
            await self.fixture.index_pending()
        self.assert_not_indexed()

    async def test_inventory_track_identity_is_not_inferred_from_path_alone(self):
        with self.store._connect() as db:
            db.execute('UPDATE library_scan_inventory SET local_track_id=NULL')
        with self.assertRaisesRegex(ValidationError, 'whole pending cohort inventory'):
            await self.fixture.index_pending()
        self.assert_not_indexed()

    async def test_incomplete_discovery_scope_cannot_index_complete_member_rows(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_run_scopes SET discovery_state='pending'")
        with self.assertRaisesRegex(StaleRevisionError, 'discovery scope changed or is incomplete'):
            await self.fixture.index_pending()
        self.assert_not_indexed()

    async def test_inventory_absolute_path_tampering_refuses_before_tag_read(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_inventory SET absolute_path='/outside-fixture/file.flac'")
        indexer = self.fixture.indexer()
        run, _, _ = await self.store.get_scan_run('continuity-scan')
        with patch.object(indexer, '_read_stable_tags', new=AsyncMock(side_effect=AssertionError('unsafe tag I/O'))) as read:
            with self.assertRaisesRegex(StaleRevisionError, 'inventory metadata disagrees'):
                await indexer.index(run, frozen_policy_revision=self.fixture.staged.bundle.policy_revision)
        read.assert_not_awaited()
        self.assert_not_indexed()

    async def test_actual_index_commit_holds_native_writer_lease(self):
        original = self.store.commit_scan_index_batch
        filesystem = self.fixture.staged.service._filesystem
        async def observed(*args, **kwargs):
            self.assertTrue(filesystem._cohort_writer_is_held('root'))
            self.assertIsNotNone(kwargs['_cohort_inventory_candidate'])
            return await original(*args, **kwargs)
        with patch.object(self.store, 'commit_scan_index_batch', side_effect=observed):
            self.assertEqual((await self.fixture.index_pending())['indexed'], 2)

    async def test_late_inventory_row_revision_change_refuses_all_writes(self):
        original = self.store.commit_scan_index_batch
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute('UPDATE library_scan_inventory SET row_revision=row_revision+1')
            return await original(*args, **kwargs)
        with patch.object(self.store, 'commit_scan_index_batch', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'inventory row changed'):
                await self.fixture.index_pending()
        self.assert_not_indexed()

    async def test_late_discovery_generation_change_refuses_all_writes(self):
        original = self.store.commit_scan_index_batch
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute('UPDATE library_scan_run_scopes SET discovery_generation=discovery_generation+1')
            return await original(*args, **kwargs)
        with patch.object(self.store, 'commit_scan_index_batch', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'discovery scope changed'):
                await self.fixture.index_pending()
        self.assert_not_indexed()

    async def test_late_pause_control_refuses_index_transaction(self):
        original = self.store.commit_scan_index_batch
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute("UPDATE library_scan_runs SET requested_control='pause'")
            return await original(*args, **kwargs)
        with patch.object(self.store, 'commit_scan_index_batch', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'not an active rescan'):
                await self.fixture.index_pending()
        self.assert_not_indexed()

    async def test_counter_tampering_is_refused_inside_native_index_transaction(self):
        original = self.store.commit_scan_index_batch
        async def changed(*args, **kwargs):
            kwargs['increments']['indexed_count'] = 1
            return await original(*args, **kwargs)
        with patch.object(self.store, 'commit_scan_index_batch', side_effect=changed):
            with self.assertRaisesRegex(ValidationError, 'Exact native cohort inventory counters'):
                await self.fixture.index_pending()
        self.assert_not_indexed()

    async def test_checkpoint_interruption_leaves_pending_inventory_then_resumes_once(self):
        counts = await self.fixture.index_pending(checkpoint=AsyncMock(return_value=False))
        self.assertEqual(counts['indexed'], 0)
        self.assert_not_indexed()
        self.assertEqual((await self.fixture.index_pending())['indexed'], 2)
        self.assertEqual((await self.fixture.index_pending())['indexed'], 0)

    async def test_inventory_mutation_during_upsert_rolls_back_catalog_and_counters(self):
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_inventory AFTER UPDATE OF processing_state ON library_scan_inventory BEGIN UPDATE library_scan_inventory SET comparison_result='excluded' WHERE run_id=NEW.run_id AND root_id=NEW.root_id AND relative_path=NEW.relative_path; END")
        with self.assertRaisesRegex(StaleRevisionError, 'inventory row changed'):
            await self.fixture.index_pending()
        self.assert_not_indexed()
