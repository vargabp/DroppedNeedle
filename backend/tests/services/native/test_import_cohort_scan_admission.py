"""Candidate pre-index preservation and negative fences; not grouping rollout."""
import importlib
import hashlib
import os
import sqlite3
import unittest
from unittest.mock import patch

import msgspec

from core.exceptions import StaleRevisionError, ValidationError
from services.native.import_cohort_scan_candidate import commit_cohort_scan_candidate


class CohortScanAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module = importlib.import_module('tests.services.native.test_import_cohort_scan')
        self.fixture = module.CohortScanContinuityTests()
        await self.fixture.asyncSetUp()
        self.addCleanup(self.fixture.doCleanups)
        self.store = self.fixture.store
        self.writes = self.fixture.native_scan_writes()
        with self.store._connect() as db:
            self.binding_id = db.execute('SELECT id FROM library_import_cohort_bindings').fetchone()[0]

    async def commit(self, **overrides):
        arguments = dict(store=self.store, filesystem=self.fixture.staged.service._filesystem,
            resolver=self.fixture.staged.fixture.resolver, binding_id=self.binding_id,
            run_id='continuity-scan', writes=self.writes, updated_at=3,
            max_total_bytes=10000000, timeout_seconds=5)
        arguments.update(overrides)
        return await commit_cohort_scan_candidate(**arguments)

    def assert_preserved(self):
        self.fixture.assert_current_binding('Candidate must retain full valid binding')
        with self.store._connect() as db:
            self.assertEqual([(row['id'], row['local_album_id']) for row in db.execute(
                'SELECT id,local_album_id FROM local_tracks ORDER BY id')], self.fixture.original)
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])

    async def test_complete_native_reindex_preserves_valid_binding_without_manual_locks(self):
        await self.commit()
        self.assert_preserved()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM local_albums').fetchone()[0], 1)
            self.assertEqual([row[0] for row in db.execute('SELECT relative_directory FROM library_scan_grouping_contexts')], ['Artist/Album'])
            self.assertEqual([row[0] for row in db.execute('SELECT membership_locked FROM local_tracks')], [0, 0])

    async def test_partial_write_page_is_not_whole_cohort_admission(self):
        with self.assertRaisesRegex(ValidationError, 'complete distinct'):
            await self.commit(writes=self.writes[:1])
        self.assert_preserved()

    async def test_new_recording_map_is_not_overwritten_by_old_binding(self):
        altered = msgspec.structs.replace(self.writes[0], track=msgspec.structs.replace(
            self.writes[0].track, embedded_recording_mbid='changed-recording'))
        with self.assertRaisesRegex(StaleRevisionError, 'metadata disagrees'):
            await self.commit(writes=[altered, self.writes[1]])
        self.assert_preserved()

    async def test_current_policy_boundary_revalidated_inside_write_transaction(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_policy_state SET desired_policy_revision='changed'")
        with self.assertRaisesRegex(StaleRevisionError, 'cohort changed'):
            await self.commit()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_scan_grouping_contexts').fetchone()[0], 0)

    async def test_late_scan_run_failure_rolls_back_catalog_and_contexts(self):
        with self.store._connect() as db:
            revision = db.execute('SELECT value FROM library_catalog_revision').fetchone()[0]
            db.execute("CREATE TRIGGER scan_failure BEFORE UPDATE ON library_scan_runs BEGIN SELECT RAISE(ABORT,'late scan fixture failure'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'late scan fixture failure'):
            await self.commit()
        self.assert_preserved()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT value FROM library_catalog_revision').fetchone()[0], revision)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_scan_grouping_contexts').fetchone()[0], 0)

    async def test_coordinator_lease_is_held_through_actual_native_commit(self):
        original = self.store.commit_scan_index_batch
        filesystem = self.fixture.staged.service._filesystem
        async def observed(*args, **kwargs):
            self.assertTrue(filesystem._cohort_writer_is_held('root'))
            return await original(*args, **kwargs)
        with patch.object(self.store, 'commit_scan_index_batch', side_effect=observed):
            await self.commit()
        self.assert_preserved()

    async def test_missing_native_coordinator_fails_closed(self):
        with self.assertRaisesRegex(ValidationError, 'coordinator'):
            await self.commit(filesystem=None)
        self.assert_preserved()

    async def test_postwrite_validation_rejects_catalog_change_and_rolls_back(self):
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_scan AFTER UPDATE ON local_tracks BEGIN UPDATE local_tracks SET embedded_recording_mbid='changed-by-fixture' WHERE id=NEW.id; END")
        with self.assertRaisesRegex(StaleRevisionError, 'cohort changed'):
            await self.commit()
        self.assert_preserved()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_scan_grouping_contexts').fetchone()[0], 0)

    async def test_same_stat_changed_bytes_do_not_reuse_import_fingerprints(self):
        path = self.fixture.staged.fixture.root / self.fixture.staged.bundle.files[0].destination_relative_path
        before = path.stat()
        content = path.read_bytes()
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaises((StaleRevisionError, ValueError)):
            await self.commit()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_scan_grouping_contexts').fetchone()[0], 0)
            self.assertEqual([(row['id'], row['local_album_id']) for row in db.execute(
                'SELECT id,local_album_id FROM local_tracks ORDER BY id')], self.fixture.original)

    async def test_policy_changes_after_hashing_are_rejected_at_sql_fence(self):
        original = self.store.commit_scan_index_batch
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute("UPDATE library_policy_state SET desired_policy_revision='late-policy'")
            return await original(*args, **kwargs)
        with patch.object(self.store, 'commit_scan_index_batch', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'cohort changed'):
                await self.commit()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_scan_grouping_contexts').fetchone()[0], 0)

    async def test_index_write_cannot_erase_applied_policy_revision(self):
        changed = msgspec.structs.replace(self.writes[0], track=msgspec.structs.replace(
            self.writes[0].track, applied_policy_revision=''))
        with self.assertRaisesRegex(StaleRevisionError, 'metadata disagrees'):
            await self.commit(writes=[changed, self.writes[1]])
        self.assert_preserved()

    async def test_new_unbound_album_member_cannot_be_forced_into_existing_cohort(self):
        with self.store._connect() as db:
            row = dict(db.execute('SELECT * FROM local_tracks LIMIT 1').fetchone())
            row.update(id='unbound-extra', relative_path='Artist/Album/extra.flac',
                file_path=str(self.fixture.staged.fixture.root / 'Artist/Album/extra.flac'),
                path_hash=hashlib.sha256(b'Artist/Album/extra.flac').hexdigest())
            columns = ','.join(row)
            db.execute(f"INSERT INTO local_tracks({columns}) VALUES({','.join('?' for _ in row)})", tuple(row.values()))
        with self.assertRaisesRegex(StaleRevisionError, 'album membership'):
            await self.commit()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM local_tracks').fetchone()[0], 3)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_scan_grouping_contexts').fetchone()[0], 0)
