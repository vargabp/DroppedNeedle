"""Private native startup migration and reference preservation, disposable DBs."""
import importlib
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from infrastructure.persistence import import_cohort_binding as candidate
from infrastructure.persistence.native_library_store import NativeLibraryStore


class CohortMigrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'native.db'
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE auth_users(id TEXT PRIMARY KEY)')
            db.execute("INSERT INTO auth_users VALUES('preserved-user')")
        self.store = NativeLibraryStore(self.path, threading.Lock())

    def enable(self):
        return NativeLibraryStore(self.path, threading.Lock(), _cohort_schema_candidate_enabled=True)

    def test_native_default_startup_does_not_install_candidate(self):
        with self.store._connect() as db:
            self.assertEqual(candidate._cohort_schema_objects(db), {})
        NativeLibraryStore(self.path, threading.Lock())
        with self.store._connect() as db:
            self.assertEqual(candidate._cohort_schema_objects(db), {})

    def test_native_opt_in_upgrade_is_exact_and_restart_idempotent(self):
        self.enable()
        with self.store._connect() as db:
            first = candidate._cohort_schema_objects(db)
            core, native = candidate._cohort_schema_manifest()
            self.assertEqual(first, core | native)
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])
        self.enable()
        with self.store._connect() as db:
            self.assertEqual(candidate._cohort_schema_objects(db), first)
            self.assertEqual(db.execute('SELECT id FROM auth_users').fetchone()[0], 'preserved-user')

    def test_exact_v1_schema_with_native_tracking_upgrades_additively_to_v2(self):
        with self.store._connect() as db:
            db.executescript(candidate.CANDIDATE_V1_SCHEMA_SQL)
            NativeLibraryStore._ensure_foreign_key_validation_triggers(db)
            old = candidate._cohort_schema_objects(db)
        self.enable()
        with self.store._connect() as db:
            new = candidate._cohort_schema_objects(db)
            self.assertTrue(old.items() <= new.items())
            self.assertEqual(tuple(db.execute('SELECT singleton,version FROM library_import_cohort_schema_state').fetchone()), (1, 2))
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_import_cohort_policy_invalidations').fetchone()[0], 0)

    def test_v2_marker_prevents_missing_lifecycle_table_being_treated_as_legacy(self):
        self.enable()
        with self.store._connect() as db:
            db.execute('DROP TABLE library_import_cohort_policy_invalidations')
            before = candidate._cohort_schema_objects(db)
        with self.assertRaisesRegex(ValueError, 'Partial or changed'):
            self.enable()
        with self.store._connect() as db:
            self.assertEqual(candidate._cohort_schema_objects(db), before)

    def test_partial_schema_refuses_without_repair_or_removal(self):
        with self.store._connect() as db:
            db.execute(next(candidate._cohort_schema_statements()))
            before = candidate._cohort_schema_objects(db)
        with self.assertRaisesRegex(ValueError, 'Partial or changed'):
            self.enable()
        with self.store._connect() as db:
            self.assertEqual(candidate._cohort_schema_objects(db), before)

    def test_changed_immutable_trigger_refuses_without_overwrite(self):
        self.enable()
        with self.store._connect() as db:
            db.execute('DROP TRIGGER immutable_cohort_snapshot_source')
            db.execute("CREATE TRIGGER immutable_cohort_snapshot_source BEFORE UPDATE ON library_import_cohort_snapshot_sources BEGIN SELECT 1; END")
            before = candidate._cohort_schema_objects(db)
        with self.assertRaisesRegex(ValueError, 'Partial or changed'):
            self.enable()
        with self.store._connect() as db:
            self.assertEqual(candidate._cohort_schema_objects(db), before)

    def test_unknown_candidate_index_is_not_silently_adopted(self):
        self.enable()
        with self.store._connect() as db:
            db.execute('CREATE INDEX foreign_candidate_index ON library_import_cohort_bindings(album_id)')
            before = candidate._cohort_schema_objects(db)
        with self.assertRaisesRegex(ValueError, 'Partial or changed'):
            self.enable()
        with self.store._connect() as db:
            self.assertEqual(candidate._cohort_schema_objects(db), before)

    def test_changed_native_fk_tracking_trigger_refuses(self):
        self.enable()
        name = 'trg_fk_validation_library_import_cohort_members_insert'
        with self.store._connect() as db:
            db.execute(f'DROP TRIGGER {name}')
            db.execute(f'CREATE TRIGGER {name} AFTER INSERT ON library_import_cohort_members BEGIN SELECT 1; END')
            before = candidate._cohort_schema_objects(db)
        with self.assertRaisesRegex(ValueError, 'foreign-key validation trigger changed'):
            self.enable()
        with self.store._connect() as db:
            self.assertEqual(candidate._cohort_schema_objects(db), before)

    def test_installer_never_commits_outer_transaction(self):
        with self.store._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("INSERT INTO auth_users VALUES('outer-pending')")
            self.assertTrue(candidate.install_cohort_schema_candidate(db))
            self.assertTrue(db.in_transaction)
            db.rollback()
            self.assertEqual(candidate._cohort_schema_objects(db), {})
            self.assertIsNone(db.execute("SELECT id FROM auth_users WHERE id='outer-pending'").fetchone())

    def test_late_ddl_failure_rolls_back_all_candidate_objects_not_outer_work(self):
        statements = list(candidate._cohort_schema_statements()) + ['INVALID LATE DDL']
        with self.store._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("INSERT INTO auth_users VALUES('outer-preserved')")
            with patch.object(candidate, '_cohort_schema_statements', return_value=iter(statements)):
                with self.assertRaises(sqlite3.OperationalError):
                    candidate.install_cohort_schema_candidate(db)
            self.assertTrue(db.in_transaction)
            self.assertEqual(candidate._cohort_schema_objects(db), {})
            self.assertIsNotNone(db.execute("SELECT id FROM auth_users WHERE id='outer-preserved'").fetchone())
            db.commit()

    def test_missing_transaction_or_disabled_foreign_keys_refuses(self):
        with self.store._connect() as db:
            with self.assertRaisesRegex(ValueError, 'transaction with enforced foreign keys'):
                candidate.install_cohort_schema_candidate(db)
            db.execute('PRAGMA foreign_keys=OFF')
            db.execute('BEGIN')
            with self.assertRaisesRegex(ValueError, 'transaction with enforced foreign keys'):
                candidate.install_cohort_schema_candidate(db)
            self.assertEqual(candidate._cohort_schema_objects(db), {})

    def test_missing_native_parent_schema_refuses_without_creating_objects(self):
        with sqlite3.connect(':memory:') as db:
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('BEGIN')
            with self.assertRaisesRegex(ValueError, 'parents or foreign key integrity'):
                candidate.install_cohort_schema_candidate(db)
            self.assertEqual(candidate._cohort_schema_objects(db), {})

    def test_later_native_startup_failure_rolls_back_candidate_schema(self):
        with patch.object(NativeLibraryStore, '_ensure_foreign_key_validation_triggers', side_effect=RuntimeError('later native startup failure')):
            with self.assertRaisesRegex(RuntimeError, 'later native startup failure'):
                self.enable()
        with self.store._connect() as db:
            self.assertEqual(candidate._cohort_schema_objects(db), {})
            self.assertEqual(db.execute('SELECT id FROM auth_users').fetchone()[0], 'preserved-user')

    def test_existing_foreign_key_damage_is_preserved_and_refused_not_repaired(self):
        with self.store._connect() as db:
            db.execute('PRAGMA foreign_keys=OFF')
            db.execute("INSERT INTO local_album_aliases VALUES('orphan-fixture','missing-album','merged_album',1)")
            db.commit()
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('BEGIN')
            with self.assertRaisesRegex(ValueError, 'parents or foreign key integrity'):
                candidate.install_cohort_schema_candidate(db)
            self.assertEqual(candidate._cohort_schema_objects(db), {})
            self.assertIsNotNone(db.execute("SELECT 1 FROM local_album_aliases WHERE alias='orphan-fixture'").fetchone())


class CohortBoundUpgradeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module = importlib.import_module('tests.infrastructure.test_import_cohort_snapshot')
        self.fixture = module.CohortSnapshotCommitTests()
        await self.fixture.asyncSetUp()
        self.addCleanup(self.fixture.doCleanups)
        await self.fixture.atomic.commit()
        self.store = self.fixture.atomic.store
        self.path = self.fixture.atomic.fixture.root.parent / 'catalog.db'

    def rows(self):
        with self.store._connect() as db:
            return {table: [tuple(row) for row in db.execute(f'SELECT * FROM {table} ORDER BY rowid')]
                for table in ('local_albums', 'local_tracks', 'library_management_import_bundles',
                              'library_management_import_journal', 'library_management_metadata_snapshots',
                              'library_import_cohort_snapshot_sources', 'library_import_cohort_bindings',
                              'library_import_cohort_members', 'library_catalog_revision')}

    async def test_opt_in_native_restart_preserves_committed_binding_and_all_source_rows(self):
        before = self.rows()
        NativeLibraryStore(self.path, threading.Lock(), _cohort_schema_candidate_enabled=True)
        self.assertEqual(self.rows(), before)
        with self.store._connect() as db:
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])

    async def test_bound_snapshot_bundle_album_and_member_remain_delete_restricted_after_upgrade(self):
        NativeLibraryStore(self.path, threading.Lock(), _cohort_schema_candidate_enabled=True)
        before = self.rows()
        with self.store._connect() as db:
            for table in ('library_management_metadata_snapshots', 'library_management_import_bundles', 'local_albums', 'local_tracks'):
                with self.subTest(table=table), self.assertRaises(sqlite3.IntegrityError):
                    db.execute(f'DELETE FROM {table}')
                db.rollback()
        self.assertEqual(self.rows(), before)


class CohortSourceUpgradeTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_restart_retains_committed_provider_source_and_restricts_its_deletion(self):
        module = importlib.import_module('tests.services.native.test_import_cohort_provider')
        fixture = module.CohortProviderTests()
        await fixture.asyncSetUp()
        self.addCleanup(fixture.doCleanups)
        projection = await fixture.project()
        await fixture.correlate(projection)
        await fixture.atomic.commit(source_context=fixture.context)
        store = fixture.atomic.store
        path = fixture.atomic.fixture.root.parent / 'catalog.db'
        with store._connect() as db:
            before = [tuple(row) for row in db.execute('SELECT * FROM library_import_cohort_snapshot_sources')]
            self.assertEqual(len(before), 1)
        NativeLibraryStore(path, threading.Lock(), _cohort_schema_candidate_enabled=True)
        with store._connect() as db:
            self.assertEqual([tuple(row) for row in db.execute('SELECT * FROM library_import_cohort_snapshot_sources')], before)
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute('DELETE FROM library_import_cohort_snapshot_sources')
            db.rollback()
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])


class CohortLegacyBoundUpgradeTests(unittest.IsolatedAsyncioTestCase):
    async def test_v1_committed_snapshot_binding_is_preserved_and_readable_only_after_upgrade(self):
        atomic_module = importlib.import_module('tests.infrastructure.test_import_cohort_atomic_commit')
        snapshot_module = importlib.import_module('tests.infrastructure.test_import_cohort_snapshot')
        fixture = snapshot_module.CohortSnapshotCommitTests()
        with patch.object(atomic_module, 'CANDIDATE_SCHEMA_SQL', candidate.CANDIDATE_V1_SCHEMA_SQL):
            await fixture.asyncSetUp()
        self.addCleanup(fixture.doCleanups)
        track_ids = await fixture.atomic.commit()
        store = fixture.atomic.store
        path = fixture.atomic.fixture.root.parent / 'catalog.db'
        def read(db):
            return candidate.read_consistent_cohort(db, binding_id='binding', root_id='root',
                root_path=str(fixture.atomic.fixture.root), album_parent=fixture.atomic.candidate.album_parent,
                current_fingerprints={track_id: dict(fixture.atomic.candidate.fingerprints)[ordinal]
                    for (ordinal, _), track_id in zip(fixture.atomic.writes, track_ids)})
        with store._connect() as db:
            db.execute('BEGIN')
            before = [tuple(row) for row in db.execute('SELECT * FROM library_import_cohort_bindings')]
            self.assertIsNone(read(db))
        NativeLibraryStore(path, threading.Lock(), _cohort_schema_candidate_enabled=True)
        with store._connect() as db:
            db.execute('BEGIN')
            self.assertEqual([tuple(row) for row in db.execute('SELECT * FROM library_import_cohort_bindings')], before)
            self.assertIsNotNone(read(db))
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])
