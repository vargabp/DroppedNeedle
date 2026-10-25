"""Read-only inspector over native DDL and private structural reference fixtures."""
import sqlite3
import unittest
from infrastructure.persistence.import_cohort_retention import inspect_cohort_retention
from tests.infrastructure import test_import_cohort_binding as binding_fixture


class ImportCohortRetentionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=binding_fixture.ImportCohortBindingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.db=self.fixture.db

    def inspect(self, **kwargs):
        self.db.execute('PRAGMA query_only=ON')
        self.db.execute('BEGIN')
        before=self.db.total_changes
        statements=[]
        self.db.set_trace_callback(statements.append)
        try:
            result=inspect_cohort_retention(self.db,**kwargs)
            self.assertTrue(self.db.in_transaction)
            self.assertEqual(self.db.total_changes,before)
            self.assertTrue(all(s.lstrip().upper().startswith(('SELECT','PRAGMA')) for s in statements))
            self.assertFalse(result['execution_enabled'])
            self.assertFalse(result['safe_to_release'])
            self.assertFalse(result['eligibility_proven'])
            return result
        finally:
            self.db.set_trace_callback(None)
            self.db.rollback()
            self.db.execute('PRAGMA query_only=OFF')

    def seed_proof(self):
        self.db.execute("INSERT INTO library_management_metadata_snapshots(id,provider,entity_kind,entity_id,input_hash,canonical_payload_json,payload_sha256,fetched_at,expires_at) VALUES('snapshot','musicbrainz','release','release',?,'{}',?,1,2)",('a'*64,'b'*64))
        self.db.execute("INSERT INTO library_import_cohort_snapshot_sources VALUES('proof','snapshot','fixture','source',1,?,?,?,1)",('c'*64,'a'*64,'b'*64))
        self.db.commit()

    def invalidate(self):
        self.db.execute("INSERT INTO library_import_cohort_policy_invalidations SELECT id,payload_sha256,'p1','p2','POLICY_BOUNDARY_CHANGED',10 FROM library_import_cohort_bindings")
        self.db.commit()

    def test_bound_is_not_eligibility_proof_and_does_not_mutate(self):
        self.fixture.committed()
        result=self.inspect(binding_id='binding')
        self.assertEqual(result['state'],'BOUND_WITHOUT_POLICY_INVALIDATION_NOT_ELIGIBILITY_PROOF')
        self.assertEqual(result['member_reference_count'],2)
        self.assertEqual(result['album_id'],'album')

    def test_invalidation_retains_references_and_never_authorizes_release(self):
        self.fixture.committed()
        self.invalidate()
        result=self.inspect(binding_id='binding')
        self.assertEqual(result['state'],'BOUND_WITH_POLICY_INVALIDATION_RETAINS_REFERENCES')
        self.assertEqual(result['member_reference_count'],2)
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("DELETE FROM local_tracks WHERE id='t1'")
        self.db.rollback()

    def test_unbound_proof_and_expired_snapshot_are_retained_not_garbage(self):
        self.seed_proof()
        result=self.inspect(source_proof_id='proof')
        self.assertEqual(result['state'],'UNBOUND_SOURCE_PROOF_RETAINED')
        self.assertEqual(result['snapshot_reference_counts'],dict(cohort_bindings=0,source_proofs=1,native_jobs=0))
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute("DELETE FROM library_management_metadata_snapshots WHERE id='snapshot'")
        self.db.rollback()

    def test_shared_source_stays_bound_after_invalidation(self):
        self.seed_proof()
        self.fixture.committed()
        # Structural FK graph fixture only; not provider/recording-map proof.
        self.db.execute("UPDATE library_import_cohort_bindings SET metadata_snapshot_id='snapshot',source_proof_id='proof'")
        self.db.commit()
        self.invalidate()
        result=self.inspect(source_proof_id='proof')
        self.assertEqual(result['state'],'BOUND_SOURCE_PROOF_RETAINED')
        self.assertEqual(result['binding_reference_count'],1)
        self.assertEqual(result['invalidated_binding_reference_count'],1)
        self.assertEqual(result['snapshot_reference_counts']['cohort_bindings'],1)

    def test_missing_ids_are_not_release_authority(self):
        self.assertEqual(self.inspect(binding_id='absent')['state'],'BINDING_NOT_FOUND')
        self.assertEqual(self.inspect(source_proof_id='absent')['state'],'SOURCE_PROOF_NOT_FOUND')

    def test_requires_caller_read_transaction_and_exact_identifier(self):
        with self.assertRaises(ValueError):
            inspect_cohort_retention(self.db,binding_id='binding')
        self.db.execute('BEGIN')
        with self.assertRaises(ValueError):
            inspect_cohort_retention(self.db,binding_id='binding')
        self.db.rollback()
        self.db.execute('PRAGMA query_only=ON')
        self.db.execute('BEGIN')
        for arguments in ({},{'binding_id':'a','source_proof_id':'b'},{'binding_id':''},{'binding_id':1}):
            with self.assertRaises(ValueError):inspect_cohort_retention(self.db,**arguments)
        self.db.rollback()
        self.db.execute('PRAGMA query_only=OFF')

    def test_absent_schema_is_reported_without_installing(self):
        with sqlite3.connect(':memory:') as connection:
            connection.execute('PRAGMA query_only=ON')
            connection.execute('BEGIN')
            result=inspect_cohort_retention(connection,binding_id='binding')
            self.assertEqual(result['state'],'PRIVATE_SCHEMA_NOT_INSTALLED')
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM sqlite_master').fetchone()[0],0)

    def test_partial_schema_is_refused_without_repair(self):
        self.db.execute('DROP TRIGGER immutable_cohort_policy_invalidation_delete')
        self.db.commit()
        with self.assertRaisesRegex(ValueError,'Exact private lifecycle schema'):
            self.inspect(binding_id='binding')
        self.assertIsNone(self.db.execute("SELECT name FROM sqlite_master WHERE name='immutable_cohort_policy_invalidation_delete'").fetchone())
