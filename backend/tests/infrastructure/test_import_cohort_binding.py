"""Candidate persistence tests using native DDL and disposable SQLite only.

Not end-to-end publisher/indexer/grouping integration or production migration.
"""
import hashlib
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from infrastructure.persistence.native_library_store import NativeLibraryStore
from infrastructure.persistence.import_cohort_binding import (
    CANDIDATE_SCHEMA_SQL, persist_import_cohort, read_consistent_cohort,
)


class ImportCohortBindingTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        path=Path(self.directory.name)/'library.db'
        with sqlite3.connect(path) as seed:
            seed.execute('CREATE TABLE auth_users(id TEXT PRIMARY KEY)')
        self.store=NativeLibraryStore(path,threading.Lock())
        self.db=self.store._connect()
        self.assertIsNone(self.db.execute("SELECT name FROM sqlite_master WHERE name='library_import_cohort_bindings'").fetchone())
        self.db.executescript(CANDIDATE_SCHEMA_SQL)
        self.db.execute("INSERT INTO local_artists(id,display_name,folded_name,kind,created_at,updated_at) VALUES('artist','Artist','artist','group',1,1)")
        self.db.execute("INSERT INTO local_albums(id,root_id,grouping_key,title,title_folded,album_artist_id,grouping_source,created_at,updated_at) VALUES('album','root','Artist/Album','Album','album','artist','automatic',1,1)")
        self.db.execute("INSERT INTO library_policy_state(singleton,desired_policy_revision,updated_at) VALUES(1,'p1',1) ON CONFLICT(singleton) DO UPDATE SET desired_policy_revision='p1'")
        self.files=[];self.members={};self.canonical=[];self.fingerprints={}
        for i in (1,2):
            track_id=f't{i}';relative=f'Artist/Album/Digital Media {i:02}/01.flac';digest=str(i)*64
            file=dict(ordinal=i-1,destination_root_id='root',destination_relative_path=relative,
                      release_track_mbid=f'rt{i}',recording_mbid=f'rec{i}',medium_position=i,
                      release_track_position=1,release_mbid='release',release_group_mbid='group',
                      authoritative_mapping=True,download_task_id='source')
            self.files.append(file);self.members[i-1]=(track_id,digest);self.fingerprints[track_id]=digest
            self.canonical.append((f'rt{i}',f'rec{i}',i,1))
            fields=dict(id=track_id,local_album_id='album',root_id='root',file_path='/storage/library/'+relative,
                        relative_path=relative,path_hash=digest,file_size_bytes=10,file_mtime_ns=20,
                        stat_revision='s1',stat_revision_kind='exact',tag_revision='tags1',title='Title',title_folded='title',
                        album_title='Album',album_title_folded='album',album_artist_name='Artist',
                        tag_album_title='Album',tag_album_artist_name='Artist',
                        file_format='flac',ingest_source='download',imported_at=1,membership_source='automatic',
                        embedded_release_mbid='release',embedded_release_group_mbid='group',
                        embedded_recording_mbid=f'rec{i}',embedded_release_track_mbid=f'rt{i}',
                        disc_number=i,track_number=1,download_task_id='source')
            self.db.execute('INSERT INTO local_tracks('+','.join(fields)+') VALUES('+','.join('?' for _ in fields)+')',tuple(fields.values()))
        encoded=json.dumps(dict(origin='acquisition',policy_revision='p1',files=self.files))
        self.db.execute("INSERT INTO library_management_import_bundles(id,idempotency_key,origin,policy_revision,request_json,request_hash,state,created_at,updated_at) VALUES('bundle','key','acquisition','p1',?,?,'publishing',1,1)",(encoded,hashlib.sha256(encoded.encode()).hexdigest()))
        for file in self.files:
            self.db.execute("INSERT INTO library_management_import_journal(bundle_id,ordinal,state,source_fingerprint,source_size,source_mtime_ns,temporary_relative_path,destination_root_id,destination_relative_path,staged_fingerprint,created_at,updated_at) VALUES('bundle',?,'published',?,10,20,'temporary','root',?,?,1,1)",
                            (file['ordinal'],'a'*64,file['destination_relative_path'],self.members[file['ordinal']][1]))
        self.db.commit()

    def tearDown(self):self.db.close();self.directory.cleanup()

    def persist(self,**changes):
        args=dict(binding_id='binding',bundle_id='bundle',album_id='album',root_id='root',
                  root_path='/storage/library',album_parent='Artist/Album',canonical_map=tuple(self.canonical),members=self.members)
        args.update(changes);persist_import_cohort(self.db,**args)

    def committed(self):
        self.db.execute('BEGIN');self.persist()
        self.db.execute("UPDATE library_management_import_bundles SET state='catalog_committed'")
        self.db.execute("UPDATE library_management_import_journal SET state='catalog_committed'")
        self.db.commit()

    def read(self,**changes):
        args=dict(binding_id='binding',root_id='root',root_path='/storage/library',album_parent='Artist/Album',current_fingerprints=self.fingerprints)
        args.update(changes)
        owned=not self.db.in_transaction
        if owned:self.db.execute('BEGIN')
        try:return read_consistent_cohort(self.db,**args)
        finally:
            if owned:self.db.rollback()

    def test_whole_binding_survives_committed_catalog_and_reordered_observations(self):
        self.committed();result=self.read(current_fingerprints=dict(reversed(list(self.fingerprints.items()))))
        self.assertEqual(len(result['members']),2)

    def test_does_not_commit_caller_transaction(self):
        self.db.execute('BEGIN');self.persist();self.assertTrue(self.db.in_transaction)
        self.db.rollback();self.assertEqual(self.db.execute('SELECT COUNT(*) FROM library_import_cohort_bindings').fetchone()[0],0)

    def test_requires_outer_transaction(self):
        with self.assertRaises(ValueError):self.persist()

    def test_reader_requires_consistent_native_snapshot(self):
        self.committed()
        with self.assertRaises(ValueError):
            read_consistent_cohort(self.db,binding_id='binding',root_id='root',root_path='/storage/library',album_parent='Artist/Album',current_fingerprints=self.fingerprints)

    def test_conflicting_tag_context_cannot_create_binding(self):
        self.db.execute("UPDATE local_tracks SET tag_album_title='Other album' WHERE id='t2'")
        with self.assertRaises(ValueError):self.persist()

    def test_sql_failure_rolls_back_only_binding_slice(self):
        self.db.execute("CREATE TRIGGER fail_member BEFORE INSERT ON library_import_cohort_members WHEN NEW.ordinal=1 BEGIN SELECT RAISE(ABORT,'fixture'); END")
        self.db.execute('BEGIN');self.db.execute("UPDATE local_albums SET title='caller update'")
        with self.assertRaises(sqlite3.IntegrityError):self.persist()
        self.assertTrue(self.db.in_transaction)
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM library_import_cohort_bindings').fetchone()[0],0)
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM library_import_cohort_members').fetchone()[0],0)
        self.assertEqual(self.db.execute('SELECT title FROM local_albums').fetchone()[0],'caller update')

    def test_rolled_back_bundle_cannot_create_binding(self):
        self.db.execute("UPDATE library_management_import_bundles SET state='rolled_back'")
        with self.assertRaises(ValueError):self.persist()

    def test_catalog_map_must_be_complete(self):
        self.db.execute('BEGIN')
        with self.assertRaises(ValueError):self.persist(canonical_map=tuple(self.canonical[:1]))

    def test_duplicate_catalog_position_rejected(self):
        self.db.execute('BEGIN')
        with self.assertRaises(ValueError):self.persist(canonical_map=(self.canonical[0],('rt2','rec2',1,1)))

    def test_root_parent_scope_changes_never_reuse(self):
        self.committed()
        for key,value in [('root_id','other'),('root_path','/other'),('album_parent','Artist/Elsewhere')]:
            with self.subTest(field=key):self.assertIsNone(self.read(**{key:value}))

    def test_partial_page_cannot_endorse_whole_cohort(self):
        self.committed();self.assertIsNone(self.read(current_fingerprints={'t1':'1'*64}))

    def test_content_change_even_with_unchanged_stat_rejected(self):
        self.committed();self.assertIsNone(self.read(current_fingerprints={'t1':'a'*64,'t2':'2'*64}))

    def test_catalog_mutations_invalidate_without_manual_lock_override(self):
        self.committed()
        for field,value in [('root_id','other'),('relative_path','moved.flac'),('stat_revision','s2'),
                            ('tag_revision','tags2'),('download_task_id','other'),('membership_locked',1),
                            ('embedded_recording_mbid','other'),('embedded_release_mbid','other'),('availability','missing')]:
            with self.subTest(field=field):
                self.db.execute('SAVEPOINT mutation')
                self.db.execute(f'UPDATE local_tracks SET {field}=? WHERE id=?',(value,'t1'))
                self.assertIsNone(self.read());self.db.execute('ROLLBACK TO mutation');self.db.execute('RELEASE mutation')

    def test_policy_and_journal_mutations_invalidate(self):
        self.committed()
        for sql in ["UPDATE library_policy_state SET desired_policy_revision='p2'",
                    "UPDATE library_management_import_journal SET state='rolled_back'",
                    "UPDATE library_management_import_journal SET staged_fingerprint='"+'a'*64+"'",
                    "UPDATE library_management_import_bundles SET state='needs_attention'"]:
            self.db.execute('SAVEPOINT mutation');self.db.execute(sql)
            self.assertIsNone(self.read());self.db.execute('ROLLBACK TO mutation');self.db.execute('RELEASE mutation')

    def test_album_manual_lock_and_legacy_stat_are_not_overridden(self):
        self.committed()
        for sql in ["UPDATE local_albums SET grouping_locked=1",
                    "UPDATE local_albums SET grouping_source='manual'",
                    "UPDATE local_tracks SET stat_revision_kind='legacy_float'",
                    "UPDATE local_tracks SET metadata_incomplete=1"]:
            self.db.execute('SAVEPOINT mutation');self.db.execute(sql)
            self.assertIsNone(self.read());self.db.execute('ROLLBACK TO mutation');self.db.execute('RELEASE mutation')

    def test_bound_catalog_track_cannot_be_deleted(self):
        self.committed()
        with self.assertRaises(sqlite3.IntegrityError):self.db.execute("DELETE FROM local_tracks WHERE id='t1'")

    def test_double_publication_does_not_overwrite(self):
        self.db.execute('BEGIN');self.persist()
        with self.assertRaises(sqlite3.IntegrityError):self.persist()
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM library_import_cohort_bindings').fetchone()[0],1)


if __name__=='__main__':unittest.main()
