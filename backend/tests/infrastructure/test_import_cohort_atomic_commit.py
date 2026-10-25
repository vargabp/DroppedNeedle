"""Actual native transaction tests against disposable migrated databases only.

No automatic metadata acceptance/source-provider integration is claimed: these
requests use the native unpinned acquisition path to isolate SQL atomicity.
"""
import hashlib
import importlib
import sqlite3
import threading
import unittest

import msgspec

from infrastructure.persistence.import_cohort_binding import (
    CANDIDATE_SCHEMA_SQL, ImportCohortCommitCandidate, read_consistent_cohort,
)
from core.exceptions import StaleRevisionError, ValidationError, LibraryManagementPolicyChangedError, ConfigurationError
from infrastructure.persistence.download_store import DownloadStore


class CohortAtomicCommitTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module = importlib.import_module('tests.services.native.test_import_cohort_preparation')
        self.fixture = module.CohortPreparationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.store = self.fixture.store
        # Held acquisitions are owned by the native DownloadStore migrations,
        # not NativeLibraryStore. Initialize their real schema too.
        DownloadStore(self.fixture.root.parent / 'catalog.db', threading.Lock())
        self.writes, digests, parent = await self.fixture.prepare()
        encoded = msgspec.json.encode(self.fixture.bundle).decode()
        policy = self.fixture.bundle.policy_revision
        with self.store._connect() as db:
            db.executescript(CANDIDATE_SCHEMA_SQL)
            db.execute("INSERT INTO library_policy_state(singleton,desired_policy_revision,updated_at) VALUES(1,?,1) ON CONFLICT(singleton) DO UPDATE SET desired_policy_revision=excluded.desired_policy_revision", (policy,))
            db.execute("INSERT INTO auth_users(id) VALUES('fixture-user')")
            db.execute("INSERT INTO library_management_import_bundles(id,idempotency_key,origin,policy_revision,request_json,request_hash,state,created_at,updated_at) VALUES('bundle','candidate','acquisition',?,?,?,'publishing',1,1)",
                       (policy, encoded, hashlib.sha256(encoded.encode()).hexdigest()))
            for value in self.fixture.published:
                request = value.request
                db.execute("INSERT INTO library_management_import_journal(bundle_id,ordinal,state,source_fingerprint,source_size,source_mtime_ns,temporary_relative_path,destination_root_id,destination_relative_path,staged_fingerprint,created_at,updated_at) VALUES('bundle',?,'published',?,31,1,'temporary','root',?,?,1,1)",
                           (request.ordinal, 'a'*64, request.destination_relative_path, value.staged_fingerprint))
                db.execute("INSERT INTO held_imports(user_id,held_path,reason,source_task_id,status,created_at) VALUES('fixture-user',?,'management:FIXTURE','task','held',1)", (request.source_path,))
        self.candidate = ImportCohortCommitCandidate(
            binding_id='binding', root_id='root', root_path=str(self.fixture.root), album_parent=parent,
            canonical_map=tuple((v.release_track_mbid, v.recording_mbid, v.medium_position, v.release_track_position)
                                for v in self.fixture.bundle.files),
            fingerprints=tuple((int(key), digest) for key, digest in digests.items()))

    async def commit(self, **overrides):
        arguments = dict(writes=self.writes, replacement_track_ids={}, recycle_track_ids={},
                         requests_by_ordinal={v.ordinal: v for v in self.fixture.bundle.files},
                         automatic_requests={}, expected_policy_revision=self.fixture.bundle.policy_revision,
                         result_paths_by_ordinal={v.request.ordinal: v.destination_path for v in self.fixture.published},
                         updated_at=2, _cohort_candidate=self.candidate)
        arguments.update(overrides)
        return await self.store.commit_library_management_import_bundle('bundle', **arguments)

    def assert_uncommitted(self):
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM local_tracks').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM local_albums').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_import_cohort_bindings').fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT state FROM library_management_import_bundles WHERE id='bundle'").fetchone()[0], 'publishing')
            self.assertEqual([row[0] for row in db.execute('SELECT state FROM library_management_import_journal')], ['published', 'published'])
            self.assertEqual([row[0] for row in db.execute('SELECT status FROM held_imports')], ['held', 'held'])

    async def test_actual_store_commits_tracks_binding_and_journals_together(self):
        identifiers = await self.commit()
        self.assertEqual(len(identifiers), 2)
        with self.store._connect() as db:
            db.execute('BEGIN')
            payload = read_consistent_cohort(db, binding_id='binding', root_id='root',
                root_path=str(self.fixture.root), album_parent=self.candidate.album_parent,
                current_fingerprints={track_id: dict(self.candidate.fingerprints)[ordinal]
                                      for (ordinal, _), track_id in zip(self.writes, identifiers)})
            self.assertIsNotNone(payload)
            self.assertEqual(db.execute('SELECT COUNT(DISTINCT local_album_id) FROM local_tracks').fetchone()[0], 1)
            self.assertEqual([row[0] for row in db.execute('SELECT state FROM library_management_import_journal')], ['catalog_committed', 'catalog_committed'])
            self.assertEqual([row[0] for row in db.execute('SELECT status FROM held_imports')], ['imported', 'imported'])

    async def test_late_native_bundle_failure_rolls_back_binding_and_adoption(self):
        with self.store._connect() as db:
            before = db.execute('SELECT value FROM library_catalog_revision').fetchone()[0]
            db.execute("CREATE TRIGGER late_failure BEFORE UPDATE OF state ON library_management_import_bundles WHEN NEW.state='catalog_committed' BEGIN SELECT RAISE(ABORT,'late native fixture failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            await self.commit()
        self.assert_uncommitted()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT value FROM library_catalog_revision').fetchone()[0], before)

    async def test_partial_binding_insert_failure_rolls_back_whole_native_commit(self):
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER member_failure BEFORE INSERT ON library_import_cohort_members WHEN NEW.ordinal=1 BEGIN SELECT RAISE(ABORT,'member fixture failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            await self.commit()
        self.assert_uncommitted()

    async def test_incomplete_fingerprints_roll_back_adoption(self):
        from dataclasses import replace
        with self.assertRaises(ValidationError):
            await self.commit(_cohort_candidate=replace(self.candidate, fingerprints=self.candidate.fingerprints[:1]))
        self.assert_uncommitted()

    async def test_wrong_policy_still_fails_before_adoption(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_policy_state SET desired_policy_revision='changed'")
        with self.assertRaises(LibraryManagementPolicyChangedError):
            await self.commit()
        self.assert_uncommitted()

    async def test_candidate_replay_is_refused_without_fresh_native_validation(self):
        await self.commit()
        with self.assertRaises(StaleRevisionError):
            await self.commit()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_import_cohort_bindings').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM local_tracks').fetchone()[0], 2)

    async def test_default_native_path_requires_no_candidate_schema_and_keeps_idempotency(self):
        with self.store._connect() as db:
            db.executescript('DROP TABLE library_import_cohort_members; DROP TABLE library_import_cohort_bindings;')
        first = await self.commit(_cohort_candidate=None)
        self.assertEqual(first, await self.commit(_cohort_candidate=None))

    async def test_candidate_never_creates_missing_schema_implicitly(self):
        with self.store._connect() as db:
            db.executescript('DROP TABLE library_import_cohort_members; DROP TABLE library_import_cohort_bindings;')
        with self.assertRaises(sqlite3.OperationalError):
            await self.commit()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM local_tracks').fetchone()[0], 0)
            self.assertEqual([row[0] for row in db.execute('SELECT status FROM held_imports')], ['held', 'held'])

    async def test_mutable_candidate_inputs_and_duplicate_observations_rejected(self):
        from dataclasses import replace, FrozenInstanceError
        with self.assertRaises(ValueError):
            replace(self.candidate, canonical_map=(('rt', ['mutable'], 1, 1),))
        with self.assertRaises(ValueError):
            replace(self.candidate, fingerprints=(self.candidate.fingerprints[0],) * 2)
        with self.assertRaises(FrozenInstanceError):
            self.candidate.root_id = 'other'

    async def test_automatic_source_context_guard_is_not_bypassed(self):
        with self.assertRaises(ConfigurationError):
            await self.commit(automatic_requests={v.ordinal: v for v in self.fixture.bundle.files})
        self.assert_uncommitted()

    async def test_split_album_candidate_is_refused_and_rolled_back(self):
        original = []
        for value in self.fixture.published:
            request = value.request
            write = await self.fixture.service._build_published_import_write(
                self.fixture.root / request.destination_relative_path, value.tag, value.info,
                resolver=self.fixture.resolver, release_group_mbid=request.release_group_mbid,
                release_mbid=request.release_mbid, recording_mbid=request.recording_mbid,
                source=request.source, download_task_id=request.download_task_id,
                source_path=request.source_path, file_mtime=None)
            original.append((request.ordinal, write))
        self.assertEqual(len({write.album.id for _, write in original}), 2)
        with self.assertRaises(ValidationError):
            await self.commit(writes=original)
        self.assert_uncommitted()
