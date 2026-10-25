"""Native snapshot correlation in real SQL transactions, not provider attestation.

Like atomic fixtures, commits isolate SQL mechanics with automatic_requests={};
full automatic baseline/metadata/source-generation acceptance is not exercised.
"""
from dataclasses import replace
import hashlib
import importlib
import json
import sqlite3
import unittest

import msgspec

from api.v1.schemas.library_management import LibraryManagementProfile, NamingScriptSettings
from models.audio_metadata import DesiredAudioDocument
from models.library_management import LibraryManagementMetadataSnapshot
from models.library_management_planning import PinnedLibraryManagementProfile, naming_policy_revision
from services.native.library_management_publisher import LibraryManagementPublisher
from infrastructure.persistence.import_cohort_binding import CohortSnapshotSeal, read_consistent_cohort


class CohortSnapshotCommitTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module = importlib.import_module('tests.infrastructure.test_import_cohort_atomic_commit')
        self.atomic = module.CohortAtomicCommitTests()
        await self.atomic.asyncSetUp()
        self.addCleanup(self.atomic.doCleanups)
        await self.install_snapshot()

    async def install_snapshot(self, document=None, *, identifier='snapshot', fetched=1, expires=100):
        document = document or self.atomic.fixture.canonical
        encoded = msgspec.json.encode(document).decode()
        input_hash = hashlib.sha256(identifier.encode()).hexdigest()
        stored = await self.atomic.store.put_management_metadata_snapshot(LibraryManagementMetadataSnapshot(
            id=identifier, provider='musicbrainz', entity_kind='release', entity_id=document.identifiers.release_mbid,
            input_hash=input_hash, canonical_payload_json=encoded, payload_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
            fetched_at=fetched, expires_at=expires))
        pinned = PinnedLibraryManagementProfile(
            profile=LibraryManagementProfile(id='fixture', name='Fixture', revision='profile'),
            naming_script=NamingScriptSettings(id='script', name='Fixture', source='%album%', revision='script-revision'))
        fixture = self.atomic.fixture
        fixture.bundle = msgspec.structs.replace(fixture.bundle, files=tuple(msgspec.structs.replace(
            value, metadata_snapshot_id=stored.id, pinned_profile=pinned,
            desired_document=DesiredAudioDocument(fields=()), projection_hash='b'*64,
            settings_revision='settings', naming_policy_revision=naming_policy_revision(pinned),
            undo_retention_days=7, baseline_relative_path=value.destination_relative_path)
            for value in fixture.bundle.files))
        # The fixtures preserve the original native publication shape guards.
        LibraryManagementPublisher._validate_import_bundle(fixture.bundle)
        encoded_request = msgspec.json.encode(fixture.bundle).decode()
        with self.atomic.store._connect() as db:
            db.execute('UPDATE library_management_import_bundles SET request_json=?,request_hash=?',
                       (encoded_request, hashlib.sha256(encoded_request.encode()).hexdigest()))
        self.atomic.candidate = replace(self.atomic.candidate, snapshot=CohortSnapshotSeal(
            snapshot_id=stored.id, input_hash=stored.input_hash, payload_sha256=stored.payload_sha256))

    async def test_native_stored_snapshot_seal_persists_with_binding(self):
        track_ids = await self.atomic.commit()
        with self.atomic.store._connect() as db:
            db.execute('BEGIN')
            row = db.execute('SELECT metadata_snapshot_id,payload_json FROM library_import_cohort_bindings').fetchone()
            self.assertEqual(row[0], 'snapshot')
            self.assertEqual(json.loads(row[1])['native_snapshot']['payload_sha256'], self.atomic.candidate.snapshot.payload_sha256)
            # Fixture clock at commit is2; wall clock is long past expiry100.
            # Committed membership is not discarded at snapshot TTL expiry.
            payload = read_consistent_cohort(db, binding_id='binding', root_id='root',
                root_path=str(self.atomic.fixture.root), album_parent=self.atomic.candidate.album_parent,
                current_fingerprints={track_id: dict(self.atomic.candidate.fingerprints)[ordinal]
                    for (ordinal, _), track_id in zip(self.atomic.writes, track_ids)})
            self.assertIsNotNone(payload)

    async def test_snapshot_reference_cannot_be_deleted_after_commit(self):
        await self.atomic.commit()
        with self.atomic.store._connect() as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("DELETE FROM library_management_metadata_snapshots WHERE id='snapshot'")

    async def test_wrong_snapshot_hash_rolls_back_native_adoption(self):
        seal = replace(self.atomic.candidate.snapshot, payload_sha256='f'*64)
        with self.assertRaisesRegex(ValueError, 'Native snapshot seal'):
            await self.atomic.commit(_cohort_candidate=replace(self.atomic.candidate, snapshot=seal))
        self.atomic.assert_uncommitted()

    async def test_missing_snapshot_rolls_back_native_adoption(self):
        seal = replace(self.atomic.candidate.snapshot, snapshot_id='missing')
        with self.assertRaisesRegex(ValueError, 'Native snapshot seal'):
            await self.atomic.commit(_cohort_candidate=replace(self.atomic.candidate, snapshot=seal))
        self.atomic.assert_uncommitted()

    async def test_expired_future_or_unbounded_snapshot_rejected(self):
        for identifier, fetched, expires in [('expired', 1, 2), ('future', 3, 100), ('unbounded', 1, None)]:
            with self.subTest(identifier=identifier):
                await self.install_snapshot(identifier=identifier, fetched=fetched, expires=expires)
                with self.assertRaisesRegex(ValueError, 'stale or has invalid clock'):
                    await self.atomic.commit()
                self.atomic.assert_uncommitted()

    async def test_partial_provider_projection_cannot_attest_complete_cohort(self):
        document = self.atomic.fixture.canonical
        partial = msgspec.structs.replace(document, media=(
            msgspec.structs.replace(document.media[0], track_count=2), document.media[1]))
        await self.install_snapshot(partial, identifier='partial')
        with self.assertRaisesRegex(ValueError, 'not a complete medium'):
            await self.atomic.commit()
        self.atomic.assert_uncommitted()

    async def test_different_native_recording_map_cannot_attest_request(self):
        document = self.atomic.fixture.canonical
        first = document.media[0]
        track = msgspec.structs.replace(first.tracks[0], identifiers=msgspec.structs.replace(
            first.tracks[0].identifiers, recording_mbid='different-recording'))
        different = msgspec.structs.replace(document, media=(msgspec.structs.replace(first, tracks=(track,)), document.media[1]))
        await self.install_snapshot(different, identifier='different-map')
        with self.assertRaisesRegex(ValueError, 'recording map differs'):
            await self.atomic.commit()
        self.atomic.assert_uncommitted()

    async def test_native_snapshot_immutability_remains_enforced(self):
        with self.atomic.store._connect() as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("UPDATE library_management_metadata_snapshots SET payload_sha256=? WHERE id='snapshot'", ('f'*64,))
