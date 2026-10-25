"""Actual native staged publisher on generated FLAC; disposable storage only.

Native provider I/O is simulated by the shared provider fixture. No live media,
production settings, startup migration or scan/indexer rollout is exercised.
"""
import asyncio
import hashlib
import importlib
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import msgspec
from mutagen.flac import FLAC

from api.v1.schemas.library_management import (
    LibraryManagementSettings, LibraryManagementRootAssignment, NamingScriptSettings,
    profile_revision, settings_revision,
)
from core.exceptions import AutomaticManagementHoldError, StaleRevisionError
from infrastructure.audio.metadata_engine import AudioMetadataEngine, legacy_audio_projection
from infrastructure.library_management_blob_store import LibraryManagementBlobStore
from models.audio_metadata import DesiredAudioDocument, DesiredAudioField, SemanticTagSnapshot
from models.library_management_planning import pin_library_management_profile, naming_policy_revision
from services.native.audio_write_planning_service import AudioWritePlanningService
from services.native.library_management_publisher import LibraryManagementPublisher


class CohortStagedTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module = importlib.import_module('tests.services.native.test_import_cohort_policy')
        self.base = module.CohortPolicyTests()
        await self.base.asyncSetUp()
        self.addCleanup(self.base.doCleanups)
        await self.base.correlate()
        self.fixture = self.base.atomic.fixture
        self.store = self.base.atomic.store
        self.service = self.base.service
        self.audio = AudioMetadataEngine()
        # Remove ONLY shared disposable fixture's artificial already-published
        # ledger and 31-byte stand-ins: the real publisher must create its own.
        with self.store._connect() as db:
            db.execute('DELETE FROM library_management_import_journal')
            db.execute('DELETE FROM library_management_import_bundles')
        for value in self.fixture.published:
            Path(value.destination_path).unlink()
        profile = self.fixture.bundle.files[0].pinned_profile.profile
        profile.organization.naming_script_id = 'script'
        profile.revision = profile_revision(profile)
        self.settings = LibraryManagementSettings(profiles=[profile], default_profile_id=profile.id,
            naming_scripts=[NamingScriptSettings(id='script', name='Fixture', source='%album%', revision='script-revision')])
        pinned = pin_library_management_profile(self.settings, profile)
        self.settings.root_assignments = [LibraryManagementRootAssignment(
            root_id='root', profile_id=profile.id, enabled=True, automatic_acquisitions=True,
            activation_profile_revision=profile.revision,
            activation_policy_revision=self.fixture.bundle.policy_revision,
            activation_naming_policy_revision=naming_policy_revision(pinned),
            activation_preview_token='fixture-confirmed', activation_preview_hash='fixture-confirmed',
            activation_confirmed_at=1)]
        self.sources, self.before_hashes, self.pcm_hashes, self.before_snapshots = [], [], [], []
        requests = []
        for request in self.fixture.bundle.files:
            path = self.fixture.root.parent / f'incoming-{request.ordinal}.flac'
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                'sine=frequency=440:duration=0.15', '-c:a', 'flac', str(path)], check=True, capture_output=True)
            tags = FLAC(path)
            for key, value in dict(title='Original Track', artist='Artist', albumartist='Artist', album='Album',
                discnumber=str(request.medium_position), tracknumber='1', musicbrainz_albumid='release',
                musicbrainz_releasegroupid='group', musicbrainz_trackid=request.recording_mbid,
                musicbrainz_releasetrackid=request.release_track_mbid).items():
                tags[key] = value
            tags.save()
            tag, info = legacy_audio_projection(self.audio.read(path))
            self.sources.append(path)
            self.before_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            self.pcm_hashes.append(self.pcm_digest(path))
            self.before_snapshots.append(self.audio.snapshot(path))
            requests.append(msgspec.structs.replace(request, input_path=str(path), tag=tag, info=info,
                pinned_profile=pinned, settings_revision=settings_revision(self.settings),
                naming_policy_revision=naming_policy_revision(pinned),
                desired_document=DesiredAudioDocument(fields=(DesiredAudioField(
                    name='title', action='set', value='Managed Track'),))))
        self.bundle = msgspec.json.decode(msgspec.json.encode(msgspec.structs.replace(
            self.fixture.bundle, idempotency_key='real-staged-fixture', files=tuple(requests))),
            type=type(self.fixture.bundle))
        self.blobs = LibraryManagementBlobStore(self.fixture.root.parent / 'blobs', self.store)
        preferences = SimpleNamespace(get_library_management_settings_raw=lambda: self.settings,
            get_typed_library_settings_raw=lambda: self.fixture.resolver.settings)
        self.publisher = LibraryManagementPublisher(self.store, preferences, self.audio,
            AudioWritePlanningService(self.audio), self.blobs, self.service._filesystem, clock=lambda: 2)
        self.service._management_publisher = self.publisher

    @staticmethod
    def pcm_digest(path):
        decoded = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path),
            '-f', 's16le', '-acodec', 'pcm_s16le', '-'], check=True, capture_output=True)
        return hashlib.sha256(decoded.stdout).hexdigest()

    async def publish(self):
        module = importlib.import_module('services.native.target_import_library_service')
        with patch.object(module, 'time', SimpleNamespace(time=lambda: 2)):
            return await asyncio.wait_for(self.service.publish_import_bundle(self.bundle), 10)

    async def test_native_staging_publishes_audio_baselines_history_and_one_cohort(self):
        result = await self.publish()
        self.assertEqual(len(result.local_track_ids), 2)
        for index, request in enumerate(self.bundle.files):
            destination = self.fixture.root / request.destination_relative_path
            self.assertEqual(self.pcm_digest(destination), self.pcm_hashes[index])
            self.assertEqual(legacy_audio_projection(self.audio.read(destination))[0].title, 'Managed Track')
            # Native successful cleanup is allowed ONLY inside this disposable fixture.
            self.assertFalse(self.sources[index].exists())
            baseline = await self.store.get_management_baseline(result.local_track_ids[index])
            encoded = await self.blobs.read_bytes(baseline.semantic_snapshot_blob_sha256)
            snapshot = msgspec.json.decode(encoded, type=SemanticTagSnapshot)
            self.assertEqual(snapshot.probe.detected_format, 'flac')
            self.assertNotEqual(baseline.adapter_version, 'fixture-only')
            self.assertEqual(snapshot, self.before_snapshots[index])
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(DISTINCT local_album_id) FROM local_tracks').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_import_cohort_bindings').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT state FROM library_management_import_bundles').fetchone()[0], 'completed')
            self.assertEqual([row[0] for row in db.execute('SELECT state FROM library_management_import_journal')], ['completed', 'completed'])
            self.assertEqual(db.execute('SELECT state FROM library_operation_jobs').fetchone()[0], 'succeeded')
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])

    async def test_late_catalog_failure_removes_published_files_and_preserves_sources(self):
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER late_failure BEFORE UPDATE OF state ON library_management_import_bundles WHEN NEW.state='catalog_committed' BEGIN SELECT RAISE(ABORT,'late staged fixture failure'); END")
        with self.assertRaises(AutomaticManagementHoldError) as caught:
            await self.publish()
        self.assertIsInstance(caught.exception.__cause__, sqlite3.IntegrityError)
        self.assertIn('late staged fixture failure', str(caught.exception.__cause__))
        for index, request in enumerate(self.bundle.files):
            self.assertEqual(hashlib.sha256(self.sources[index].read_bytes()).hexdigest(), self.before_hashes[index])
            self.assertFalse((self.fixture.root / request.destination_relative_path).exists())
        with self.store._connect() as db:
            for table in ('local_tracks', 'local_albums', 'library_import_cohort_bindings',
                          'library_management_baselines', 'library_operation_jobs'):
                self.assertEqual(db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT state FROM library_management_import_bundles').fetchone()[0], 'rolled_back')
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])

    async def test_native_activation_guard_still_prevents_candidate_publication(self):
        self.settings.root_assignments[0].activation_preview_token = ''
        # Re-seal settings revision only, so this specifically tests the native
        # activation guard rather than merely the outer settings-revision gate.
        self.bundle = msgspec.structs.replace(self.bundle, files=tuple(msgspec.structs.replace(
            value, settings_revision=settings_revision(self.settings)) for value in self.bundle.files))
        with self.assertRaises(AutomaticManagementHoldError) as caught:
            await self.publish()
        self.assertIsInstance(caught.exception.__cause__, StaleRevisionError)
        self.assertIn('activation is stale', str(caught.exception.__cause__))
        for index, request in enumerate(self.bundle.files):
            self.assertEqual(hashlib.sha256(self.sources[index].read_bytes()).hexdigest(), self.before_hashes[index])
            self.assertFalse((self.fixture.root / request.destination_relative_path).exists())
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM local_tracks').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_import_cohort_bindings').fetchone()[0], 0)
