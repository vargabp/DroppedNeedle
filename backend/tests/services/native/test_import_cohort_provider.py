"""Native projection/store/source-fence integration with simulated repository I/O.

No external provider/network assertion and no full automatic baseline acceptance.
"""
from dataclasses import replace
import hashlib
import importlib
import sqlite3
import unittest
from unittest.mock import AsyncMock, patch

import msgspec

from api.v1.schemas.library_management import LibraryManagementProfile
from core.exceptions import ProviderIdentityRequiredError, StaleRevisionError
from infrastructure.persistence.import_cohort_binding import CohortSnapshotSeal
from models.library_management_canonical import IncomingTrackManagementMapping
from repositories import musicbrainz_base as mb
from repositories.musicbrainz_management_models import MbManagementRelease
from services.native.canonical_release_metadata_service import CanonicalReleaseMetadataService


class CohortProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module = importlib.import_module('tests.infrastructure.test_import_cohort_snapshot')
        self.snapshot_fixture = module.CohortSnapshotCommitTests()
        await self.snapshot_fixture.asyncSetUp()
        self.addCleanup(self.snapshot_fixture.doCleanups)
        self.atomic = self.snapshot_fixture.atomic
        self.context = mb.capture_mb_source_context()
        self.release = msgspec.convert(dict(id='release', title='Album', **{
            'release-group': {'id': 'group'},
            'artist-credit': [{'name': 'Artist', 'artist': {'id': 'artist', 'name': 'Artist', 'sort-name': 'Artist'}}],
            'media': [{'position': disc, 'format': 'Digital Media', 'track-count': 1,
                       'tracks': [{'id': f'rt{disc}', 'position': 1, 'number': '1', 'title': 'Track',
                                   'recording': {'id': f'rec{disc}', 'title': 'Track'}}]} for disc in (1, 2)]
        }), type=MbManagementRelease)
        self.repository = AsyncMock()

        async def response(*args, **kwargs):
            # Simulate the response-context observation made by the native
            # repository/cache path; this is not a real HTTP proof.
            mb._mb_response_context.set(kwargs['source_context'])
            return self.release

        self.repository.get_canonical_release.side_effect = response
        self.service = CanonicalReleaseMetadataService(self.atomic.store, self.repository,
            clock=lambda: 1, _capture_cohort_source=True)

    async def project(self):
        return await self.service.build_for_import(local_album_id='prospective', release_group_mbid='group',
            release_mbid='release', mappings=tuple(IncomingTrackManagementMapping(
                local_track_id=f't{disc}', medium_position=disc, release_track_position=1,
                recording_mbid=f'rec{disc}', release_track_mbid=f'rt{disc}') for disc in (1, 2)),
            profile=LibraryManagementProfile(id='profile', name='Fixture'), source_context=self.context)

    async def correlate(self, projection):
        fixture = self.atomic.fixture
        fixture.bundle = msgspec.structs.replace(fixture.bundle, files=tuple(msgspec.structs.replace(
            value, metadata_snapshot_id=projection.metadata_snapshot_id) for value in fixture.bundle.files))
        encoded = msgspec.json.encode(fixture.bundle).decode()
        with self.atomic.store._connect() as db:
            db.execute('UPDATE library_management_import_bundles SET request_json=?,request_hash=?',
                       (encoded, hashlib.sha256(encoded.encode()).hexdigest()))
        self.atomic.candidate = replace(self.atomic.candidate, snapshot=CohortSnapshotSeal(
            snapshot_id=projection.metadata_snapshot_id, input_hash=projection.input_hash,
            payload_sha256=projection.payload_sha256, source_proof_id=projection.cohort_source_proof_id))

    async def test_native_projection_persists_source_and_commit_checks_it(self):
        projection = await self.project()
        self.assertIsNotNone(projection.cohort_source_proof_id)
        with self.atomic.store._connect() as db:
            row = db.execute('SELECT * FROM library_import_cohort_snapshot_sources').fetchone()
            self.assertEqual(row['source_generation'], self.context.generation)
            self.assertEqual(row['source_url_sha256'], hashlib.sha256(self.context.source_url.encode()).hexdigest())
            self.assertEqual(row['metadata_snapshot_id'], projection.metadata_snapshot_id)
        await self.correlate(projection)
        self.assertEqual(len(await self.atomic.commit(source_context=self.context)), 2)

    async def test_missing_or_stale_response_marker_is_not_reused(self):
        mb._mb_response_context.set(self.context)
        self.repository.get_canonical_release.side_effect = None
        self.repository.get_canonical_release.return_value = self.release
        with self.assertRaises(ProviderIdentityRequiredError):
            await self.project()
        with self.atomic.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_import_cohort_snapshot_sources').fetchone()[0], 0)

    async def test_source_change_before_projection_refuses_publication(self):
        with patch.object(mb, '_mb_source_generation', self.context.generation + 1):
            with self.assertRaises(ProviderIdentityRequiredError):
                await self.project()

    async def test_source_fence_blocks_old_context_at_catalog_commit(self):
        projection = await self.project()
        await self.correlate(projection)
        with patch.object(mb, '_mb_source_generation', self.context.generation + 1):
            with self.assertRaises(StaleRevisionError):
                await self.atomic.commit(source_context=self.context)
        self.atomic.assert_uncommitted()

    async def test_fresh_context_cannot_relabel_old_source_observation(self):
        projection = await self.project()
        await self.correlate(projection)
        with patch.object(mb, '_mb_source_generation', self.context.generation + 1):
            current = mb.capture_mb_source_context()
            with self.assertRaisesRegex(ValueError, 'source context changed'):
                await self.atomic.commit(source_context=current)
        self.atomic.assert_uncommitted()

    async def test_source_observation_is_immutable(self):
        await self.project()
        with self.atomic.store._connect() as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute('UPDATE library_import_cohort_snapshot_sources SET source_generation=999')

    async def test_missing_candidate_schema_rolls_back_snapshot_creation(self):
        with self.atomic.store._connect() as db:
            before = db.execute('SELECT COUNT(*) FROM library_management_metadata_snapshots').fetchone()[0]
            db.execute('DROP TABLE library_import_cohort_snapshot_sources')
        with self.assertRaises(sqlite3.OperationalError):
            await self.project()
        with self.atomic.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_management_metadata_snapshots').fetchone()[0], before)

    async def test_default_projection_does_not_require_candidate_schema_or_marker(self):
        self.service._capture_cohort_source = False
        with self.atomic.store._connect() as db:
            db.execute('DROP TABLE library_import_cohort_snapshot_sources')
        self.repository.get_canonical_release.side_effect = None
        self.repository.get_canonical_release.return_value = self.release
        projection = await self.project()
        self.assertIsNone(projection.cohort_source_proof_id)
