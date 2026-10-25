"""Local prospective native write preparation; never invokes publication."""
import hashlib
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import msgspec

from api.v1.schemas.library_policies import LibraryRootSettings, TypedLibrarySettings
from core.exceptions import ValidationError, LibraryManagementPolicyChangedError
from infrastructure.persistence.native_library_store import NativeLibraryStore
from models.audio import AudioInfo, AudioTag
from models.library_management import LibraryManagementImportBundle, LibraryManagementImportFile, LibraryManagementPublishedImportFile
from models.library_management_canonical import CanonicalReleaseDocument
from services.native.identification_queue_service import IdentificationQueueService
from services.native.library_filesystem_coordinator import LibraryFilesystemCoordinator
from services.native.library_policy_resolver import LibraryPolicyResolver
from services.native.target_import_library_service import TargetImportLibraryService


class CohortPreparationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / 'music'
        self.root.mkdir()
        database = Path(self.directory.name) / 'catalog.db'
        with sqlite3.connect(database) as connection:
            connection.execute('CREATE TABLE auth_users(id TEXT PRIMARY KEY)')
        self.store = NativeLibraryStore(database, threading.Lock())
        self.resolver = LibraryPolicyResolver(TypedLibrarySettings(library_roots=[
            LibraryRootSettings(id='root', path=str(self.root), label='Music', policy='automatic')]))
        self.service = TargetImportLibraryService(
            self.store, lambda: self.resolver, IdentificationQueueService(self.store),
            filesystem_coordinator=LibraryFilesystemCoordinator())
        self.published = []
        media = []
        for disc in (1, 2):
            relative = f'Artist/Album/Digital Media {disc:02}/01.flac'
            path = self.root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(bytes([disc]) * 31)
            tag = AudioTag(title='Track', artist='Artist', album_artist='Artist', album='Album',
                           disc_number=disc, track_number=1,
                           musicbrainz_release_id='release', musicbrainz_release_group_id='group',
                           musicbrainz_recording_id=f'rec{disc}', musicbrainz_release_track_id=f'rt{disc}')
            info = AudioInfo(duration_seconds=1, bitrate=1, sample_rate=44100, channels=2,
                             file_format='flac', file_size_bytes=31)
            request = LibraryManagementImportFile(
                ordinal=disc-1, input_path=f'/secured/source{disc}.flac', destination_root_id='root',
                destination_relative_path=relative, tag=tag, info=info, release_group_mbid='group',
                release_mbid='release', recording_mbid=f'rec{disc}', confidence=1, source='download',
                download_task_id='task', source_path=f'peer/{disc}.flac', authoritative_mapping=True,
                release_track_mbid=f'rt{disc}', medium_position=disc, release_track_position=1)
            self.published.append(LibraryManagementPublishedImportFile(
                request=request, destination_path=str(path), staged_fingerprint=hashlib.sha256(path.read_bytes()).hexdigest(),
                tag=tag, info=info))
            identifiers = dict(release_group_mbid='group', release_mbid='release',
                               release_track_mbid=f'rt{disc}', recording_mbid=f'rec{disc}')
            media.append(dict(position=disc, title=None, format='Digital Media', track_count=1, tracks=[dict(
                local_track_id=f'synthetic{disc}', source_track_revision=1, source_identity_revision=1,
                title='Track', artist_credits=[], relationship_credits=[], identifiers=identifiers,
                track_number=1, track_number_text='1', total_tracks=1, disc_number=disc, total_discs=2)]))
        self.bundle = LibraryManagementImportBundle(idempotency_key='candidate', origin='acquisition',
            policy_revision=self.resolver.policy_revision, files=tuple(v.request for v in self.published))
        self.canonical = msgspec.convert(dict(local_album_id='synthetic', source_album_revision=1,
            source_identity_revision=1, title='Album', artist_credits=[],
            identifiers=dict(release_group_mbid='group', release_mbid='release'), date=None, original_date=None,
            release_status='Official', release_country=None, primary_release_type='Album', secondary_release_types=[],
            packaging=None, barcode=None, asin=None, language=None, script=None, compilation=False,
            total_discs=2, labels=[], genres=[], media=media), type=CanonicalReleaseDocument)

    async def prepare(self, **overrides):
        arguments = dict(bundle=self.bundle, files=tuple(self.published), canonical=self.canonical,
                         max_total_bytes=1000, timeout_seconds=5)
        arguments.update(overrides)
        return await self.service._prepare_cohort_import_writes_candidate(**arguments)

    async def test_actual_native_builder_splits_then_candidate_prepares_one_album_without_commit(self):
        original = []
        for value in self.published:
            request = value.request
            original.append(await self.service._build_published_import_write(
                Path(value.destination_path), value.tag, value.info, resolver=self.resolver,
                release_group_mbid='group', release_mbid='release', recording_mbid=request.recording_mbid,
                source='download', download_task_id='task', source_path=request.source_path, file_mtime=None))
        self.assertEqual(len({write.album.id for write in original}), 2)
        writes, fingerprints, parent = await self.prepare()
        self.assertEqual(len({write.album.id for _, write in writes}), 1)
        self.assertEqual(parent, 'Artist/Album')
        self.assertEqual({write.grouping_context for _, write in writes}, {parent})
        self.assertEqual({write.track.local_album_id for _, write in writes}, {write.album.id for _, write in writes})
        self.assertEqual([write.track.id for _, write in writes], [write.track.id for write in original])
        self.assertEqual(len(fingerprints), 2)
        with self.store._connect() as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM local_tracks').fetchone()[0], 0)
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM library_management_import_bundles').fetchone()[0], 0)

    async def test_incomplete_publication_and_canonical_projection_rejected(self):
        with self.assertRaises(ValidationError):
            await self.prepare(files=tuple(self.published[:1]))
        medium = msgspec.structs.replace(self.canonical.media[0], track_count=2)
        with self.assertRaises(ValidationError):
            await self.prepare(canonical=msgspec.structs.replace(self.canonical, media=(medium, self.canonical.media[1])))
        with self.assertRaises(ValidationError):
            await self.prepare(canonical=msgspec.structs.replace(self.canonical, media=self.canonical.media[:1]))

    async def test_edited_published_request_rejected(self):
        value = self.published[0]
        changed = msgspec.structs.replace(value, request=msgspec.structs.replace(value.request, download_task_id='other'))
        with self.assertRaises(ValidationError):
            await self.prepare(files=(changed, self.published[1]))

    async def test_tag_identity_and_album_context_mismatch_rejected(self):
        value = self.published[0]
        for changes in (dict(musicbrainz_recording_id='other'), dict(album='Other')):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                await self.prepare(files=(msgspec.structs.replace(value, tag=msgspec.structs.replace(value.tag, **changes)), self.published[1]))

    async def test_changed_published_bytes_rejected(self):
        Path(self.published[1].destination_path).write_bytes(b'x' * 31)
        with self.assertRaises(ValidationError):
            await self.prepare()

    async def test_wrong_policy_and_absent_coordinator_rejected(self):
        with self.assertRaises(LibraryManagementPolicyChangedError):
            await self.prepare(bundle=msgspec.structs.replace(self.bundle, policy_revision='stale'))
        self.service._filesystem = None
        with self.assertRaises(ValidationError):
            await self.prepare()

    async def test_existing_owner_not_forced_into_candidate(self):
        self.service._import_reuse_album_id = AsyncMock(return_value='existing-owner')
        with self.assertRaises(ValidationError):
            await self.prepare()

    async def test_owner_appearing_during_native_build_is_not_overridden(self):
        self.service._import_reuse_album_id = AsyncMock(side_effect=[None, 'new-owner'])
        with self.assertRaises(ValidationError):
            await self.prepare()

    async def test_mutation_during_native_build_is_rejected(self):
        original = self.service._build_published_import_write

        async def mutate(*args, **kwargs):
            write = await original(*args, **kwargs)
            Path(args[0]).write_bytes(b'x' * 31)
            return write

        self.service._build_published_import_write = mutate
        with self.assertRaises((ValueError, ValidationError)):
            await self.prepare()

    async def test_wrong_destination_and_symlink_rejected(self):
        value = self.published[0]
        with self.assertRaises(ValidationError):
            await self.prepare(files=(msgspec.structs.replace(value, destination_path='/outside/file.flac'), self.published[1]))
        path = Path(value.destination_path)
        path.unlink()
        path.symlink_to(Path(self.published[1].destination_path))
        with self.assertRaises(OSError):
            await self.prepare()
