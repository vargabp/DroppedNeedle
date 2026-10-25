"""Real scratch-file tests; no library files or production integration."""
import hashlib
import json
import os
import tempfile
import unittest
import importlib
from pathlib import Path
from unittest.mock import patch

from services.native.file_revision import revision_from_stat
from services.native.import_cohort_observation import observe_cohort_bytes


class CohortObservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.members = {}
        for number in (1, 2):
            relative = f"Album/Digital Media {number:02}/01.flac"
            file = self.root / relative
            file.parent.mkdir(parents=True)
            file.write_bytes(bytes([number]) * 31)
            self.members[str(number)] = (relative, revision_from_stat(file.stat()))

    def observe(self, **overrides):
        arguments = dict(root_path=str(self.root), members=self.members,
                         max_total_bytes=1000, timeout_seconds=5)
        arguments.update(overrides)
        return observe_cohort_bytes(**arguments)

    def test_hashes_actual_complete_input_without_writes(self):
        before = {key: (self.root / value[0]).stat() for key, value in self.members.items()}
        self.assertEqual(self.observe(), {str(i): hashlib.sha256(bytes([i]) * 31).hexdigest() for i in (1, 2)})
        for key, value in self.members.items():
            after = (self.root / value[0]).stat()
            self.assertEqual((after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                             (before[key].st_size, before[key].st_mtime_ns, before[key].st_ctime_ns))

    def test_changed_bytes_with_preserved_native_stat_get_different_hash(self):
        first = self.observe()
        file = self.root / self.members['1'][0]
        original = file.stat()
        file.write_bytes(b'x' * 31)
        os.utime(file, ns=(original.st_atime_ns, original.st_mtime_ns))
        self.assertNotEqual(self.observe()['1'], first['1'])

    def test_stale_revision_fails_whole_observation(self):
        (self.root / self.members['2'][0]).write_bytes(b'changed size')
        with self.assertRaises(ValueError):
            self.observe()

    def test_missing_member_fails_without_partial_result(self):
        (self.root / self.members['2'][0]).unlink()
        with self.assertRaises(OSError):
            self.observe()

    def test_symlink_leaf_is_not_followed(self):
        file = self.root / self.members['2'][0]
        file.unlink()
        file.symlink_to(self.root / self.members['1'][0])
        with self.assertRaises(OSError):
            self.observe()

    def test_symlink_ancestor_and_root_are_not_followed(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.root / 'Album', target_is_directory=True)
        with self.assertRaises(OSError):
            self.observe(members={'x': ('alias/Digital Media 01/01.flac', self.members['1'][1])})
        with self.assertRaises(OSError):
            self.observe(root_path=str(alias))

    def test_fifo_does_not_block_or_hash(self):
        fifo = self.root / 'pipe'
        os.mkfifo(fifo)
        with self.assertRaises(ValueError):
            self.observe(members={'x': ('pipe', revision_from_stat(fifo.stat()))})

    def test_invalid_paths_and_duplicate_paths_rejected(self):
        for path in ('../outside', '/absolute', 'Album//track', 'Album/./track', 'Album\\track'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.observe(members={'x': (path, '1:1')})
        with self.assertRaises(ValueError):
            self.observe(members={'x': self.members['1'], 'y': self.members['1']})

    def test_budgets_fail_closed(self):
        for overrides in ({'max_total_bytes': 61}, {'max_total_bytes': -1},
                          {'timeout_seconds': float('nan')}, {'timeout_seconds': 0}, {'members': {}}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.observe(**overrides)
        with patch('services.native.import_cohort_observation.time.monotonic', side_effect=[0, 10]):
            with self.assertRaises(TimeoutError):
                self.observe(timeout_seconds=1)

    def test_path_replacement_during_read_is_rejected(self):
        original_read = os.read
        replaced = False
        file = self.root / self.members['1'][0]

        def read_and_replace(descriptor, size):
            nonlocal replaced
            result = original_read(descriptor, size)
            if not replaced:
                replaced = True
                file.rename(file.with_suffix('.old'))
                file.write_bytes(bytes([1]) * 31)
            return result

        with patch('services.native.import_cohort_observation.os.read', side_effect=read_and_replace):
            with self.assertRaises(ValueError):
                self.observe()

    def test_mutation_of_earlier_member_is_rejected_at_final_check(self):
        original_read = os.read
        calls = 0

        def read_and_mutate(descriptor, size):
            nonlocal calls
            calls += 1
            result = original_read(descriptor, size)
            if calls == 3:
                (self.root / self.members['1'][0]).write_bytes(b'x' * 31)
            return result

        with patch('services.native.import_cohort_observation.os.read', side_effect=read_and_mutate):
            with self.assertRaises(ValueError):
                self.observe()

    def test_native_persistence_reader_rejects_observed_same_stat_byte_change(self):
        # Reuse the real NativeLibraryStore fixture, not a hand-written schema.
        fixture_module = importlib.import_module('tests.infrastructure.test_import_cohort_binding')
        fixture = fixture_module.ImportCohortBindingTests()
        fixture.setUp()
        try:
            observed = self.observe()
            for index, key in enumerate(('1', '2')):
                relative, revision = self.members[key]
                fixture.db.execute(
                    'UPDATE local_tracks SET relative_path=?,file_path=?,stat_revision=? WHERE id=?',
                    (relative, str(self.root / relative), revision, f't{key}'))
                fixture.db.execute(
                    'UPDATE library_management_import_journal SET destination_relative_path=?,staged_fingerprint=? WHERE ordinal=?',
                    (relative, observed[key], index))
                fixture.files[index]['destination_relative_path'] = relative
            encoded = json.dumps(dict(origin='acquisition', policy_revision='p1', files=fixture.files))
            fixture.db.execute('UPDATE library_management_import_bundles SET request_json=?,request_hash=?',
                               (encoded, hashlib.sha256(encoded.encode()).hexdigest()))
            fixture.persist(root_path=str(self.root), album_parent='Album',
                            members={i: (f't{k}', observed[k]) for i, k in enumerate(('1', '2'))})
            fixture.db.execute("UPDATE library_management_import_bundles SET state='catalog_committed'")
            fixture.db.execute("UPDATE library_management_import_journal SET state='catalog_committed'")
            fixture.db.commit()
            arguments = dict(root_path=str(self.root), album_parent='Album',
                             current_fingerprints={f't{k}': v for k, v in observed.items()})
            self.assertIsNotNone(fixture.read(**arguments))
            file = self.root / self.members['1'][0]
            original = file.stat()
            file.write_bytes(b'x' * 31)
            os.utime(file, ns=(original.st_atime_ns, original.st_mtime_ns))
            arguments['current_fingerprints'] = {f't{k}': v for k, v in self.observe().items()}
            self.assertIsNone(fixture.read(**arguments))
        finally:
            fixture.tearDown()
