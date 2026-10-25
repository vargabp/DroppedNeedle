"""Fenced whole-context grouping slice; NOT the full native scan loop."""
import importlib
import unittest
from unittest.mock import AsyncMock, patch

import msgspec

from core.exceptions import StaleRevisionError, ValidationError
from services.native.import_cohort_grouping_candidate import apply_cohort_grouping_candidate
from services.native.local_album_grouper import LocalAlbumGrouper
from services.native.local_album_grouping_service import grouping_track_from_row


class CohortGroupingCandidateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module = importlib.import_module('tests.services.native.test_import_cohort_scan_admission')
        self.fixture = module.CohortScanAdmissionTests()
        await self.fixture.asyncSetUp()
        self.addCleanup(self.fixture.doCleanups)
        await self.fixture.commit()
        self.store = self.fixture.store

    async def apply(self):
        return await apply_cohort_grouping_candidate(store=self.store,
            filesystem=self.fixture.fixture.staged.service._filesystem,
            resolver=self.fixture.fixture.staged.fixture.resolver,
            binding_id=self.fixture.binding_id, run_id='continuity-scan', now=4,
            max_total_bytes=10000000, timeout_seconds=5)

    def loop(self):
        return self.fixture.fixture.cohort_grouping

    async def regroup(self, **kwargs):
        return await self.loop().regroup_run('continuity-scan', now=4,
            frozen_policy_revision=self.fixture.fixture.staged.bundle.policy_revision, **kwargs)

    def assert_loop_pending(self):
        self.fixture.assert_preserved()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT state FROM library_scan_grouping_contexts').fetchone()[0], 'pending')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_identification_jobs').fetchone()[0], 0)

    async def test_public_index_drains_pending_grouping_on_restart_without_reindex_claim(self):
        # No pending inventory: exercises the actual public index() terminal
        # branch and native regroup loop, NOT discovery/tag-reading admission.
        from models.library_work import ScanRun
        from services.native.library_indexer import LibraryIndexer
        indexer = LibraryIndexer(self.store, self.fixture.fixture.staged.audio, self.loop(),
            self.fixture.fixture.staged.service._filesystem)
        counts = await indexer.index(ScanRun(id='continuity-scan', kind='rescan_files', trigger='manual',
            state='indexing', phase='indexing'), frozen_policy_revision=self.fixture.fixture.staged.bundle.policy_revision)
        self.assertEqual(counts['indexed'], 0)
        self.assertEqual(counts['tag_reads'], 0)
        self.assertEqual(counts['identification_enqueued'], 1)
        self.fixture.assert_preserved()
        self.assertEqual(await self.regroup(), 0)
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT state FROM library_scan_grouping_contexts').fetchone()[0], 'completed')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_identification_jobs').fetchone()[0], 1)

    async def test_loop_pause_after_application_remains_pending_and_resumes_once(self):
        self.assertEqual(await self.regroup(checkpoint=AsyncMock(side_effect=[True, False])), 0)
        self.assert_loop_pending()
        self.assertEqual(await self.regroup(), 1)
        self.assertEqual(await self.regroup(), 0)

    async def test_loop_refuses_paged_path_without_falling_back(self):
        module = importlib.import_module('services.native.local_album_grouping_service')
        with patch.object(module, 'STAGED_GROUPING_THRESHOLD', 0):
            with self.assertRaisesRegex(ValidationError, 'paged cohort grouping is not integrated'):
                await self.regroup()
        self.assert_loop_pending()

    async def test_loop_refuses_child_context_containing_only_one_bound_medium(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_grouping_contexts SET relative_directory='Artist/Album/Digital Media 01'")
        with self.assertRaisesRegex(StaleRevisionError, 'exclusive complete'):
            await self.regroup()
        self.assert_loop_pending()

    async def test_loop_frozen_policy_mismatch_does_not_apply(self):
        with self.assertRaisesRegex(StaleRevisionError, 'Frozen cohort scan policy changed'):
            await self.loop().regroup_run('continuity-scan', now=4, frozen_policy_revision='stale')
        self.assert_loop_pending()

    async def test_loop_honors_native_pause_control_at_application_transaction(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_runs SET requested_control='pause'")
        with self.assertRaisesRegex(StaleRevisionError, 'not pending/running'):
            await self.regroup()
        self.assert_loop_pending()

    async def test_loop_refuses_manual_exclusion_before_application(self):
        with self.store._connect() as db:
            db.execute('UPDATE local_tracks SET manual_excluded=1')
        with self.assertRaisesRegex(StaleRevisionError, 'exclusion or applied policy'):
            await self.regroup()
        self.assert_loop_pending()

    async def test_default_grouping_never_queries_candidate_tables(self):
        default = self.fixture.fixture.grouping
        with patch.object(default, '_cohort_binding_for_context', new=AsyncMock(side_effect=AssertionError('candidate enabled'))) as discovery:
            await default.regroup_run('continuity-scan', now=4)
        discovery.assert_not_awaited()

    async def test_opt_in_unbound_context_cannot_fall_through_to_legacy_writes(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_grouping_contexts SET relative_directory='Unbound'")
        with self.assertRaisesRegex(ValidationError, 'requires a complete bound context'):
            await self.regroup()
        self.assert_loop_pending()

    def add_foreign_context_track(self):
        with self.store._connect() as db:
            album = dict(db.execute('SELECT * FROM local_albums LIMIT 1').fetchone())
            album.update(id='foreign-album', grouping_key='foreign-group')
            track = dict(db.execute('SELECT * FROM local_tracks LIMIT 1').fetchone())
            track.update(id='foreign-track', local_album_id='foreign-album',
                relative_path='Artist/Album/Bonus.flac', file_path='/foreign-fixture/Bonus.flac')
            for table, row in [('local_albums', album), ('local_tracks', track)]:
                db.execute(f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", tuple(row.values()))

    async def test_loop_refuses_foreign_sibling_album_in_same_context(self):
        self.add_foreign_context_track()
        with self.assertRaisesRegex(StaleRevisionError, 'exclusive complete'):
            await self.regroup()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT state FROM library_scan_grouping_contexts').fetchone()[0], 'pending')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_identification_jobs').fetchone()[0], 0)

    async def test_foreign_arrival_after_grouping_is_refused_at_completion_sql_fence(self):
        original = self.store.enqueue_identification_job_results
        async def arrival(*args, **kwargs):
            self.add_foreign_context_track()
            return await original(*args, **kwargs)
        with patch.object(self.store, 'enqueue_identification_job_results', side_effect=arrival):
            with self.assertRaisesRegex(StaleRevisionError, 'exclusive complete'):
                await self.regroup()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT state FROM library_scan_grouping_contexts').fetchone()[0], 'pending')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_identification_jobs').fetchone()[0], 0)

    async def test_exact_native_context_applies_one_group_without_locks_or_false_completion(self):
        album_ids, _revision = await self.apply()
        self.assertEqual(album_ids, [self.fixture.fixture.original[0][1]])
        self.fixture.assert_preserved()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM local_albums').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT state FROM library_scan_grouping_contexts').fetchone()[0], 'pending')
            self.assertEqual([row[0] for row in db.execute('SELECT membership_locked FROM local_tracks')], [0, 0])

    async def tracks(self):
        rows = await self.store.get_grouping_context_tracks('root', 'Artist/Album')
        return [grouping_track_from_row(row) for row in rows]

    async def test_context_changes_proposals_not_paths_or_default_directory_rules(self):
        tracks = await self.tracks()
        paths = [track.relative_path for track in tracks]
        grouper = LocalAlbumGrouper()
        self.assertEqual(len(grouper.group(tracks)), 2)
        self.assertEqual(len(grouper.group(tracks, _cohort_directories_candidate={
            track.local_track_id: 'Artist/Album' for track in tracks})), 1)
        self.assertEqual([track.relative_path for track in tracks], paths)
        self.assertEqual(len(grouper.group(tracks)), 2)

    async def test_partial_context_map_is_not_complete_proposal_admission(self):
        tracks = await self.tracks()
        with self.assertRaisesRegex(ValueError, 'Complete single-root'):
            LocalAlbumGrouper().group(tracks, _cohort_directories_candidate={
                tracks[0].local_track_id: 'Artist/Album'})
        self.fixture.assert_preserved()

    async def test_late_policy_drift_is_refused_by_actual_grouping_write(self):
        original = self.store.apply_grouping_context
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute("UPDATE library_policy_state SET desired_policy_revision='late-group-policy'")
            return await original(*args, **kwargs)
        with patch.object(self.store, 'apply_grouping_context', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'cohort changed'):
                await self.apply()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT state FROM library_scan_grouping_contexts').fetchone()[0], 'pending')

    async def test_partial_application_is_refused_inside_native_grouping_transaction(self):
        original = self.store.apply_grouping_context
        async def partial(run_id, root_id, directory, applications, **kwargs):
            changed = msgspec.structs.replace(applications[0], group=msgspec.structs.replace(
                applications[0].group, track_ids=applications[0].group.track_ids[:1]))
            return await original(run_id, root_id, directory, [changed], **kwargs)
        with patch.object(self.store, 'apply_grouping_context', side_effect=partial):
            with self.assertRaisesRegex(ValidationError, 'Exact complete'):
                await self.apply()
        self.fixture.assert_preserved()

    async def test_postwrite_catalog_mutation_rolls_back_grouping(self):
        with self.store._connect() as db:
            revision = db.execute('SELECT value FROM library_catalog_revision').fetchone()[0]
            db.execute("CREATE TRIGGER corrupt_group AFTER UPDATE ON library_catalog_revision BEGIN UPDATE local_tracks SET source_path='changed-by-group-fixture'; END")
        with self.assertRaisesRegex(StaleRevisionError, 'cohort changed'):
            await self.apply()
        self.fixture.assert_preserved()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT value FROM library_catalog_revision').fetchone()[0], revision)

    async def test_complete_but_changed_album_title_is_refused_by_sql_fence(self):
        original = self.store.apply_grouping_context
        async def altered(run_id, root_id, directory, applications, **kwargs):
            changed = msgspec.structs.replace(applications[0], group=msgspec.structs.replace(
                applications[0].group, title='Unrelated album title'))
            return await original(run_id, root_id, directory, [changed], **kwargs)
        with patch.object(self.store, 'apply_grouping_context', side_effect=altered):
            with self.assertRaisesRegex(ValidationError, 'proposal disagrees'):
                await self.apply()
        self.fixture.assert_preserved()


class CohortPagedTrackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = CohortGroupingCandidateTests()
        await self.fixture.asyncSetUp()
        self.addCleanup(self.fixture.doCleanups)
        self.store = self.fixture.store

    async def stage(self):
        from services.native.import_cohort_grouping_candidate import stage_cohort_track_page_candidate
        return await stage_cohort_track_page_candidate(store=self.store,
            filesystem=self.fixture.fixture.fixture.staged.service._filesystem,
            resolver=self.fixture.fixture.fixture.staged.fixture.resolver,
            binding_id=self.fixture.fixture.binding_id, run_id='continuity-scan',
            root_id='root', directory='Artist/Album', limit=1, max_total_bytes=10000000, timeout_seconds=5)

    async def run_large(self, **kwargs):
        module = importlib.import_module('services.native.local_album_grouping_service')
        with patch.object(module, 'STAGED_GROUPING_THRESHOLD', 0), patch.object(module, 'STAGING_BATCH_SIZE', 1):
            return await self.fixture.regroup(**kwargs)

    def state(self):
        self.fixture.fixture.assert_preserved()
        with self.store._connect() as db:
            context = dict(db.execute('SELECT * FROM library_scan_grouping_contexts').fetchone())
            rows = [dict(row) for row in db.execute('SELECT * FROM library_scan_grouping_evidence ORDER BY local_track_id')]
            self.assertEqual(context['state'], 'pending')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_identification_jobs').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_scan_grouping_groups').fetchone()[0], 0)
            return context, rows

    async def test_actual_paged_loop_stages_every_medium_then_stops_before_tokens(self):
        with patch.object(self.store, 'prepare_staged_grouping_tokens', new=AsyncMock(side_effect=AssertionError('unfenced token write'))) as tokens:
            with self.assertRaisesRegex(ValidationError, 'beyond validated track staging'):
                await self.run_large()
        tokens.assert_not_awaited()
        context, rows = self.state()
        self.assertEqual((context['staging_state'], context['staging_cursor']), ('tokens', None))
        self.assertEqual({row['local_track_id'] for row in rows}, set(self.fixture.fixture.fixture.result.local_track_ids))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['reason_code'] == 'COMPATIBLE_DISC_DIRECTORIES' for row in rows))

    async def test_actual_checkpoint_pause_resumes_durable_prefix_without_duplicates(self):
        self.assertEqual(await self.run_large(checkpoint=AsyncMock(side_effect=[True, True, False, False])), 0)
        context, rows = self.state()
        self.assertEqual(context['staging_state'], 'tracks')
        self.assertEqual(len(rows), 1)
        first = rows[0]
        with self.assertRaisesRegex(ValidationError, 'beyond validated track staging'):
            await self.run_large()
        context, rows = self.state()
        self.assertEqual(context['staging_state'], 'tokens')
        self.assertEqual(len(rows), 2)
        self.assertIn(first, rows)

    async def test_changed_page_evidence_is_refused_inside_native_sql(self):
        original = self.store.stage_grouping_track_page
        async def changed(run_id, root_id, directory, evidence, **kwargs):
            evidence = [dict(row, title='Not the native tag title') for row in evidence]
            return await original(run_id, root_id, directory, evidence, **kwargs)
        with patch.object(self.store, 'stage_grouping_track_page', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'page, cursor or exhaustion changed'):
                await self.stage()
        self.assertEqual(self.state()[1], [])

        self.assertEqual(self.state()[0]['staging_state'], 'pending')

    async def test_false_exhaustion_cannot_skip_unstaged_members(self):
        original = self.store.stage_grouping_track_page
        async def exhausted(run_id, root_id, directory, evidence, **kwargs):
            kwargs.update(next_cursor=None, exhausted=True)
            return await original(run_id, root_id, directory, [], **kwargs)
        with patch.object(self.store, 'stage_grouping_track_page', side_effect=exhausted):
            with self.assertRaisesRegex(StaleRevisionError, 'page, cursor or exhaustion changed'):
                await self.stage()
        self.assertEqual(self.state()[1], [])

    async def test_late_policy_drift_does_not_advance_cursor_or_stage_rows(self):
        original = self.store.stage_grouping_track_page
        async def changed(*args, **kwargs):
            filesystem = self.fixture.fixture.fixture.staged.service._filesystem
            self.assertTrue(filesystem._cohort_writer_is_held('root'))
            with self.store._connect() as db:
                db.execute("UPDATE library_policy_state SET desired_policy_revision='late-paged-policy'")
            return await original(*args, **kwargs)
        with patch.object(self.store, 'stage_grouping_track_page', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'cohort changed'):
                await self.stage()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_scan_grouping_evidence').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT staging_state FROM library_scan_grouping_contexts').fetchone()[0], 'pending')

    async def test_native_pause_refuses_page_before_staging(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_runs SET requested_control='pause'")
        with self.assertRaisesRegex(StaleRevisionError, 'not pending/running'):
            await self.stage()
        self.assertEqual(self.state()[1], [])

    async def test_postinsert_evidence_mutation_rolls_back_entire_page_and_cursor(self):
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_page AFTER INSERT ON library_scan_grouping_evidence BEGIN UPDATE library_scan_grouping_evidence SET title='changed-by-fixture' WHERE local_track_id=NEW.local_track_id; END")
        with self.assertRaisesRegex(StaleRevisionError, 'exact validated prefix'):
            await self.stage()
        self.assertEqual(self.state()[1], [])
        self.assertEqual(self.state()[0]['staging_state'], 'pending')

    async def test_tampered_prior_page_blocks_resume_without_overwriting_it(self):
        self.assertFalse(await self.stage())
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_grouping_evidence SET old_album_id=old_album_id,title='tampered-prefix'")
        with self.assertRaisesRegex(StaleRevisionError, 'exact validated prefix'):
            await self.stage()
        context, rows = self.state()
        self.assertEqual(context['staging_state'], 'tracks')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['title'], 'tampered-prefix')

    async def test_unknown_member_cursor_cannot_skip_evidence(self):
        with self.store._connect() as db:
            db.execute('UPDATE library_scan_grouping_contexts SET staging_state=?,staging_cursor=?',
                ('tracks', '["not-a-member","not-a-track"]'))
        with self.assertRaisesRegex(StaleRevisionError, 'not a member boundary'):
            await self.stage()
        self.assertEqual(self.state()[1], [])

    async def test_late_context_revision_change_refuses_page_even_with_same_cursor(self):
        original = self.store.stage_grouping_track_page
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute('UPDATE library_scan_grouping_contexts SET row_revision=row_revision+1')
            return await original(*args, **kwargs)
        with patch.object(self.store, 'stage_grouping_track_page', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'context revision changed'):
                await self.stage()
        self.assertEqual(self.state()[1], [])

class CohortTokenTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = CohortPagedTrackTests()
        await self.fixture.asyncSetUp()
        self.addCleanup(self.fixture.doCleanups)
        self.store = self.fixture.store
        self.assertFalse(await self.fixture.stage())
        self.assertFalse(await self.fixture.stage())
        self.assertTrue(await self.fixture.stage())

    async def token(self):
        from services.native.import_cohort_grouping_candidate import prepare_cohort_token_page_candidate
        source = self.fixture.fixture.fixture.fixture
        return await prepare_cohort_token_page_candidate(store=self.store,
            filesystem=source.staged.service._filesystem, resolver=source.staged.fixture.resolver,
            binding_id=self.fixture.fixture.fixture.binding_id, run_id='continuity-scan',
            root_id='root', directory='Artist/Album', limit=1, max_total_bytes=10000000, timeout_seconds=5)

    def context(self):
        self.fixture.fixture.fixture.assert_preserved()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_identification_jobs').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM local_albums').fetchone()[0], 1)
            return dict(db.execute('SELECT * FROM library_scan_grouping_contexts').fetchone())

    async def test_native_token_pages_count_every_member_once_then_stop_at_summary(self):
        self.assertFalse(await self.token())
        self.assertFalse(await self.token())
        self.assertTrue(await self.token())
        self.assertEqual(self.context()['staging_state'], 'groups')
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT automatic_track_count FROM library_scan_grouping_groups').fetchone()[0], 2)
            self.assertEqual(db.execute('SELECT overlap_count FROM library_scan_grouping_edges').fetchone()[0], 2)
        with self.assertRaisesRegex(ValidationError, 'Exact bounded native token page'):
            await self.token()

    async def test_actual_private_token_loop_never_calls_unfenced_summary(self):
        self.fixture.fixture.loop()._cohort_tokens_candidate_enabled = True
        with patch.object(self.store, 'stage_grouping_summary_page', new=AsyncMock(side_effect=AssertionError('unfenced summary'))) as summary:
            with self.assertRaisesRegex(ValidationError, 'beyond validated tokens'):
                await self.fixture.run_large()
        summary.assert_not_awaited()
        self.assertEqual(self.context()['staging_state'], 'groups')

    async def test_processed_prefix_resume_does_not_double_count(self):
        self.assertFalse(await self.token())
        self.assertFalse(await self.token())
        self.assertTrue(await self.token())
        self.context()
        with self.store._connect() as db:
            self.assertEqual([row[0] for row in db.execute('SELECT occurrence_count FROM library_scan_grouping_values')], [2, 2, 2])

    async def test_changed_accumulator_blocks_next_page_without_overwrite(self):
        self.assertFalse(await self.token())
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_grouping_groups SET stat_revision_accumulator='bad'")
        with self.assertRaisesRegex(StaleRevisionError, 'revision accumulators changed'):
            await self.token()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT automatic_track_count FROM library_scan_grouping_groups').fetchone()[0], 1)

    async def test_late_context_revision_refuses_entire_token_page(self):
        original = self.store.prepare_staged_grouping_tokens
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute('UPDATE library_scan_grouping_contexts SET row_revision=row_revision+1')
            return await original(*args, **kwargs)
        with patch.object(self.store, 'prepare_staged_grouping_tokens', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'context revision changed'):
                await self.token()
        self.assertEqual(self.context()['grouping_merge_ready'], 0)

    async def test_postwrite_value_count_corruption_rolls_back_token_and_merge_state(self):
        with self.store._connect() as db:
            db.execute('CREATE TRIGGER corrupt_token AFTER INSERT ON library_scan_grouping_values BEGIN UPDATE library_scan_grouping_values SET occurrence_count=99; END')
        with self.assertRaisesRegex(StaleRevisionError, 'value counts changed'):
            await self.token()
        self.assertEqual(self.context()['grouping_merge_ready'], 0)
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_scan_grouping_groups').fetchone()[0], 0)

    async def test_native_pause_leaves_unassigned_tokens(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_runs SET requested_control='pause'")
        with self.assertRaisesRegex(StaleRevisionError, 'not pending/running'):
            await self.token()
        self.assertEqual(self.context()['grouping_merge_ready'], 0)

    async def test_actual_token_checkpoint_resume_does_not_recount_prefix(self):
        self.fixture.fixture.loop()._cohort_tokens_candidate_enabled = True
        self.assertEqual(await self.fixture.run_large(checkpoint=AsyncMock(side_effect=[True, True, False, False])), 0)
        self.assertEqual(self.context()['staging_state'], 'tokens')
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT automatic_track_count FROM library_scan_grouping_groups').fetchone()[0], 1)
        with self.assertRaisesRegex(ValidationError, 'beyond validated tokens'):
            await self.fixture.run_large()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT automatic_track_count FROM library_scan_grouping_groups').fetchone()[0], 2)

    async def test_foreign_token_on_unprocessed_member_is_not_relabelled(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_grouping_evidence SET grouping_token='foreign' WHERE local_track_id=(SELECT MAX(local_track_id) FROM library_scan_grouping_evidence)")
        with self.assertRaisesRegex(StaleRevisionError, 'evidence or processed prefix changed'):
            await self.token()
        self.assertEqual(self.context()['grouping_merge_ready'], 0)

    async def test_late_durable_policy_disagreement_blocks_token_writes(self):
        original = self.store.prepare_staged_grouping_tokens
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute("UPDATE library_policy_state SET desired_policy_revision='token-policy-drift'")
            return await original(*args, **kwargs)
        with patch.object(self.store, 'prepare_staged_grouping_tokens', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'cohort changed'):
                await self.token()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_scan_grouping_groups').fetchone()[0], 0)


class CohortSummaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = CohortTokenTests()
        await self.fixture.asyncSetUp()
        self.addCleanup(self.fixture.doCleanups)
        self.store = self.fixture.store
        self.assertFalse(await self.fixture.token())
        self.assertFalse(await self.fixture.token())
        self.assertTrue(await self.fixture.token())

    async def summary(self):
        from services.native.import_cohort_grouping_candidate import summarize_cohort_page_candidate
        source = self.fixture.fixture.fixture.fixture.fixture
        return await summarize_cohort_page_candidate(store=self.store,
            filesystem=source.staged.service._filesystem, resolver=source.staged.fixture.resolver,
            binding_id=self.fixture.fixture.fixture.fixture.binding_id, run_id='continuity-scan',
            root_id='root', directory='Artist/Album', limit=1, max_total_bytes=10000000, timeout_seconds=5)

    def group(self):
        self.assertEqual(self.fixture.context()['state'], 'pending')
        with self.store._connect() as db:
            return dict(db.execute('SELECT * FROM library_scan_grouping_groups').fetchone())

    async def loop(self, **kwargs):
        loop = self.fixture.fixture.fixture.loop()
        loop._cohort_tokens_candidate_enabled = True
        loop._cohort_summaries_candidate_enabled = True
        return await self.fixture.fixture.run_large(**kwargs)

    async def test_actual_summary_writes_native_fields_then_stops_at_continuity(self):
        from infrastructure.persistence.native_library_store import _normalize_exact
        self.assertFalse(await self.summary())
        group = self.group()
        self.assertEqual(group['grouping_key'], f"root:Artist/Album:{_normalize_exact(group['title'])}:{_normalize_exact(group['album_artist_name'])}")
        self.assertTrue(group['title'])
        self.assertEqual(group['reason_code'], 'COMPATIBLE_DISC_DIRECTORIES')
        self.assertIsNone(group['retained_album_id'])
        self.assertTrue(await self.summary())
        self.assertEqual(self.fixture.context()['staging_state'], 'continuity')
        with self.assertRaisesRegex(ValidationError, 'Exact bounded native summary page'):
            await self.summary()

    async def test_actual_loop_does_not_enter_unfenced_continuity(self):
        with patch.object(self.store, 'assign_isolated_grouping_continuity', new=AsyncMock(side_effect=AssertionError('unfenced continuity'))) as continuity:
            with self.assertRaisesRegex(ValidationError, 'beyond validated summaries'):
                await self.loop()
        continuity.assert_not_awaited()
        self.assertEqual(self.fixture.context()['staging_state'], 'continuity')
        self.assertEqual(self.group()['automatic_track_count'], 2)

    async def test_checkpoint_resume_keeps_summary_and_does_not_recount(self):
        self.assertEqual(await self.loop(checkpoint=AsyncMock(side_effect=[True, True, False, False])), 0)
        group = self.group()
        self.assertEqual(self.fixture.context()['staging_state'], 'groups')
        with self.assertRaisesRegex(ValidationError, 'beyond validated summaries'):
            await self.loop()
        self.assertEqual(self.group(), group)

    async def test_late_context_revision_prevents_summary_write(self):
        original = self.store.stage_grouping_summary_page
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute('UPDATE library_scan_grouping_contexts SET row_revision=row_revision+1')
            return await original(*args, **kwargs)
        with patch.object(self.store, 'stage_grouping_summary_page', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'summary context revision changed'):
                await self.summary()
        self.assertEqual(self.group()['grouping_key'], '')

    async def test_late_policy_disagreement_leaves_summary_unwritten(self):
        original = self.store.stage_grouping_summary_page
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute("UPDATE library_policy_state SET desired_policy_revision='summary-policy-drift'")
            return await original(*args, **kwargs)
        with patch.object(self.store, 'stage_grouping_summary_page', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'cohort changed'):
                await self.summary()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT grouping_key FROM library_scan_grouping_groups').fetchone()[0], '')

    async def test_unknown_summary_cursor_cannot_skip_unsummarized_token(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_grouping_contexts SET staging_cursor='unknown-token'")
        with self.assertRaisesRegex(StaleRevisionError, 'summary phase, merge target or cursor changed'):
            await self.summary()
        self.assertEqual(self.group()['grouping_key'], '')

    async def test_incomplete_assigned_token_set_cannot_be_summarized(self):
        with self.store._connect() as db:
            db.execute('UPDATE library_scan_grouping_evidence SET grouping_token=NULL WHERE local_track_id=(SELECT MAX(local_track_id) FROM library_scan_grouping_evidence)')
        with self.assertRaisesRegex(StaleRevisionError, 'evidence or processed prefix changed'):
            await self.summary()
        self.assertEqual(self.group()['grouping_key'], '')

    async def test_corrupt_value_frequency_cannot_choose_summary(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_grouping_values SET occurrence_count=99 WHERE value_kind='title'")
        with self.assertRaisesRegex(StaleRevisionError, 'value counts changed'):
            await self.summary()
        self.assertEqual(self.group()['grouping_key'], '')

    async def test_postwrite_retained_identity_mutation_rolls_back_summary(self):
        with self.store._connect() as db:
            db.execute('CREATE TRIGGER corrupt_summary AFTER UPDATE OF grouping_key ON library_scan_grouping_groups BEGIN UPDATE library_scan_grouping_groups SET retained_album_id=(SELECT id FROM local_albums LIMIT 1); END')
        with self.assertRaisesRegex(StaleRevisionError, 'summary or revision accumulators changed'):
            await self.summary()
        self.assertEqual(self.group()['grouping_key'], '')
        self.assertIsNone(self.group()['retained_album_id'])
        self.assertIsNone(self.fixture.context()['staging_cursor'])

    async def test_native_pause_blocks_summary_before_any_field_change(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_runs SET requested_control='pause'")
        with self.assertRaisesRegex(StaleRevisionError, 'not pending/running'):
            await self.summary()
        self.assertEqual(self.group()['grouping_key'], '')


class CohortContinuityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = CohortSummaryTests()
        await self.fixture.asyncSetUp()
        self.addCleanup(self.fixture.doCleanups)
        self.store = self.fixture.store
        self.assertFalse(await self.fixture.summary())
        self.assertTrue(await self.fixture.summary())

    async def retain(self, *, finish=False):
        from services.native.import_cohort_grouping_candidate import retain_cohort_continuity_candidate
        source = self.fixture.fixture.fixture.fixture.fixture.fixture
        return await retain_cohort_continuity_candidate(store=self.store,
            filesystem=source.staged.service._filesystem, resolver=source.staged.fixture.resolver,
            binding_id=self.fixture.fixture.fixture.fixture.fixture.binding_id, run_id='continuity-scan',
            root_id='root', directory='Artist/Album', limit=1, max_total_bytes=10000000, timeout_seconds=5, finish=finish)

    def context(self):
        return self.fixture.fixture.context()

    def group(self):
        return self.fixture.group()

    async def loop(self, **kwargs):
        self.fixture.fixture.fixture.fixture.loop()._cohort_continuity_candidate_enabled = True
        return await self.fixture.loop(**kwargs)

    async def test_native_isolated_assignment_retains_original_identity_and_replays_without_work(self):
        self.assertEqual(await self.retain(), 1)
        self.assertEqual(await self.retain(), 0)
        group = self.group()
        with self.store._connect() as db:
            original = db.execute('SELECT id FROM local_albums').fetchone()[0]
            self.assertEqual(db.execute('SELECT processed FROM library_scan_grouping_edges').fetchone()[0], 1)
        self.assertEqual(group['retained_album_id'], original)
        self.assertEqual(group['continuity_reason_code'], 'MAXIMUM_TRACK_OVERLAP')
        self.assertIsNone(await self.retain(finish=True))
        self.assertEqual(self.context()['staging_state'], 'albums')
        with self.assertRaisesRegex(ValidationError, 'Exact isolated native continuity context'):
            await self.retain(finish=True)

    async def test_actual_isolated_loop_stops_before_unfenced_provisioning(self):
        with patch.object(self.store, 'provision_staged_grouping_groups', new=AsyncMock(side_effect=AssertionError('unfenced catalog provisioning'))) as provision:
            with self.assertRaisesRegex(ValidationError, 'beyond validated continuity'):
                await self.loop()
        provision.assert_not_awaited()
        self.assertEqual(self.context()['staging_state'], 'albums')
        self.assertIsNotNone(self.group()['retained_album_id'])

    async def test_checkpoint_resume_does_not_reassign_or_duplicate_continuity(self):
        self.assertEqual(await self.loop(checkpoint=AsyncMock(side_effect=[True, True, False, False])), 0)
        retained = self.group()
        self.assertEqual(self.context()['staging_state'], 'continuity')
        with self.assertRaisesRegex(ValidationError, 'beyond validated continuity'):
            await self.loop()
        self.assertEqual(self.group(), retained)
        self.assertEqual(self.context()['staging_state'], 'albums')

    async def test_unassigned_continuity_cannot_be_completed(self):
        with self.assertRaisesRegex(StaleRevisionError, 'assignment is incomplete'):
            await self.retain(finish=True)
        self.assertEqual(self.context()['staging_state'], 'continuity')
        self.assertIsNone(self.group()['retained_album_id'])

    async def test_late_context_revision_does_not_assign_identity(self):
        original = self.store.assign_isolated_grouping_continuity
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute('UPDATE library_scan_grouping_contexts SET row_revision=row_revision+1')
            return await original(*args, **kwargs)
        with patch.object(self.store, 'assign_isolated_grouping_continuity', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'continuity context revision changed'):
                await self.retain()
        self.assertIsNone(self.group()['retained_album_id'])

    async def test_late_policy_disagreement_does_not_assign_identity(self):
        original = self.store.assign_isolated_grouping_continuity
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute("UPDATE library_policy_state SET desired_policy_revision='continuity-policy-drift'")
            return await original(*args, **kwargs)
        with patch.object(self.store, 'assign_isolated_grouping_continuity', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'cohort changed'):
                await self.retain()
        with self.store._connect() as db:
            self.assertIsNone(db.execute('SELECT retained_album_id FROM library_scan_grouping_groups').fetchone()[0])

    async def test_wider_degree_or_wrong_overlap_never_forces_identity(self):
        for table, field, invalid, valid in [('library_scan_grouping_old_nodes', 'degree', 2, 1),
                                            ('library_scan_grouping_edges', 'overlap_count', 3, 2)]:
            with self.subTest(field=field):
                with self.store._connect() as db:
                    db.execute(f'UPDATE {table} SET {field}=?', (invalid,))
                with self.assertRaisesRegex(StaleRevisionError, 'continuity edges or nodes changed'):
                    await self.retain()
                self.assertIsNone(self.group()['retained_album_id'])
                with self.store._connect() as db:
                    db.execute(f'UPDATE {table} SET {field}=?', (valid,))

    async def test_postassignment_reason_corruption_rolls_back_retention_and_edge(self):
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_continuity AFTER UPDATE OF retained_album_id ON library_scan_grouping_groups BEGIN UPDATE library_scan_grouping_groups SET continuity_reason_code='foreign-reason'; END")
        with self.assertRaisesRegex(StaleRevisionError, 'summary or revision accumulators changed'):
            await self.retain()
        self.assertIsNone(self.group()['retained_album_id'])
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT processed FROM library_scan_grouping_edges').fetchone()[0], 0)

    async def test_postfinish_cursor_corruption_rolls_back_only_finish(self):
        self.assertEqual(await self.retain(), 1)
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_finish AFTER UPDATE OF staging_state ON library_scan_grouping_contexts WHEN NEW.staging_state='albums' BEGIN UPDATE library_scan_grouping_contexts SET queue_cursor='foreign-cursor'; END")
        with self.assertRaisesRegex(StaleRevisionError, 'not pending/running'):
            await self.retain(finish=True)
        self.assertEqual(self.context()['staging_state'], 'continuity')
        self.assertIsNone(self.context()['queue_cursor'])
        self.assertIsNotNone(self.group()['retained_album_id'])

    async def test_native_pause_prevents_identity_assignment(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_runs SET requested_control='pause'")
        with self.assertRaisesRegex(StaleRevisionError, 'not pending/running'):
            await self.retain()
        self.assertIsNone(self.group()['retained_album_id'])


class CohortProvisionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = CohortContinuityTests()
        await self.fixture.asyncSetUp()
        self.addCleanup(self.fixture.doCleanups)
        self.store = self.fixture.store
        self.assertEqual(await self.fixture.retain(), 1)
        await self.fixture.retain(finish=True)
        self.source = self.fixture.fixture.fixture.fixture.fixture.fixture.fixture
        with self.store._connect() as db:
            self.binding_id = db.execute('SELECT id FROM library_import_cohort_bindings').fetchone()[0]

    async def provision(self, *, finish=False):
        from services.native.import_cohort_grouping_candidate import provision_cohort_album_candidate
        return await provision_cohort_album_candidate(store=self.store,
            filesystem=self.source.staged.service._filesystem, resolver=self.source.staged.fixture.resolver,
            binding_id=self.binding_id, run_id='continuity-scan', root_id='root', directory='Artist/Album',
            now=6, max_total_bytes=10000000, timeout_seconds=5, finish=finish)

    def context(self):
        return self.fixture.context()

    def group(self):
        return self.fixture.group()

    def catalog(self):
        with self.store._connect() as db:
            return dict(db.execute('SELECT * FROM local_albums').fetchone()), db.execute('SELECT value FROM library_catalog_revision').fetchone()[0]

    async def loop(self, **kwargs):
        self.fixture.fixture.fixture.fixture.fixture.loop()._cohort_provision_candidate_enabled = True
        return await self.fixture.loop(**kwargs)

    async def test_native_provision_keeps_existing_album_artist_and_credit_then_finishes(self):
        before, _revision = self.catalog()
        with self.store._connect() as db:
            artists_before = [dict(row) for row in db.execute('SELECT * FROM local_artists ORDER BY id')]
        self.assertTrue(await self.provision())
        self.assertFalse(await self.provision())
        group = self.group()
        self.assertEqual(group['local_album_id'], before['id'])
        self.assertEqual(group['local_artist_id'], before['album_artist_id'])
        self.assertEqual(self.catalog()[0]['grouping_key'], group['grouping_key'])
        await self.provision(finish=True)
        self.assertEqual(self.context()['staging_state'], 'memberships')
        with self.assertRaisesRegex(ValidationError, 'Exact native album provisioning context'):
            await self.provision(finish=True)
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM local_albums').fetchone()[0], 1)
            self.assertEqual([dict(row) for row in db.execute('SELECT * FROM local_artists ORDER BY id')], artists_before)
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])

    async def test_actual_loop_never_resolves_artist_or_applies_unfenced_membership(self):
        with patch.object(self.store, 'resolve_or_create_local_artists', new=AsyncMock(side_effect=AssertionError('unfenced artist write'))) as artist:
            with patch.object(self.store, 'apply_staged_grouping_membership_page', new=AsyncMock(side_effect=AssertionError('unfenced membership'))) as members:
                with self.assertRaisesRegex(ValidationError, 'beyond validated provisioning'):
                    await self.loop()
        artist.assert_not_awaited()
        members.assert_not_awaited()
        self.assertEqual(self.context()['staging_state'], 'memberships')

    async def test_checkpoint_resume_does_not_duplicate_or_revise_provisioned_album(self):
        self.assertEqual(await self.loop(checkpoint=AsyncMock(side_effect=[True, True, False, False])), 0)
        catalog = self.catalog()
        self.assertEqual(self.context()['staging_state'], 'albums')
        with self.assertRaisesRegex(ValidationError, 'beyond validated provisioning'):
            await self.loop()
        self.assertEqual(self.catalog(), catalog)
        self.assertEqual(self.context()['staging_state'], 'memberships')

    async def test_incomplete_provisioning_cannot_advance_to_memberships(self):
        with self.assertRaisesRegex(StaleRevisionError, 'provisioning is incomplete'):
            await self.provision(finish=True)
        self.assertIsNone(self.group()['local_album_id'])
        self.assertEqual(self.context()['staging_state'], 'albums')

    async def test_foreign_album_application_is_refused_before_native_catalog_write(self):
        original = self.store.provision_staged_grouping_groups
        async def changed(run_id, root_id, directory, groups, **kwargs):
            groups = [dict(groups[0], local_album_id='foreign-album')]
            return await original(run_id, root_id, directory, groups, **kwargs)
        before = self.catalog()
        with patch.object(self.store, 'provision_staged_grouping_groups', side_effect=changed):
            with self.assertRaisesRegex(ValidationError, 'Exact unprovisioned owned album/artist'):
                await self.provision()
        self.assertEqual(self.catalog(), before)
        self.assertIsNone(self.group()['local_album_id'])

    async def test_late_context_revision_prevents_catalog_provision(self):
        original = self.store.provision_staged_grouping_groups
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute('UPDATE library_scan_grouping_contexts SET row_revision=row_revision+1')
            return await original(*args, **kwargs)
        before = self.catalog()
        with patch.object(self.store, 'provision_staged_grouping_groups', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'provisioning context revision changed'):
                await self.provision()
        self.assertEqual(self.catalog(), before)

    async def test_late_policy_disagreement_prevents_catalog_provision(self):
        original = self.store.provision_staged_grouping_groups
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute("UPDATE library_policy_state SET desired_policy_revision='provision-policy-drift'")
            return await original(*args, **kwargs)
        before = self.catalog()
        with patch.object(self.store, 'provision_staged_grouping_groups', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'cohort changed'):
                await self.provision()
        self.assertEqual(self.catalog(), before)

    async def test_changed_primary_credit_is_not_overwritten_as_automatic_repair(self):
        with self.store._connect() as db:
            db.execute("UPDATE local_album_artists SET credited_name='user-changed-credit'")
        with self.assertRaisesRegex(StaleRevisionError, 'album, artist or primary credit disagrees'):
            await self.provision()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT credited_name FROM local_album_artists').fetchone()[0], 'user-changed-credit')
            self.assertIsNone(db.execute('SELECT local_album_id FROM library_scan_grouping_groups').fetchone()[0])

    async def test_postwrite_album_corruption_rolls_back_catalog_and_group_ids(self):
        before = self.catalog()
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_provision AFTER UPDATE OF local_album_id ON library_scan_grouping_groups BEGIN UPDATE local_albums SET title='changed-by-provision-fixture'; END")
        with self.assertRaises(StaleRevisionError):
            await self.provision()
        self.assertEqual(self.catalog(), before)
        self.assertIsNone(self.group()['local_album_id'])

    async def test_native_pause_refuses_catalog_provision(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_runs SET requested_control='pause'")
        before = self.catalog()
        with self.assertRaisesRegex(StaleRevisionError, 'not pending/running'):
            await self.provision()
        self.assertEqual(self.catalog(), before)

    async def test_postfinish_cursor_corruption_keeps_provisioned_album_but_not_phase_advance(self):
        self.assertTrue(await self.provision())
        before = self.catalog()
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_provision_finish AFTER UPDATE OF staging_state ON library_scan_grouping_contexts WHEN NEW.staging_state='memberships' BEGIN UPDATE library_scan_grouping_contexts SET queue_cursor='foreign-cursor'; END")
        with self.assertRaisesRegex(StaleRevisionError, 'not pending/running'):
            await self.provision(finish=True)
        self.assertEqual(self.catalog(), before)
        self.assertEqual(self.context()['staging_state'], 'albums')
        self.assertIsNone(self.context()['queue_cursor'])

    async def test_resumed_provisioning_refuses_changed_album_key_without_overwriting_it(self):
        self.assertTrue(await self.provision())
        with self.store._connect() as db:
            db.execute("UPDATE local_albums SET grouping_key='user-changed-key'")
        with self.assertRaises(StaleRevisionError):
            await self.provision(finish=True)
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT grouping_key FROM local_albums').fetchone()[0], 'user-changed-key')
            self.assertEqual(db.execute('SELECT staging_state FROM library_scan_grouping_contexts').fetchone()[0], 'albums')


class CohortMembershipTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = CohortProvisionTests()
        await self.fixture.asyncSetUp()
        self.addCleanup(self.fixture.doCleanups)
        self.store = self.fixture.store
        await self.fixture.provision()
        await self.fixture.provision(finish=True)

    async def membership(self, *, limit=1):
        from services.native.import_cohort_grouping_candidate import apply_cohort_membership_candidate
        source = self.fixture.source
        return await apply_cohort_membership_candidate(store=self.store,
            filesystem=source.staged.service._filesystem, resolver=source.staged.fixture.resolver,
            binding_id=self.fixture.binding_id, run_id='continuity-scan', root_id='root', directory='Artist/Album',
            limit=limit, max_total_bytes=10000000, timeout_seconds=5)

    def rows(self):
        with self.store._connect() as db:
            return [dict(row) for row in db.execute('SELECT * FROM local_tracks ORDER BY id')]

    def legacy(self):
        with self.store._connect() as db:
            db.execute("UPDATE local_tracks SET membership_source='legacy_import'")

    async def loop(self, **kwargs):
        self.fixture.source.cohort_grouping._cohort_membership_candidate_enabled = True
        return await self.fixture.loop(**kwargs)

    async def test_native_pages_normalize_only_membership_and_preserve_complete_source_map(self):
        self.legacy()
        before = self.rows()
        catalog = self.fixture.catalog()
        self.assertFalse(await self.membership())
        expected = [dict(row) for row in before]
        expected[0].update(membership_source='automatic', row_revision=before[0]['row_revision'] + 1)
        self.assertEqual(self.rows(), expected)
        self.assertEqual(self.fixture.context()['application_cursor'], before[0]['id'])
        self.assertFalse(await self.membership())
        expected[1].update(membership_source='automatic', row_revision=before[1]['row_revision'] + 1)
        self.assertEqual(self.rows(), expected)
        self.assertTrue(await self.membership())
        self.assertEqual(self.fixture.context()['staging_state'], 'retirement')
        self.assertIsNone(self.fixture.context()['application_cursor'])
        self.assertEqual(self.fixture.catalog(), (catalog[0], catalog[1] + 2))
        with self.assertRaisesRegex(ValidationError, 'Exact bounded native membership page'):
            await self.membership()

    async def test_already_automatic_pages_do_not_revise_tracks_or_catalog(self):
        before, catalog = self.rows(), self.fixture.catalog()
        self.assertFalse(await self.membership(limit=2))
        self.assertTrue(await self.membership(limit=2))
        self.assertEqual(self.rows(), before)
        self.assertEqual(self.fixture.catalog(), catalog)

    async def test_actual_private_loop_stops_before_retirement_or_queue(self):
        self.legacy()
        with patch.object(self.store, 'retire_staged_grouping_albums', new=AsyncMock(side_effect=AssertionError('unfenced retirement'))) as retire:
            with self.assertRaisesRegex(ValidationError, 'beyond validated membership'):
                await self.loop()
        retire.assert_not_awaited()
        self.assertEqual(self.fixture.context()['staging_state'], 'retirement')
        self.assertTrue(all(row['membership_source'] == 'automatic' for row in self.rows()))

    async def test_checkpoint_resume_keeps_processed_members_without_double_revision(self):
        self.legacy()
        self.assertEqual(await self.loop(checkpoint=AsyncMock(side_effect=[True, True, False, False])), 0)
        first = self.rows()[0]
        self.assertEqual(first['membership_source'], 'automatic')
        self.assertEqual(self.fixture.context()['staging_state'], 'memberships')
        with self.assertRaisesRegex(ValidationError, 'beyond validated membership'):
            await self.loop()
        self.assertEqual(self.rows()[0], first)
        self.assertEqual(self.fixture.context()['staging_state'], 'retirement')

    async def test_unknown_cursor_cannot_skip_members(self):
        self.legacy()
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_grouping_contexts SET application_cursor='unknown-member'")
        before = self.rows()
        with self.assertRaisesRegex(StaleRevisionError, 'not a member boundary'):
            await self.membership()
        self.assertEqual(self.rows(), before)

    async def test_member_cursor_cannot_claim_unprocessed_legacy_prefix(self):
        self.legacy()
        with self.store._connect() as db:
            db.execute('UPDATE library_scan_grouping_contexts SET application_cursor=?', (self.rows()[-1]['id'],))
        with self.assertRaisesRegex(StaleRevisionError, 'processed prefix disagrees'):
            await self.membership()

    async def test_unlocked_manual_membership_is_not_automatically_overwritten(self):
        with self.store._connect() as db:
            db.execute("UPDATE local_tracks SET membership_source='manual',membership_locked=0")
        before = self.rows()
        with self.assertRaisesRegex(StaleRevisionError, 'membership source'):
            await self.membership()
        self.assertEqual(self.rows(), before)

    async def test_late_revision_refuses_native_membership_write(self):
        self.legacy()
        original = self.store.apply_staged_grouping_membership_page
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute('UPDATE library_scan_grouping_contexts SET row_revision=row_revision+1')
            return await original(*args, **kwargs)
        before = self.rows()
        with patch.object(self.store, 'apply_staged_grouping_membership_page', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'membership context revision changed'):
                await self.membership()
        self.assertEqual(self.rows(), before)

    async def test_late_policy_drift_refuses_native_membership_write(self):
        self.legacy()
        original = self.store.apply_staged_grouping_membership_page
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute("UPDATE library_policy_state SET desired_policy_revision='membership-policy-drift'")
            return await original(*args, **kwargs)
        before = self.rows()
        with patch.object(self.store, 'apply_staged_grouping_membership_page', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'cohort changed'):
                await self.membership()
        self.assertEqual(self.rows(), before)

    async def test_postwrite_source_corruption_rolls_back_page_cursor_and_catalog(self):
        self.legacy()
        before, catalog, context = self.rows(), self.fixture.catalog(), self.fixture.context()
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_membership AFTER UPDATE OF membership_source ON local_tracks BEGIN UPDATE local_tracks SET source_path='changed-by-membership-fixture' WHERE id=NEW.id; END")
        with self.assertRaises(StaleRevisionError):
            await self.membership()
        self.assertEqual(self.rows(), before)
        self.assertEqual(self.fixture.catalog(), catalog)
        self.assertEqual(self.fixture.context(), context)

    async def test_postexhaustion_false_queue_advance_rolls_back(self):
        await self.membership(limit=2)
        context = self.fixture.context()
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_membership_finish AFTER UPDATE OF staging_state ON library_scan_grouping_contexts WHEN NEW.staging_state='retirement' BEGIN UPDATE library_scan_grouping_contexts SET staging_state='queue'; END")
        with self.assertRaises(StaleRevisionError):
            await self.membership()
        self.assertEqual(self.fixture.context(), context)

    async def test_native_pause_refuses_membership_and_preserves_rows(self):
        self.legacy()
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_runs SET requested_control='pause'")
        before = self.rows()
        with self.assertRaisesRegex(StaleRevisionError, 'not pending/running'):
            await self.membership()
        self.assertEqual(self.rows(), before)

    async def test_locked_member_is_never_normalized_or_unlocked(self):
        self.legacy()
        with self.store._connect() as db:
            db.execute('UPDATE local_tracks SET membership_locked=1')
        before = self.rows()
        with self.assertRaises(StaleRevisionError):
            await self.membership()
        self.assertEqual(self.rows(), before)

    async def test_postwrite_extra_track_revision_rolls_back_even_when_binding_fields_match(self):
        self.legacy()
        before, catalog, context = self.rows(), self.fixture.catalog(), self.fixture.context()
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_membership_revision AFTER UPDATE OF membership_source ON local_tracks BEGIN UPDATE local_tracks SET row_revision=row_revision+7 WHERE id=NEW.id; END")
        with self.assertRaisesRegex(StaleRevisionError, 'membership page, source, catalog or revision changed'):
            await self.membership()
        self.assertEqual(self.rows(), before)
        self.assertEqual(self.fixture.catalog(), catalog)
        self.assertEqual(self.fixture.context(), context)


class CohortRetirementTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = CohortMembershipTests()
        await self.fixture.asyncSetUp()
        self.addCleanup(self.fixture.doCleanups)
        self.store = self.fixture.store
        self.assertFalse(await self.fixture.membership(limit=2))
        self.assertTrue(await self.fixture.membership(limit=2))

    async def retire(self):
        from services.native.import_cohort_grouping_candidate import retire_cohort_album_candidate
        source = self.fixture.fixture.source
        return await retire_cohort_album_candidate(store=self.store,
            filesystem=source.staged.service._filesystem, resolver=source.staged.fixture.resolver,
            binding_id=self.fixture.fixture.binding_id, run_id='continuity-scan', root_id='root', directory='Artist/Album',
            limit=1, now=7, max_total_bytes=10000000, timeout_seconds=5)

    def context(self):
        return self.fixture.fixture.context()

    def catalog(self):
        with self.store._connect() as db:
            return {table: [dict(row) for row in db.execute(f'SELECT * FROM {table} ORDER BY rowid')]
                for table in ('local_tracks', 'local_albums', 'local_artists', 'local_album_artists',
                              'local_album_aliases', 'library_catalog_revision')}

    async def loop(self, **kwargs):
        self.fixture.fixture.source.cohort_grouping._cohort_retirement_candidate_enabled = True
        return await self.fixture.loop(**kwargs)

    async def test_native_retained_album_page_and_eof_change_no_catalog_or_aliases(self):
        before = self.catalog()
        album_id = before['local_albums'][0]['id']
        self.assertFalse(await self.retire())
        self.assertEqual(self.context()['application_cursor'], album_id)
        self.assertEqual(self.catalog(), before)
        self.assertTrue(await self.retire())
        self.assertEqual(self.context()['staging_state'], 'queue')
        self.assertIsNone(self.context()['application_cursor'])
        self.assertEqual(self.catalog(), before)
        with self.assertRaisesRegex(ValidationError, 'Exact bounded native retained-album retirement page'):
            await self.retire()

    async def test_actual_private_loop_stops_before_queue_read_or_completion(self):
        before = self.catalog()
        with patch.object(self.store, 'get_staged_grouping_queue_page', new=AsyncMock(side_effect=AssertionError('unfenced queue'))) as queue:
            with self.assertRaisesRegex(ValidationError, 'beyond validated retirement'):
                await self.loop()
        queue.assert_not_awaited()
        self.assertEqual(self.context()['staging_state'], 'queue')
        self.assertEqual(self.context()['state'], 'pending')
        self.assertEqual(self.catalog(), before)

    async def test_checkpoint_resume_does_not_reprocess_retained_album(self):
        before = self.catalog()
        self.assertEqual(await self.loop(checkpoint=AsyncMock(side_effect=[True, True, False, False])), 0)
        context = self.context()
        self.assertIsNotNone(context['application_cursor'])
        with self.assertRaisesRegex(ValidationError, 'beyond validated retirement'):
            await self.loop()
        self.assertEqual(self.context()['row_revision'], context['row_revision'] + 1)
        self.assertEqual(self.context()['staging_state'], 'queue')
        self.assertEqual(self.catalog(), before)

    async def test_unknown_album_cursor_cannot_skip_retained_album(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_grouping_contexts SET application_cursor='foreign-album'")
        before = self.catalog()
        with self.assertRaisesRegex(StaleRevisionError, 'retained album boundary'):
            await self.retire()
        self.assertEqual(self.catalog(), before)

    async def test_membership_prefix_must_be_complete_before_retirement(self):
        self.fixture.legacy()
        before = self.catalog()
        with self.assertRaisesRegex(StaleRevisionError, 'processed prefix disagrees'):
            await self.retire()
        self.assertEqual(self.catalog(), before)

    async def test_late_revision_refuses_native_retirement(self):
        original = self.store.retire_staged_grouping_albums
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute('UPDATE library_scan_grouping_contexts SET row_revision=row_revision+1')
            return await original(*args, **kwargs)
        before = self.catalog()
        with patch.object(self.store, 'retire_staged_grouping_albums', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'retirement context revision changed'):
                await self.retire()
        self.assertEqual(self.catalog(), before)

    async def test_late_policy_drift_refuses_native_retirement(self):
        original = self.store.retire_staged_grouping_albums
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute("UPDATE library_policy_state SET desired_policy_revision='retirement-policy-drift'")
            return await original(*args, **kwargs)
        before = self.catalog()
        with patch.object(self.store, 'retire_staged_grouping_albums', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'cohort changed'):
                await self.retire()
        self.assertEqual(self.catalog(), before)

    async def test_native_pause_prevents_retirement(self):
        with self.store._connect() as db:
            db.execute("UPDATE library_scan_runs SET requested_control='pause'")
        before = self.catalog()
        with self.assertRaisesRegex(StaleRevisionError, 'not pending/running'):
            await self.retire()
        self.assertEqual(self.catalog(), before)

    async def test_postpage_album_revision_corruption_rolls_back_cursor(self):
        before, context = self.catalog(), self.context()
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_retirement_album AFTER UPDATE OF application_cursor ON library_scan_grouping_contexts BEGIN UPDATE local_albums SET row_revision=row_revision+7; END")
        with self.assertRaisesRegex(StaleRevisionError, 'retirement changed catalog'):
            await self.retire()
        self.assertEqual(self.catalog(), before)
        self.assertEqual(self.context(), context)

    async def test_postpage_alias_creation_is_not_silently_accepted(self):
        before, context = self.catalog(), self.context()
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_retirement_alias AFTER UPDATE OF application_cursor ON library_scan_grouping_contexts BEGIN INSERT INTO local_album_aliases(alias,local_album_id,kind,created_at) SELECT 'fixture-alias',id,'merged_album',7 FROM local_albums; END")
        with self.assertRaisesRegex(StaleRevisionError, 'retirement changed catalog'):
            await self.retire()
        self.assertEqual(self.catalog(), before)
        self.assertEqual(self.context(), context)

    async def test_postexhaustion_false_completion_rolls_back(self):
        await self.retire()
        before, context = self.catalog(), self.context()
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_retirement_finish AFTER UPDATE OF staging_state ON library_scan_grouping_contexts WHEN NEW.staging_state='queue' BEGIN UPDATE library_scan_grouping_contexts SET state='completed'; END")
        with self.assertRaises(StaleRevisionError):
            await self.retire()
        self.assertEqual(self.catalog(), before)
        self.assertEqual(self.context(), context)

    async def test_changed_recording_map_is_not_repaired_or_retired(self):
        with self.store._connect() as db:
            db.execute("UPDATE local_tracks SET embedded_recording_mbid='different-recording'")
        before = self.catalog()
        with self.assertRaises(StaleRevisionError):
            await self.retire()
        self.assertEqual(self.catalog(), before)


class CohortPagedCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = CohortRetirementTests()
        await self.fixture.asyncSetUp()
        self.addCleanup(self.fixture.doCleanups)
        self.store = self.fixture.store
        await self.fixture.retire()
        await self.fixture.retire()
        self.source = self.fixture.fixture.fixture.source

    async def complete(self):
        from services.native.import_cohort_grouping_candidate import complete_paged_cohort_candidate
        return await complete_paged_cohort_candidate(store=self.store,
            filesystem=self.source.staged.service._filesystem, resolver=self.source.staged.fixture.resolver,
            binding_id=self.fixture.fixture.fixture.binding_id, run_id='continuity-scan', root_id='root',
            directory='Artist/Album', now=8, max_total_bytes=10000000, timeout_seconds=5)

    def state(self):
        self.source.assert_current_binding('Paged completion must preserve full source binding')
        with self.store._connect() as db:
            return dict(db.execute('SELECT * FROM library_scan_grouping_contexts').fetchone()), [dict(row) for row in db.execute('SELECT * FROM library_identification_jobs')], db.execute('SELECT identification_enqueued_count FROM library_scan_runs').fetchone()[0]

    def assert_pending(self):
        context, jobs, count = self.state()
        self.assertEqual((context['state'], context['staging_state'], context['queue_cursor']), ('pending', 'queue', None))
        self.assertEqual((jobs, count), ([], 0))

    async def loop(self, **kwargs):
        self.source.cohort_grouping._cohort_queue_candidate_enabled = True
        return await self.fixture.loop(**kwargs)

    async def test_native_job_cursor_counter_and_completion_commit_once(self):
        before = self.fixture.catalog()
        result = await self.complete()
        self.assertTrue(result[0][1])
        context, jobs, count = self.state()
        self.assertEqual((context['state'], context['staging_state'], len(jobs), count), ('completed', 'completed', 1, 1))
        self.assertEqual(self.fixture.catalog(), before)
        with self.assertRaisesRegex(StaleRevisionError, 'not pending/running'):
            await self.complete()
        self.assertEqual(len(self.state()[1]), 1)

    async def test_actual_private_loop_never_calls_unfenced_completion(self):
        with patch.object(self.store, 'complete_grouping_context', new=AsyncMock(side_effect=AssertionError('unfenced completion'))) as complete:
            self.assertEqual(await self.loop(), 1)
        complete.assert_not_awaited()
        self.assertEqual(self.state()[0]['state'], 'completed')

    async def test_checkpoint_pause_keeps_queue_pending(self):
        self.assertEqual(await self.loop(checkpoint=AsyncMock(return_value=False)), 0)
        self.assert_pending()

    async def test_existing_native_job_is_reused_without_counter_credit(self):
        from services.native.identification_queue_service import IdentificationQueueService
        page = await self.store.get_staged_grouping_queue_page('continuity-scan', 'root', 'Artist/Album', limit=1)
        row = page[0]
        existing = await IdentificationQueueService(self.store).enqueue_album(row['local_album_id'], input_revision=row['input_revision'], now=7)
        self.assertEqual(await self.complete(), [(existing, False)])
        self.assertEqual((len(self.state()[1]), self.state()[2]), (1, 0))

    async def test_intentional_exclusion_cannot_be_completed_or_resurrected(self):
        album = self.fixture.catalog()['local_albums'][0]['id']
        with self.store._connect() as db:
            db.execute("INSERT INTO library_identification_reviews(id,local_album_id,state,reason_code,input_revision,created_at,updated_at) VALUES('excluded-paged',?,'excluded','USER_EXCLUDED','older',1,1)", (album,))
        with self.assertRaisesRegex(StaleRevisionError, 'identification admission refused'):
            await self.complete()
        self.assert_pending()

    async def test_late_context_revision_prevents_job_and_completion(self):
        original = self.store.enqueue_identification_job_results
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute('UPDATE library_scan_grouping_contexts SET row_revision=row_revision+1')
            return await original(*args, **kwargs)
        with patch.object(self.store, 'enqueue_identification_job_results', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'completion context revision changed'):
                await self.complete()
        self.assert_pending()

    async def test_late_policy_drift_prevents_job_and_completion(self):
        original = self.store.enqueue_identification_job_results
        async def changed(*args, **kwargs):
            with self.store._connect() as db:
                db.execute("UPDATE library_policy_state SET desired_policy_revision='paged-completion-drift'")
            return await original(*args, **kwargs)
        with patch.object(self.store, 'enqueue_identification_job_results', side_effect=changed):
            with self.assertRaisesRegex(StaleRevisionError, 'cohort changed'):
                await self.complete()
        # Deliberate policy drift invalidates the binding; do not assert it is
        # still current just to inspect the correctly untouched queue state.
        with self.store._connect() as db:
            row = db.execute('SELECT state,staging_state,queue_cursor FROM library_scan_grouping_contexts').fetchone()
            self.assertEqual(tuple(row), ('pending', 'queue', None))
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_identification_jobs').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT identification_enqueued_count FROM library_scan_runs').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT desired_policy_revision FROM library_policy_state').fetchone()[0], 'paged-completion-drift')

    async def test_changed_native_job_revision_is_refused(self):
        original = self.store.enqueue_identification_job_results
        async def changed(jobs, **kwargs):
            return await original([msgspec.structs.replace(jobs[0], input_revision='wrong-revision')], **kwargs)
        with patch.object(self.store, 'enqueue_identification_job_results', side_effect=changed):
            with self.assertRaisesRegex(ValidationError, 'Exact fresh native paged'):
                await self.complete()
        self.assert_pending()

    async def test_false_empty_native_page_cannot_mark_context_complete(self):
        with patch.object(self.store, 'get_staged_grouping_queue_page', new=AsyncMock(return_value=[])):
            with self.assertRaisesRegex(StaleRevisionError, 'complete native cohort queue page'):
                await self.complete()
        self.assert_pending()

    async def test_late_completion_failure_rolls_back_job_cursor_and_counter(self):
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER fail_paged_completion BEFORE UPDATE OF state ON library_scan_grouping_contexts WHEN NEW.state='completed' BEGIN SELECT RAISE(ABORT,'paged completion fixture'); END")
        import sqlite3
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'paged completion fixture'):
            await self.complete()
        self.assert_pending()

    async def test_postwrite_job_corruption_rolls_back_everything(self):
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_paged_job BEFORE UPDATE OF state ON library_scan_grouping_contexts WHEN NEW.state='completed' BEGIN UPDATE library_identification_jobs SET input_revision='corrupt-job'; END")
        with self.assertRaisesRegex(StaleRevisionError, 'completion job, catalog, context or counters changed'):
            await self.complete()
        self.assert_pending()

    async def test_postwrite_counter_corruption_rolls_back_everything(self):
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_paged_counter BEFORE UPDATE OF state ON library_scan_grouping_contexts WHEN NEW.state='completed' BEGIN UPDATE library_scan_runs SET identification_enqueued_count=identification_enqueued_count+7; END")
        with self.assertRaisesRegex(StaleRevisionError, 'completion job, catalog, context or counters changed'):
            await self.complete()
        self.assert_pending()

    async def test_postwrite_priority_corruption_rolls_back_everything(self):
        with self.store._connect() as db:
            db.execute("CREATE TRIGGER corrupt_paged_priority BEFORE UPDATE OF state ON library_scan_grouping_contexts WHEN NEW.state='completed' BEGIN UPDATE library_identification_jobs SET priority=99; END")
        with self.assertRaisesRegex(StaleRevisionError, 'completion job, catalog, context or counters changed'):
            await self.complete()
        self.assert_pending()

    async def test_album_retired_after_previous_phase_cannot_be_queued(self):
        with self.store._connect() as db:
            db.execute('UPDATE local_albums SET retired_into_album_id=id')
        # The immutable binding reader rejects retired albums even earlier than
        # the later owned-album fence. Preserve that stronger first refusal.
        with self.assertRaisesRegex(StaleRevisionError, 'cohort changed'):
            await self.complete()
        with self.store._connect() as db:
            self.assertEqual(db.execute('SELECT state,staging_state,queue_cursor FROM library_scan_grouping_contexts').fetchone()[:], ('pending', 'queue', None))
            self.assertEqual(db.execute('SELECT COUNT(*) FROM library_identification_jobs').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT identification_enqueued_count FROM library_scan_runs').fetchone()[0], 0)
            album = db.execute('SELECT id,retired_into_album_id FROM local_albums').fetchone()
            self.assertEqual(album['retired_into_album_id'], album['id'])
