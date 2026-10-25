"""Private small grouping-loop candidates; no index-loop or paged rollout."""
from core.exceptions import StaleRevisionError, ValidationError
from infrastructure.persistence.import_cohort_scan import require_current_cohort
from models.identification import ExistingAlbumMembership, GroupingApplication
from services.native.import_cohort_scan_candidate import observed_cohort_candidate, settle_cohort_write_candidate
from services.native.local_album_grouper import LocalAlbumGrouper
from services.native.local_album_grouping_service import grouping_track_from_row


async def commit_policy_transition_cohort_candidate(*, store, filesystem, resolver_getter,
                                                     transition_lock, updated_at):
    """Explicit PRIVATE commit only, not wired into settings saves or recovery."""
    from infrastructure.persistence.import_cohort_binding import capture_cohort_policy_transition_candidate
    if filesystem is None or transition_lock is None:
        raise ValidationError('Native transition lock and root coordinator required.')
    resolver = resolver_getter()
    def discover(connection):
        if not connection.in_transaction:
            connection.execute('BEGIN')
        return capture_cohort_policy_transition_candidate(connection, resolver.policy_revision)
    seal = await store._read(discover)
    # Match native publication: root lease BEFORE the policy lock. Discovery
    # is optimistic; SQL rechecks the whole prepared row and active binding set.
    async with filesystem.write_many(seal.root_ids):
        async with transition_lock:
            if resolver_getter().policy_revision != seal.configured_policy_revision:
                raise StaleRevisionError('Configured cohort transition changed under root lease.')
            return await settle_cohort_write_candidate(store.commit_policy_transition(
                proposed_policy_revision=seal.configured_policy_revision, updated_at=updated_at,
                _cohort_policy_transition_candidate=seal))


async def invalidate_policy_cohort_candidate(*, store, filesystem, resolver, binding_id, invalidated_at):
    import json
    from infrastructure.persistence.import_cohort_binding import CohortPolicyInvalidationCandidate
    def discover(connection):
        row = connection.execute('SELECT payload_json,payload_sha256 FROM library_import_cohort_bindings WHERE id=?', (binding_id,)).fetchone()
        if row is None:
            raise StaleRevisionError('Native cohort invalidation binding missing.')
        return json.loads(row['payload_json']), row['payload_sha256']
    payload, digest = await store._read(discover)
    roots = [root for root in resolver.settings.library_roots if root.id == payload['root_id']]
    if (len(roots) != 1 or roots[0].path != payload['root_path'] or roots[0].policy != 'automatic' or filesystem is None):
        raise ValidationError('Configured native automatic root required for invalidation.')
    seal = CohortPolicyInvalidationCandidate(binding_id, digest, roots[0].id, roots[0].path, resolver.policy_revision)
    async with filesystem.write_many([roots[0].id]):
        return await settle_cohort_write_candidate(store._invalidate_cohort_policy_candidate(seal, invalidated_at=invalidated_at))


async def complete_paged_cohort_candidate(*, store, filesystem, resolver, binding_id,
                                         run_id, root_id, directory, now,
                                         max_total_bytes, timeout_seconds):
    from infrastructure.persistence.import_cohort_paging import validate_token_state
    from services.native.identification_queue_service import IdentificationQueueService
    async with observed_cohort_candidate(store=store, filesystem=filesystem, resolver=resolver,
            binding_id=binding_id, max_total_bytes=max_total_bytes, timeout_seconds=timeout_seconds) as admission:
        if (root_id, directory) != (admission.root_id, admission.album_parent):
            raise StaleRevisionError('Native paged completion context changed.')
        def read(connection):
            if not connection.in_transaction:
                connection.execute('BEGIN')
            context, _, _, token = validate_token_state(connection, admission, run_id, completion=True)
            return context['row_revision'], token
        revision, token = await store._read(read)
        page = await store.get_staged_grouping_queue_page(run_id, root_id, directory, limit=1)
        if len(page) != 1 or page[0].get('grouping_token') != token or page[0].get('local_album_id') is None:
            raise StaleRevisionError('Exact complete native cohort queue page required.')
        row = page[0]
        return await settle_cohort_write_candidate(IdentificationQueueService(store).enqueue_albums_with_disposition(
            [(row['local_album_id'], row['input_revision'], row['trigger'])], now=now,
            scan_run_id=run_id, grouping_context=(root_id, directory), queue_cursor=token, background=True,
            _cohort_paged_completion_candidate=admission, _cohort_context_revision=revision))


async def retire_cohort_album_candidate(*, store, filesystem, resolver, binding_id,
                                       run_id, root_id, directory, limit, now,
                                       max_total_bytes, timeout_seconds):
    from infrastructure.persistence.import_cohort_paging import validate_token_state
    async with observed_cohort_candidate(store=store, filesystem=filesystem, resolver=resolver,
            binding_id=binding_id, max_total_bytes=max_total_bytes, timeout_seconds=timeout_seconds) as admission:
        if (root_id, directory) != (admission.root_id, admission.album_parent):
            raise StaleRevisionError('Native retirement context changed.')
        def read(connection):
            if not connection.in_transaction:
                connection.execute('BEGIN')
            context, _, _, _ = validate_token_state(connection, admission, run_id, retirement=True)
            return context['row_revision']
        revision = await store._read(read)
        return await settle_cohort_write_candidate(store.retire_staged_grouping_albums(
            run_id, root_id, directory, limit=limit, now=now,
            _cohort_retirement_candidate=admission, _cohort_context_revision=revision))


async def apply_cohort_membership_candidate(*, store, filesystem, resolver, binding_id,
                                           run_id, root_id, directory, limit,
                                           max_total_bytes, timeout_seconds):
    from infrastructure.persistence.import_cohort_paging import validate_token_state
    async with observed_cohort_candidate(store=store, filesystem=filesystem, resolver=resolver,
            binding_id=binding_id, max_total_bytes=max_total_bytes, timeout_seconds=timeout_seconds) as admission:
        if (root_id, directory) != (admission.root_id, admission.album_parent):
            raise StaleRevisionError('Native membership context changed.')
        def read(connection):
            if not connection.in_transaction:
                connection.execute('BEGIN')
            context, _, _, _ = validate_token_state(connection, admission, run_id, membership=True)
            return context['row_revision']
        revision = await store._read(read)
        return await settle_cohort_write_candidate(store.apply_staged_grouping_membership_page(
            run_id, root_id, directory, limit=limit,
            _cohort_membership_candidate=admission, _cohort_context_revision=revision))


async def provision_cohort_album_candidate(*, store, filesystem, resolver, binding_id,
                                           run_id, root_id, directory, now,
                                           max_total_bytes, timeout_seconds, finish=False):
    from infrastructure.persistence.import_cohort_paging import validate_token_state
    async with observed_cohort_candidate(store=store, filesystem=filesystem, resolver=resolver,
            binding_id=binding_id, max_total_bytes=max_total_bytes, timeout_seconds=timeout_seconds) as admission:
        if (root_id, directory) != (admission.root_id, admission.album_parent):
            raise StaleRevisionError('Native provisioning context changed.')
        def read(connection):
            if not connection.in_transaction:
                connection.execute('BEGIN')
            context, _, _, token = validate_token_state(connection, admission, run_id, provision=True)
            if context['staging_state'] != 'albums':
                raise ValidationError('Exact native album provisioning context required.')
            group = dict(connection.execute('SELECT * FROM library_scan_grouping_groups WHERE run_id=? AND root_id=? AND relative_directory=? AND grouping_token=?',
                (run_id, root_id, directory, token)).fetchone())
            album = connection.execute('SELECT album_artist_id FROM local_albums WHERE id=?', (group['retained_album_id'],)).fetchone()
            application = dict(group, local_album_id=group['retained_album_id'], local_artist_id=album['album_artist_id'])
            return context['row_revision'], group['local_album_id'] is not None, application
        revision, provisioned, application = await store._read(read)
        arguments = dict(_cohort_provision_candidate=admission, _cohort_context_revision=revision)
        if finish:
            await settle_cohort_write_candidate(store.finish_grouping_album_provisioning(run_id, root_id, directory, **arguments))
            return None
        if provisioned:
            return False
        await settle_cohort_write_candidate(store.provision_staged_grouping_groups(run_id, root_id, directory,
            [application], now=now, **arguments))
        return True


async def retain_cohort_continuity_candidate(*, store, filesystem, resolver, binding_id,
                                            run_id, root_id, directory, limit,
                                            max_total_bytes, timeout_seconds, finish=False):
    if not isinstance(limit, int) or not 0 < limit <= 500:
        raise ValidationError('Exact isolated native continuity context required.')
    async with observed_cohort_candidate(store=store, filesystem=filesystem, resolver=resolver,
            binding_id=binding_id, max_total_bytes=max_total_bytes, timeout_seconds=timeout_seconds) as admission:
        if (root_id, directory) != (admission.root_id, admission.album_parent):
            raise StaleRevisionError('Native continuity context changed.')
        def revision(connection):
            row = connection.execute('SELECT row_revision FROM library_scan_grouping_contexts WHERE run_id=? AND root_id=? AND relative_directory=?',
                (run_id, root_id, directory)).fetchone()
            if row is None:
                raise StaleRevisionError('Native continuity context is missing.')
            return row['row_revision']
        captured = await store._read(revision)
        arguments = dict(_cohort_continuity_candidate=admission, _cohort_context_revision=captured)
        if finish:
            return await settle_cohort_write_candidate(store.finish_grouping_continuity(run_id, root_id, directory, **arguments))
        return await settle_cohort_write_candidate(store.assign_isolated_grouping_continuity(run_id, root_id, directory,
            limit=limit, **arguments))


async def summarize_cohort_page_candidate(*, store, filesystem, resolver, binding_id,
                                          run_id, root_id, directory, limit,
                                          max_total_bytes, timeout_seconds):
    if not isinstance(limit, int) or not 0 < limit <= 500:
        raise ValidationError('Exact bounded native summary page required.')
    async with observed_cohort_candidate(store=store, filesystem=filesystem, resolver=resolver,
            binding_id=binding_id, max_total_bytes=max_total_bytes, timeout_seconds=timeout_seconds) as admission:
        if (root_id, directory) != (admission.root_id, admission.album_parent):
            raise StaleRevisionError('Native summary context changed.')
        def revision(connection):
            row = connection.execute('SELECT row_revision FROM library_scan_grouping_contexts WHERE run_id=? AND root_id=? AND relative_directory=?',
                (run_id, root_id, directory)).fetchone()
            if row is None:
                raise StaleRevisionError('Native summary context is missing.')
            return row['row_revision']
        captured = await store._read(revision)
        return await settle_cohort_write_candidate(store.stage_grouping_summary_page(run_id, root_id, directory,
            limit=limit, _cohort_summary_candidate=admission, _cohort_context_revision=captured))


async def prepare_cohort_token_page_candidate(*, store, filesystem, resolver, binding_id,
                                              run_id, root_id, directory, limit,
                                              max_total_bytes, timeout_seconds):
    if not isinstance(limit, int) or not 0 < limit <= 500:
        raise ValidationError('Exact bounded native token page required.')
    async with observed_cohort_candidate(store=store, filesystem=filesystem, resolver=resolver,
            binding_id=binding_id, max_total_bytes=max_total_bytes, timeout_seconds=timeout_seconds) as admission:
        if (root_id, directory) != (admission.root_id, admission.album_parent):
            raise StaleRevisionError('Native token context changed.')
        def revision(connection):
            row = connection.execute('SELECT row_revision FROM library_scan_grouping_contexts WHERE run_id=? AND root_id=? AND relative_directory=?',
                (run_id, root_id, directory)).fetchone()
            if row is None:
                raise StaleRevisionError('Native token context is missing.')
            return row['row_revision']
        captured = await store._read(revision)
        return await settle_cohort_write_candidate(store.prepare_staged_grouping_tokens(run_id, root_id, directory,
            limit=limit, _cohort_token_candidate=admission, _cohort_context_revision=captured))


async def stage_cohort_track_page_candidate(*, store, filesystem, resolver, binding_id,
                                           run_id, root_id, directory, limit,
                                           max_total_bytes, timeout_seconds):
    from services.native.local_album_grouping_service import LocalAlbumGroupingService
    if not isinstance(limit, int) or not 0 < limit <= 500:
        raise ValidationError('Exact bounded native cohort track page required.')
    async with observed_cohort_candidate(store=store, filesystem=filesystem, resolver=resolver,
            binding_id=binding_id, max_total_bytes=max_total_bytes, timeout_seconds=timeout_seconds) as admission:
        if (root_id, directory) != (admission.root_id, admission.album_parent):
            raise StaleRevisionError('Native cohort page context changed.')
        def context_revision(connection):
            row = connection.execute('SELECT row_revision FROM library_scan_grouping_contexts '
                'WHERE run_id=? AND root_id=? AND relative_directory=?', (run_id, root_id, directory)).fetchone()
            if row is None:
                raise StaleRevisionError('Native cohort page context is missing.')
            return row['row_revision']
        revision = await store._read(context_revision)
        rows, cursor = await store.get_grouping_context_track_page(run_id, root_id, directory, limit=limit)
        # All rows belong to the freshly admitted exclusive parent context;
        # no directory regex may silently discard bound medium tracks.
        evidence = [LocalAlbumGroupingService._grouping_evidence(row, directory) for row in rows]
        await settle_cohort_write_candidate(store.stage_grouping_track_page(run_id, root_id, directory,
            evidence, next_cursor=cursor, exhausted=not rows,
            _cohort_paged_candidate=admission, _cohort_page_limit=limit, _cohort_context_revision=revision))
        return not rows


async def apply_cohort_grouping_candidate(*, store, filesystem, resolver, binding_id,
                                         run_id, now, max_total_bytes, timeout_seconds, grouper=None):
    async with observed_cohort_candidate(store=store, filesystem=filesystem, resolver=resolver,
            binding_id=binding_id, max_total_bytes=max_total_bytes, timeout_seconds=timeout_seconds) as admission:
        def read(connection):
            if not connection.in_transaction:
                connection.execute('BEGIN')
            payload = require_current_cohort(connection, admission)
            rows = connection.execute('SELECT t.*, a.created_at AS album_created_at, '
                'a.row_revision AS album_row_revision, a.album_artist_id AS cohort_artist_id, '
                'a.album_artist_name AS cohort_artist_name FROM local_tracks t '
                'JOIN library_import_cohort_members m ON m.local_track_id=t.id '
                'JOIN local_albums a ON a.id=t.local_album_id WHERE m.binding_id=? ORDER BY t.id',
                (binding_id,)).fetchall()
            return payload, [dict(row) for row in rows]
        payload, rows = await store._read(read)
        if {row['id'] for row in rows} != set(dict(admission.fingerprints)):
            raise StaleRevisionError('Whole cohort grouping rows changed.')
        groups = (grouper or LocalAlbumGrouper()).group([grouping_track_from_row(row) for row in rows],
            existing=[ExistingAlbumMembership(local_album_id=payload['album_id'],
                track_ids=[row['id'] for row in rows], created_at=rows[0]['album_created_at'])],
            _cohort_directories_candidate={row['id']: admission.album_parent for row in rows})
        if (len(groups) != 1 or groups[0].retained_album_id != payload['album_id']
                or groups[0].album_artist_name != rows[0]['cohort_artist_name']):
            raise ValidationError('Native grouper did not preserve one exact cohort.')
        # Application alone stays pending. The separate completion candidate
        # atomically admits the native queue job and completes this small context.
        return await settle_cohort_write_candidate(store.apply_grouping_context(run_id, admission.root_id, admission.album_parent,
            [GroupingApplication(group=groups[0], local_album_id=payload['album_id'],
                                 local_artist_id=rows[0]['cohort_artist_id'])],
            now=now, _cohort_grouping_candidate=admission))


async def complete_cohort_grouping_candidate(*, store, filesystem, resolver, binding_id,
                                            run_id, now, max_total_bytes, timeout_seconds,
                                            checkpoint=None):
    from services.native.identification_queue_service import IdentificationQueueService
    from services.native.identification_revisions import album_input_revisions
    if checkpoint is not None and not await checkpoint(run_id, resolver.policy_revision):
        return None
    async with observed_cohort_candidate(store=store, filesystem=filesystem, resolver=resolver,
            binding_id=binding_id, max_total_bytes=max_total_bytes, timeout_seconds=timeout_seconds) as admission:
        def read(connection):
            if not connection.in_transaction:
                connection.execute('BEGIN')
            payload = require_current_cohort(connection, admission)
            rows = [dict(row) for row in connection.execute('SELECT * FROM local_tracks WHERE local_album_id=?',
                                                          (payload['album_id'],)).fetchall()]
            return payload, rows
        payload, rows = await store._read(read)
        if checkpoint is not None and not await checkpoint(run_id, resolver.policy_revision):
            return None
        revision = ':'.join(album_input_revisions(rows))
        return await settle_cohort_write_candidate(IdentificationQueueService(store).enqueue_albums_with_disposition(
            [(payload['album_id'], revision, 'automatic')], now=now, scan_run_id=run_id,
            grouping_context=(admission.root_id, admission.album_parent), queue_cursor=payload['album_id'],
            background=True, _cohort_completion_candidate=admission))
