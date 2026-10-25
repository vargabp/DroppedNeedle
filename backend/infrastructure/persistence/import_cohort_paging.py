"""Private cohort grouping/provisioning fences; no production rollout."""
import json
import hashlib

from core.exceptions import StaleRevisionError, ValidationError
from infrastructure.persistence.import_cohort_scan import require_current_cohort


def validate_paged_track_state(connection, admission, run_id):
    from services.native.local_album_grouping_service import LocalAlbumGroupingService
    payload = require_current_cohort(connection, admission)
    context = connection.execute('SELECT * FROM library_scan_grouping_contexts '
        'WHERE run_id=? AND root_id=? AND relative_directory=?',
        (run_id, admission.root_id, admission.album_parent)).fetchone()
    run = connection.execute('SELECT state,requested_control FROM library_scan_runs WHERE id=?', (run_id,)).fetchone()
    if (context is None or context['state'] != 'pending'
            or context['staging_state'] not in {'pending', 'tracks', 'tokens'}
            or context['application_cursor'] is not None or context['queue_cursor'] is not None
            or context['grouping_merge_target'] is not None or context['grouping_merge_ready']
            or run is None or run['state'] not in {'indexing', 'reconciling'} or run['requested_control'] != 'none'):
        raise StaleRevisionError('Native cohort track staging is not pending/running.')
    rows = [dict(row) for row in connection.execute('SELECT t.*,a.created_at AS album_created_at '
        'FROM local_tracks t JOIN local_albums a ON a.id=t.local_album_id '
        'WHERE t.local_album_id=? ORDER BY t.relative_path,t.id', (payload['album_id'],))]
    cursor = context['staging_cursor']
    if context['staging_state'] == 'tracks':
        try:
            cursor_key = tuple(json.loads(cursor))
        except (TypeError, ValueError):
            raise StaleRevisionError('Native cohort track staging cursor is invalid.') from None
        keys = [(row['relative_path'], row['id']) for row in rows]
        if cursor_key not in keys:
            raise StaleRevisionError('Native cohort track staging cursor is not a member boundary.')
        prefix = rows[:keys.index(cursor_key) + 1]
    else:
        if cursor is not None:
            raise StaleRevisionError('Native cohort track staging cursor disagrees with phase.')
        prefix = rows if context['staging_state'] == 'tokens' else []
    expected = {row['id']: LocalAlbumGroupingService._grouping_evidence(row, admission.album_parent) for row in prefix}
    staged = [dict(row) for row in connection.execute('SELECT * FROM library_scan_grouping_evidence '
        'WHERE run_id=? AND root_id=? AND relative_directory=?',
        (run_id, admission.root_id, admission.album_parent))]
    if ({row['local_track_id'] for row in staged} != set(expected)
            or len(staged) != len(expected)
            or any(row['grouping_token'] is not None
                or any(row[key] != value for key, value in expected[row['local_track_id']].items()) for row in staged)):
        raise StaleRevisionError('Native cohort staged evidence is not the exact validated prefix.')
    return context, rows, len(prefix)


def prepare_paged_track_write(connection, admission, run_id, root_id, directory,
                              evidence, cursor, exhausted, limit, revision):
    from services.native.local_album_grouping_service import LocalAlbumGroupingService
    context, rows, prefix_count = validate_paged_track_state(connection, admission, run_id)
    if (root_id != admission.root_id or directory != admission.album_parent
            or context['staging_state'] not in {'pending', 'tracks'} or not isinstance(limit, int) or not 0 < limit <= 500):
        raise ValidationError('Exact bounded native cohort track page required.')
    if not isinstance(revision, int) or context['row_revision'] != revision:
        raise StaleRevisionError('Native cohort context revision changed before page staging.')
    page = rows[prefix_count:prefix_count + limit]
    expected = [LocalAlbumGroupingService._grouping_evidence(row, directory) for row in page]
    expected_cursor = json.dumps([page[-1]['relative_path'], page[-1]['id']]) if page else None
    if evidence != expected or cursor != expected_cursor or exhausted != (not page):
        raise StaleRevisionError('Native cohort page, cursor or exhaustion changed before staging.')
    return ('tokens', None, revision + 1) if not page else ('tracks', expected_cursor, revision + 1)


def validate_token_state(connection, admission, run_id, *, summary=False, continuity=False, provision=False, membership=False, retirement=False, completion=False, post=False):
    """Exact native single-tagged-fold aggregate proof, not a grouping override."""
    from services.native.local_album_grouping_service import LocalAlbumGroupingService
    payload = require_current_cohort(connection, admission)
    params = (run_id, admission.root_id, admission.album_parent)
    context = connection.execute('SELECT * FROM library_scan_grouping_contexts WHERE run_id=? AND root_id=? AND relative_directory=?', params).fetchone()
    run = connection.execute('SELECT state,requested_control FROM library_scan_runs WHERE id=?', (run_id,)).fetchone()
    retirement = retirement or completion
    membership = membership or retirement
    provision = provision or membership
    allowed = ({'completed'} if post else {'queue'}) if completion else ({'retirement', 'queue'} if retirement else ({'memberships', 'retirement'} if membership else ({'albums', 'memberships'} if provision else ({'continuity', 'albums'} if continuity else ({'groups', 'continuity'} if summary else {'tokens', 'groups'})))))
    continuity = continuity or provision
    if (context is None or context['state'] != ('completed' if completion and post else 'pending') or context['staging_state'] not in allowed
            or (not membership and context['application_cursor'] is not None) or (not (completion and post) and context['queue_cursor'] is not None)
            or run is None or run['state'] not in {'indexing', 'reconciling'} or run['requested_control'] != 'none'):
        raise StaleRevisionError('Native cohort token state is not pending/running.')
    tracks = [dict(row) for row in connection.execute('SELECT t.*,a.created_at AS album_created_at '
        'FROM local_tracks t JOIN local_albums a ON a.id=t.local_album_id WHERE t.local_album_id=? ORDER BY t.id', (payload['album_id'],))]
    if membership:
        ids = [row['id'] for row in tracks]
        cursor = context['application_cursor']
        if retirement:
            if ((completion and cursor is not None) or (context['staging_state'] == 'retirement' and cursor not in {None, payload['album_id']})
                    or (context['staging_state'] == 'queue' and cursor is not None)):
                raise StaleRevisionError('Native retirement cursor is not the retained album boundary.')
        elif cursor is not None and (context['staging_state'] != 'memberships' or cursor not in ids):
            raise StaleRevisionError('Native membership cursor is not a member boundary.')
        count = len(tracks) if retirement or context['staging_state'] == 'retirement' else (ids.index(cursor) + 1 if cursor else 0)
        if (any(row['membership_source'] not in {'automatic', 'legacy_import'} for row in tracks)
                or any(row['membership_source'] != 'automatic' for row in tracks[:count])):
            raise StaleRevisionError('Native membership source or processed prefix disagrees.')
    evidence = {row['id']: LocalAlbumGroupingService._grouping_evidence(row, admission.album_parent) for row in tracks}
    keys = {row['preliminary_key'] for row in evidence.values()}
    if len(keys) != 1 or not next(iter(keys)).startswith('tagged:'):
        raise ValidationError('Private token admission requires one exact native tagged fold.')
    token = next(iter(keys))
    ready = context['grouping_merge_ready']
    cursor = context['staging_cursor']
    summary_done = False
    if continuity:
        if not ready or context['grouping_merge_target'] != token or cursor is not None:
            raise StaleRevisionError('Native continuity phase, merge target or cursor changed.')
        prefix = tracks
        summary_done = True
    elif summary:
        if (not ready or context['grouping_merge_target'] != token
                or (context['staging_state'] == 'groups' and cursor not in {None, token})
                or (context['staging_state'] == 'continuity' and cursor is not None)):
            raise StaleRevisionError('Native summary phase, merge target or cursor changed.')
        prefix = tracks
        summary_done = context['staging_state'] == 'continuity' or cursor == token
    elif not ready:
        if context['staging_state'] != 'tokens' or cursor is not None or context['grouping_merge_target'] is not None:
            raise StaleRevisionError('Native token initialization state disagrees.')
        prefix = []
    else:
        if context['grouping_merge_target'] != token:
            raise StaleRevisionError('Native cohort token merge target changed.')
        if context['staging_state'] == 'groups':
            if cursor is not None:
                raise StaleRevisionError('Native token completion cursor changed.')
            prefix = tracks
        else:
            ids = [row['id'] for row in tracks]
            if cursor not in ids:
                raise StaleRevisionError('Native token cursor is not a member boundary.')
            prefix = tracks[:ids.index(cursor) + 1]
    processed = {row['id'] for row in prefix}
    def table(name):
        return [dict(row) for row in connection.execute(f'SELECT * FROM {name} WHERE run_id=? AND root_id=? AND relative_directory=?', params)]
    staged = table('library_scan_grouping_evidence')
    if ({row['local_track_id'] for row in staged} != set(evidence) or len(staged) != len(evidence)
            or any(any(row[key] != value for key, value in evidence[row['local_track_id']].items())
                   or row['grouping_token'] != (token if row['local_track_id'] in processed else None) for row in staged)):
        raise StaleRevisionError('Native token evidence or processed prefix changed.')
    expected_group = dict(grouping_token=token, grouping_key='', title='', album_artist_name='',
        reason_code='AMBIGUOUS_FALLBACK_GROUP', retained_album_id=None, continuity_reason_code=None,
        local_album_id=None, local_artist_id=None, automatic_track_count=len(prefix), local_metadata_track_count=0,
        excluded_track_count=0, embedded_identity_count=sum(row['embedded_release_group_mbid'] is not None
            or row['embedded_release_mbid'] is not None for row in prefix))
    matched = 0
    if continuity:
        edges = table('library_scan_grouping_edges')
        if len(edges) != 1 or edges[0]['processed'] not in {0, 1}:
            raise StaleRevisionError('Private continuity requires one exact isolated edge.')
        matched = edges[0]['processed']
        if (context['staging_state'] == 'albums' or provision) and not matched:
            raise StaleRevisionError('Native continuity was not assigned before completion.')
        if matched:
            expected_group.update(retained_album_id=payload['album_id'], continuity_reason_code='MAXIMUM_TRACK_OVERLAP')
    if provision:
        groups = table('library_scan_grouping_groups')
        album = connection.execute('SELECT * FROM local_albums WHERE id=?', (payload['album_id'],)).fetchone()
        if len(groups) != 1:
            raise StaleRevisionError('Native provisioning requires one retained group.')
        pair = (groups[0]['local_album_id'], groups[0]['local_artist_id'])
        if pair not in {(None, None), (payload['album_id'], album['album_artist_id'])}:
            raise StaleRevisionError('Native provisioned identity differs from owned album/artist.')
        if (context['staging_state'] == 'memberships' or membership) and pair[0] is None:
            raise StaleRevisionError('Native album provisioning is incomplete.')
        expected_group.update(local_album_id=pair[0], local_artist_id=pair[1])
    for name, values in (
        ('tag_revision_accumulator', [f"{row['id']}:{row['tag_revision'] or ''}" for row in prefix]),
        ('stat_revision_accumulator', [f"{row['id']}:{row['stat_revision']}" for row in prefix]),
        ('policy_revision_accumulator', [f"{row['id']}:{row['applied_policy_revision']}:{row['applied_policy']}" for row in prefix]),
    ):
        expected_group[name] = f"{sum(int.from_bytes(hashlib.sha256(value.encode()).digest(), 'big') for value in values) % (1 << 256):064x}"
    values = {}
    for track in prefix:
        row = evidence[track['id']]
        for kind, normalized, display in (
            ('title', row['title_normalized'], row['title'].strip()),
            ('artist', row['album_artist_normalized'], row['album_artist_name'].strip()),
            ('reason', row['reason_code'], row['reason_code']),
        ):
            if not normalized or (kind == 'reason' and normalized == 'MISSING_ALBUM_TAGS'):
                continue
            key = (token, kind, normalized)
            old = values.get(key)
            values[key] = (min(display, old[0]) if old else display, (old[1] if old else 0) + 1)
    actual_values = table('library_scan_grouping_values')
    if (len(actual_values) != len(values) or {(row['grouping_token'], row['value_kind'], row['normalized_value']):
            (row['display_value'], row['occurrence_count']) for row in actual_values} != values):
        raise StaleRevisionError('Native token value counts changed.')
    if summary_done:
        from infrastructure.persistence.native_library_store import _normalize_exact
        def choose(kind, fallback):
            options = [(key[2], value[0], value[1]) for key, value in values.items() if key[1] == kind]
            options.sort(key=(lambda item: item[0]) if kind == 'reason' else (lambda item: (-item[2], item[0])))
            return options[0][1] if options else fallback
        fallback = admission.album_parent.rstrip('/').rsplit('/', 1)[-1] if admission.album_parent != '.' else 'Unknown Album'
        title, artist = choose('title', fallback), choose('artist', 'Unknown Artist')
        expected_group.update(title=title, album_artist_name=artist,
            reason_code=choose('reason', 'AMBIGUOUS_FALLBACK_GROUP'),
            grouping_key=f'{admission.root_id}:{admission.album_parent}:{_normalize_exact(title)}:{_normalize_exact(artist)}')
    groups = table('library_scan_grouping_groups')
    if len(groups) != bool(prefix) or (prefix and any(groups[0][key] != value for key, value in expected_group.items())):
        raise StaleRevisionError('Native token group counts, summary or revision accumulators changed.')
    if provision:
        artist = connection.execute('SELECT * FROM local_artists WHERE id=?', (album['album_artist_id'],)).fetchone()
        credit = connection.execute('SELECT * FROM local_album_artists WHERE local_album_id=? AND position=0', (payload['album_id'],)).fetchone()
        if (pair[0] is not None and album['grouping_key'] != expected_group['grouping_key']):
            raise StaleRevisionError('Native provisioned album grouping key changed before resume.')
        if (album['retired_into_album_id'] is not None or album['root_id'] != admission.root_id or album['title'] != expected_group['title']
                or album['album_artist_name'] != expected_group['album_artist_name']
                or artist is None or artist['display_name'] != expected_group['album_artist_name']
                or artist['retired_into_artist_id'] is not None or credit is None
                or credit['local_artist_id'] != artist['id'] or credit['credited_name'] != expected_group['album_artist_name']
                or credit['role'] != 'primary'):
            raise StaleRevisionError('Native provisioning album, artist or primary credit disagrees.')
    for name, fields, expected in (
        ('library_scan_grouping_edges', ('old_album_id', 'grouping_token', 'overlap_count', 'processed'), [(payload['album_id'], token, len(prefix), matched)]),
        ('library_scan_grouping_old_nodes', ('old_album_id', 'degree', 'matched_grouping_token'), [(payload['album_id'], 1, None)]),
        ('library_scan_grouping_new_nodes', ('grouping_token', 'degree', 'matched_old_album_id'), [(token, 1, None)]),
    ):
        actual = [tuple(row[field] for field in fields) for row in table(name)]
        if actual != (expected if prefix else []):
            raise StaleRevisionError('Native token continuity edges or nodes changed.')
    return context, tracks, len(prefix), token


def prepare_token_write(connection, admission, run_id, root_id, directory, limit, revision):
    context, tracks, count, token = validate_token_state(connection, admission, run_id)
    if ((root_id, directory) != (admission.root_id, admission.album_parent)
            or context['staging_state'] != 'tokens' or not isinstance(limit, int) or not 0 < limit <= 500):
        raise ValidationError('Exact bounded native token page required.')
    if context['row_revision'] != revision:
        raise StaleRevisionError('Native token context revision changed before write.')
    page = tracks[count:count + limit]
    increments = 1 + int(not context['grouping_merge_ready'])
    return ('tokens' if page else 'groups', page[-1]['id'] if page else None,
        token, 1, revision + increments)


def prepare_summary_write(connection, admission, run_id, root_id, directory, limit, revision):
    context, _, _, token = validate_token_state(connection, admission, run_id, summary=True)
    if ((root_id, directory) != (admission.root_id, admission.album_parent)
            or context['staging_state'] != 'groups' or not isinstance(limit, int) or not 0 < limit <= 500):
        raise ValidationError('Exact bounded native summary page required.')
    if context['row_revision'] != revision:
        raise StaleRevisionError('Native summary context revision changed before write.')
    return ('groups', token, revision + 1) if context['staging_cursor'] is None else ('continuity', None, revision + 1)


def prepare_continuity_write(connection, admission, run_id, root_id, directory, revision, *, finish=False, limit=1):
    context, _, _, _ = validate_token_state(connection, admission, run_id, continuity=True)
    if ((root_id, directory) != (admission.root_id, admission.album_parent)
            or context['staging_state'] != 'continuity' or not isinstance(limit, int) or not 0 < limit <= 500):
        raise ValidationError('Exact isolated native continuity context required.')
    if context['row_revision'] != revision:
        raise StaleRevisionError('Native continuity context revision changed before write.')
    matched = connection.execute('SELECT processed FROM library_scan_grouping_edges WHERE run_id=? AND root_id=? AND relative_directory=?',
        (run_id, root_id, directory)).fetchone()[0]
    if finish and not matched:
        raise StaleRevisionError('Native continuity assignment is incomplete.')
    return ('albums' if finish else 'continuity', None, revision + int(finish)), (None if finish else 1 - matched)


def validate_continuity_write(connection, admission, run_id, expected):
    context, _, _, _ = validate_token_state(connection, admission, run_id, continuity=True)
    if tuple(context[key] for key in ('staging_state', 'staging_cursor', 'row_revision')) != expected:
        raise StaleRevisionError('Native continuity state changed during write.')
    edge = connection.execute('SELECT processed FROM library_scan_grouping_edges WHERE run_id=? AND root_id=? AND relative_directory=?',
        (run_id, admission.root_id, admission.album_parent)).fetchone()
    if edge['processed'] != 1:
        raise StaleRevisionError('Native isolated continuity did not retain the complete cohort.')


def prepare_provision_write(connection, admission, run_id, root_id, directory, revision, groups, *, finish=False):
    context, _, _, token = validate_token_state(connection, admission, run_id, provision=True)
    if (root_id, directory) != (admission.root_id, admission.album_parent) or context['staging_state'] != 'albums':
        raise ValidationError('Exact native album provisioning context required.')
    if context['row_revision'] != revision:
        raise StaleRevisionError('Native provisioning context revision changed before write.')
    params = (run_id, root_id, directory)
    group = dict(connection.execute('SELECT * FROM library_scan_grouping_groups WHERE run_id=? AND root_id=? AND relative_directory=?', params).fetchone())
    album = dict(connection.execute('SELECT * FROM local_albums WHERE id=?', (group['retained_album_id'],)).fetchone())
    artist = dict(connection.execute('SELECT * FROM local_artists WHERE id=?', (album['album_artist_id'],)).fetchone())
    if finish:
        if group['local_album_id'] != album['id'] or groups:
            raise StaleRevisionError('Native album provisioning is incomplete.')
    else:
        expected = dict(group, local_album_id=album['id'], local_artist_id=artist['id'])
        if group['local_album_id'] is not None or groups != [expected]:
            raise ValidationError('Exact unprovisioned owned album/artist application required.')
    credits = [dict(row) for row in connection.execute('SELECT * FROM local_album_artists WHERE local_album_id=? ORDER BY position', (album['id'],))]
    return dict(album=album, artist=artist, credits=credits, group=group, finish=finish,
        phase=('memberships' if finish else 'albums'), revision=revision + int(finish),
        catalog=connection.execute('SELECT value FROM library_catalog_revision WHERE singleton=1').fetchone()[0])


def validate_provision_write(connection, admission, run_id, before, now):
    context, _, _, _ = validate_token_state(connection, admission, run_id, provision=True)
    if context['staging_state'] != before['phase'] or context['row_revision'] != before['revision']:
        raise StaleRevisionError('Native provisioning state changed during write.')
    expected = dict(before['album'])
    changed = not before['finish'] and expected['grouping_key'] != before['group']['grouping_key']
    if changed:
        expected.update(grouping_key=before['group']['grouping_key'], updated_at=now, row_revision=expected['row_revision'] + 1)
    album = dict(connection.execute('SELECT * FROM local_albums WHERE id=?', (expected['id'],)).fetchone())
    artist = dict(connection.execute('SELECT * FROM local_artists WHERE id=?', (before['artist']['id'],)).fetchone())
    credits = [dict(row) for row in connection.execute('SELECT * FROM local_album_artists WHERE local_album_id=? ORDER BY position', (expected['id'],))]
    group = connection.execute('SELECT local_album_id,local_artist_id FROM library_scan_grouping_groups WHERE run_id=? AND root_id=? AND relative_directory=?',
        (run_id, admission.root_id, admission.album_parent)).fetchone()
    catalog = connection.execute('SELECT value FROM library_catalog_revision WHERE singleton=1').fetchone()[0]
    if (album != expected or artist != before['artist'] or credits != before['credits']
            or tuple(group) != (expected['id'], artist['id']) or catalog != before['catalog'] + int(changed)):
        raise StaleRevisionError('Native provisioning catalog, identity, credit or revision changed.')


def prepare_membership_write(connection, admission, run_id, root_id, directory, limit, revision):
    context, tracks, _, _ = validate_token_state(connection, admission, run_id, membership=True)
    if ((root_id, directory) != (admission.root_id, admission.album_parent)
            or context['staging_state'] != 'memberships' or type(limit) is not int or not 0 < limit <= 500):
        raise ValidationError('Exact bounded native membership page required.')
    if context['row_revision'] != revision:
        raise StaleRevisionError('Native membership context revision changed before write.')
    cursor = context['application_cursor']
    page = [row for row in tracks if row['id'] > (cursor or '')][:limit]
    expected_tracks = [dict(row) for row in tracks]
    page_ids = {row['id'] for row in page}
    changed = False
    for row in expected_tracks:
        if row['id'] in page_ids and row['membership_source'] != 'automatic':
            row.update(membership_source='automatic', row_revision=row['row_revision'] + 1)
            changed = True
    expected_context = dict(context)
    expected_context.update(application_cursor=page[-1]['id'] if page else None,
        staging_state='memberships' if page else 'retirement', row_revision=revision + 1)
    album_id = tracks[0]['local_album_id']
    album = dict(connection.execute('SELECT * FROM local_albums WHERE id=?', (album_id,)).fetchone())
    artist = dict(connection.execute('SELECT * FROM local_artists WHERE id=?', (album['album_artist_id'],)).fetchone())
    credits = [dict(row) for row in connection.execute('SELECT * FROM local_album_artists WHERE local_album_id=? ORDER BY position', (album_id,))]
    return dict(context=expected_context, tracks=expected_tracks, count=len(page), album=album, artist=artist, credits=credits,
        catalog=connection.execute('SELECT value FROM library_catalog_revision WHERE singleton=1').fetchone()[0] + int(changed))


def validate_membership_write(connection, admission, run_id, expected, count):
    context, tracks, _, _ = validate_token_state(connection, admission, run_id, membership=True)
    album = dict(connection.execute('SELECT * FROM local_albums WHERE id=?', (expected['album']['id'],)).fetchone())
    artist = dict(connection.execute('SELECT * FROM local_artists WHERE id=?', (expected['artist']['id'],)).fetchone())
    credits = [dict(row) for row in connection.execute('SELECT * FROM local_album_artists WHERE local_album_id=? ORDER BY position', (album['id'],))]
    catalog = connection.execute('SELECT value FROM library_catalog_revision WHERE singleton=1').fetchone()[0]
    if (dict(context) != expected['context'] or tracks != expected['tracks'] or count != expected['count']
            or album != expected['album'] or artist != expected['artist'] or credits != expected['credits']
            or catalog != expected['catalog']):
        raise StaleRevisionError('Native membership page, source, catalog or revision changed during write.')


def prepare_retirement_write(connection, admission, run_id, root_id, directory, limit, revision):
    context, tracks, _, _ = validate_token_state(connection, admission, run_id, retirement=True)
    if ((root_id, directory) != (admission.root_id, admission.album_parent)
            or context['staging_state'] != 'retirement' or type(limit) is not int or not 0 < limit <= 500):
        raise ValidationError('Exact bounded native retained-album retirement page required.')
    if context['row_revision'] != revision:
        raise StaleRevisionError('Native retirement context revision changed before write.')
    album_id = tracks[0]['local_album_id']
    album = dict(connection.execute('SELECT * FROM local_albums WHERE id=?', (album_id,)).fetchone())
    if album['retired_into_album_id'] is not None:
        raise StaleRevisionError('Owned cohort album is already retired.')
    artist = dict(connection.execute('SELECT * FROM local_artists WHERE id=?', (album['album_artist_id'],)).fetchone())
    credits = [dict(row) for row in connection.execute('SELECT * FROM local_album_artists WHERE local_album_id=? ORDER BY position', (album_id,))]
    aliases = [dict(row) for row in connection.execute('SELECT * FROM local_album_aliases WHERE local_album_id=? OR alias=? ORDER BY alias,local_album_id,kind', (album_id, album_id))]
    count = int(context['application_cursor'] is None)
    expected_context = dict(context)
    expected_context.update(application_cursor=album_id if count else None,
        staging_state='retirement' if count else 'queue', row_revision=revision + 1)
    return dict(context=expected_context, tracks=tracks, count=count, album=album, artist=artist, credits=credits, aliases=aliases,
        catalog=connection.execute('SELECT value FROM library_catalog_revision WHERE singleton=1').fetchone()[0])


def validate_retirement_write(connection, admission, run_id, expected, count):
    context, tracks, _, _ = validate_token_state(connection, admission, run_id, retirement=True)
    album_id = expected['album']['id']
    album = dict(connection.execute('SELECT * FROM local_albums WHERE id=?', (album_id,)).fetchone())
    artist = dict(connection.execute('SELECT * FROM local_artists WHERE id=?', (expected['artist']['id'],)).fetchone())
    credits = [dict(row) for row in connection.execute('SELECT * FROM local_album_artists WHERE local_album_id=? ORDER BY position', (album_id,))]
    aliases = [dict(row) for row in connection.execute('SELECT * FROM local_album_aliases WHERE local_album_id=? OR alias=? ORDER BY alias,local_album_id,kind', (album_id, album_id))]
    catalog = connection.execute('SELECT value FROM library_catalog_revision WHERE singleton=1').fetchone()[0]
    if (dict(context) != expected['context'] or tracks != expected['tracks'] or count != expected['count']
            or album != expected['album'] or artist != expected['artist'] or credits != expected['credits']
            or aliases != expected['aliases'] or catalog != expected['catalog']):
        raise StaleRevisionError('Native retained-album retirement changed catalog, aliases, source or revision.')


def paged_completion_catalog(connection, tracks):
    album_id = tracks[0]['local_album_id']
    album = dict(connection.execute('SELECT * FROM local_albums WHERE id=?', (album_id,)).fetchone())
    return dict(tracks=tracks, album=album,
        artist=dict(connection.execute('SELECT * FROM local_artists WHERE id=?', (album['album_artist_id'],)).fetchone()),
        credits=[dict(row) for row in connection.execute('SELECT * FROM local_album_artists WHERE local_album_id=? ORDER BY position', (album_id,))],
        aliases=[dict(row) for row in connection.execute('SELECT * FROM local_album_aliases WHERE local_album_id=? OR alias=? ORDER BY alias', (album_id, album_id))],
        revision=connection.execute('SELECT value FROM library_catalog_revision WHERE singleton=1').fetchone()[0])


def prepare_paged_completion(connection, admission, jobs, run_id, context_key, cursor, revision):
    context, tracks, _, token = validate_token_state(connection, admission, run_id, completion=True)
    if context_key != (admission.root_id, admission.album_parent) or cursor != token:
        raise ValidationError('Exact native paged completion context and token required.')
    if context['row_revision'] != revision:
        raise StaleRevisionError('Native paged completion context revision changed.')
    group = connection.execute('SELECT * FROM library_scan_grouping_groups WHERE run_id=? AND root_id=? AND relative_directory=?', (run_id, *context_key)).fetchone()
    native_revision = ':'.join(group[field] for field in ('tag_revision_accumulator', 'stat_revision_accumulator', 'policy_revision_accumulator'))
    album_id = tracks[0]['local_album_id']
    if (len(jobs) != 1 or jobs[0].local_album_id != album_id or jobs[0].local_track_id is not None
            or jobs[0].kind != 'automatic' or jobs[0].input_revision != native_revision
            or jobs[0].dedupe_key != f'automatic:{album_id}:{native_revision}'
            or jobs[0].priority != 20 or jobs[0].requested_by_user_id is not None):
        raise ValidationError('Exact fresh native paged identification job required.')
    run = dict(connection.execute('SELECT * FROM library_scan_runs WHERE id=?', (run_id,)).fetchone())
    return dict(context=dict(context), catalog=paged_completion_catalog(connection, tracks), run=run)


def validate_paged_completion(connection, admission, jobs, run_id, before, results):
    context, tracks, _, token = validate_token_state(connection, admission, run_id, completion=True, post=True)
    expected = dict(before['context'], state='completed', staging_state='completed', queue_cursor=token,
        row_revision=before['context']['row_revision'] + 2)
    run = dict(connection.execute('SELECT * FROM library_scan_runs WHERE id=?', (run_id,)).fetchone())
    expected_run = dict(before['run'])
    created = sum(created for _, created in results)
    if created:
        for field in ('identification_enqueued_count', 'row_revision', 'event_revision'):
            expected_run[field] += created
    queued = connection.execute('SELECT * FROM library_identification_jobs WHERE id=?', (results[0][0],)).fetchone()
    if (dict(context) != expected or run != expected_run or paged_completion_catalog(connection, tracks) != before['catalog']
            or queued is None or queued['requested_by_user_id'] is not None or queued['priority'] > jobs[0].priority
            or any(queued[field] != getattr(jobs[0], field) for field in
                ('local_album_id', 'local_track_id', 'kind', 'input_revision', 'dedupe_key'))):
        raise StaleRevisionError('Native paged completion job, catalog, context or counters changed.')
