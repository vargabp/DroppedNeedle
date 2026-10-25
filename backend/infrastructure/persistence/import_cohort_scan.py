"""Unwired, complete-cohort-only native re-index admission candidate.

Not a partial-page token or execution authority. Caller must maintain a native
root write lease across fresh byte observation and the fenced SQL commit.
"""
from dataclasses import dataclass

import msgspec

from core.exceptions import StaleRevisionError, ValidationError
from infrastructure.persistence.import_cohort_binding import read_consistent_cohort

INVENTORY_FIELDS = ('root_id', 'relative_path', 'scope_relative_path', 'discovery_generation',
    'absolute_path', 'file_size_bytes', 'file_mtime_ns', 'stat_revision', 'policy_revision',
    'effective_policy', 'comparison_result', 'local_track_id', 'processing_state', 'row_revision')


@dataclass(frozen=True)
class CohortInventorySeal:
    rows: tuple[tuple, ...]


def cohort_inventory_increments(seal):
    rows = [dict(zip(INVENTORY_FIELDS, values)) for values in seal.rows]
    return dict(inspected_count=len(rows), indexed_count=len(rows),
        new_count=sum(row['comparison_result'] == 'new' for row in rows),
        changed_count=sum(row['comparison_result'] == 'changed' for row in rows),
        unchanged_count=0, excluded_count=0, errored_count=0)


def require_cohort_inventory(connection, admission, seal, run_id, *, post=False):
    payload = require_current_cohort(connection, admission)
    run = connection.execute('SELECT kind,state,requested_control FROM library_scan_runs WHERE id=?', (run_id,)).fetchone()
    if (run is None or run['kind'] != 'rescan_files' or run['state'] != 'indexing' or run['requested_control'] != 'none'):
        raise StaleRevisionError('Native cohort inventory scan is not an active rescan.')
    members = {member['track']['id']: member['track'] for member in payload['members']}
    rows = [dict(zip(INVENTORY_FIELDS, values)) for values in seal.rows]
    if (len(rows) != len(members) or {row['local_track_id'] for row in rows} != set(members)):
        raise ValidationError('Exact whole pending cohort inventory is required.')
    for expected in rows:
        track = members[expected['local_track_id']]
        if (expected['root_id'] != admission.root_id or expected['relative_path'] != track['relative_path']
                or expected['absolute_path'] != track['file_path'] or expected['stat_revision'] != track['stat_revision']
                or expected['effective_policy'] != 'automatic' or expected['policy_revision'] != payload['policy_revision']
                or expected['processing_state'] != 'pending' or expected['comparison_result'] not in {'new', 'changed', 'unchanged'}):
            raise StaleRevisionError('Native cohort inventory metadata disagrees with binding.')
        row = connection.execute('SELECT * FROM library_scan_inventory WHERE run_id=? AND root_id=? AND relative_path=?',
            (run_id, expected['root_id'], expected['relative_path'])).fetchone()
        compare = dict(expected)
        if post:
            compare.update(processing_state='indexed', row_revision=expected['row_revision'] + 1)
        if row is None or any(row[key] != value for key, value in compare.items()):
            raise StaleRevisionError('Native cohort inventory row changed before commit.')
        scope = connection.execute('SELECT * FROM library_scan_run_scopes WHERE run_id=? AND root_id=? AND relative_path=?',
            (run_id, expected['root_id'], expected['scope_relative_path'])).fetchone()
        if (scope is None or scope['discovery_state'] != 'completed'
                or scope['discovery_generation'] != expected['discovery_generation']
                or scope['root_path'] != admission.root_path or scope['effective_policy'] != 'automatic'
                or scope['policy_revision'] != payload['policy_revision']):
            raise StaleRevisionError('Native cohort discovery scope changed or is incomplete.')
        prefix = '' if scope['relative_path'] == '.' else scope['relative_path'].rstrip('/') + '/'
        if prefix and not expected['relative_path'].startswith(prefix):
            raise StaleRevisionError('Native cohort inventory is outside its discovery scope.')
    return payload


@dataclass(frozen=True)
class CohortScanAdmission:
    binding_id: str
    root_id: str
    root_path: str
    album_parent: str
    fingerprints: tuple[tuple[str, str], ...]


def require_current_cohort(connection, admission):
    if len(dict(admission.fingerprints)) != len(admission.fingerprints):
        raise ValidationError('Duplicate cohort scan fingerprints.')
    payload = read_consistent_cohort(connection, binding_id=admission.binding_id,
        root_id=admission.root_id, root_path=admission.root_path,
        album_parent=admission.album_parent, current_fingerprints=dict(admission.fingerprints))
    if payload is None:
        raise StaleRevisionError('Complete native cohort changed before scan commit.')
    current = {row[0] for row in connection.execute('SELECT id FROM local_tracks WHERE local_album_id=?',
                                                   (payload['album_id'],))}
    if current != {member['track']['id'] for member in payload['members']}:
        raise StaleRevisionError('Current album membership is not the complete native cohort.')
    # Completion covers the native parent context, not just one album. Match
    # native context candidate selection and refuse extra sibling-album tracks,
    # including arrivals after discovery or grouping but before queue commit.
    prefix = '' if admission.album_parent == '.' else f"{admission.album_parent.rstrip('/')}/"
    context_ids = {row[0] for row in connection.execute(
        'SELECT id FROM local_tracks WHERE root_id=? AND relative_path>=? AND relative_path<? '
        "AND (length(substr(relative_path,length(?)+1))-length(replace(substr(relative_path,length(?)+1),'/', '')))<=1",
        (admission.root_id, prefix, prefix + '\U0010ffff', prefix, prefix))}
    if context_ids != current:
        raise StaleRevisionError('Grouping context is not the exclusive complete native cohort.')
    if connection.execute('SELECT 1 FROM local_tracks WHERE local_album_id=? AND '
            '(manual_excluded<>0 OR applied_policy<>? OR applied_policy_revision<>?) LIMIT 1',
            (payload['album_id'], 'automatic', payload['policy_revision'])).fetchone():
        raise StaleRevisionError('Native cohort membership or applied policy changed: track exclusion or applied policy drift.')
    return payload


def prepare_scan_writes(connection, admission, writes):
    payload = require_current_cohort(connection, admission)
    members = {member['track']['id']: member['track'] for member in payload['members']}
    if len(writes) != len(members) or {write.track.id for write in writes} != set(members):
        raise ValidationError('Cohort scan candidate requires the complete distinct write set.')
    album = connection.execute('SELECT grouping_key FROM local_albums WHERE id=?',
                               (payload['album_id'],)).fetchone()
    admitted = []
    comparable = ('id', 'root_id', 'relative_path', 'file_path', 'stat_revision',
        'stat_revision_kind', 'tag_revision', 'embedded_release_mbid', 'embedded_release_group_mbid',
        'embedded_recording_mbid', 'embedded_release_track_mbid', 'disc_number', 'track_number',
        'tag_album_title', 'tag_album_artist_name', 'metadata_incomplete', 'membership_locked', 'availability')
    for write in writes:
        track = members[write.track.id]
        if (write.root_id != admission.root_id or write.relative_path != track['relative_path']
                or write.track.applied_policy != 'automatic'
                or write.track.applied_policy_revision != payload['policy_revision']
                or write.track.desired_policy_revision != payload['policy_revision']
                or any(getattr(write.track, field) != track[field] for field in comparable)):
            raise StaleRevisionError('Native scan metadata disagrees with the complete cohort.')
        existing = connection.execute('SELECT imported_at,manual_excluded,applied_policy,applied_policy_revision FROM local_tracks WHERE id=?',
                                      (write.track.id,)).fetchone()
        if (existing['manual_excluded'] or existing['applied_policy'] != 'automatic'
                or existing['applied_policy_revision'] != payload['policy_revision']):
            raise StaleRevisionError('Native cohort track exclusion or applied policy changed.')
        admitted.append(msgspec.structs.replace(write,
            album=msgspec.structs.replace(write.album, id=payload['album_id'], grouping_key=album['grouping_key']),
            track=msgspec.structs.replace(write.track, local_album_id=payload['album_id'],
                ingest_source=track['ingest_source'], download_task_id=track['download_task_id'],
                source_path=track['source_path'], imported_at=existing['imported_at']),
            grouping_context=admission.album_parent))
    return admitted
