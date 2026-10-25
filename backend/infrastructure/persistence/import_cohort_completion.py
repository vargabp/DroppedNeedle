"""Unwired small-context completion guard inside the native queue transaction."""
from core.exceptions import StaleRevisionError, ValidationError
from infrastructure.persistence.import_cohort_scan import require_current_cohort


def require_completion_candidate(connection, admission, jobs, run_id, context, cursor):
    from services.native.identification_revisions import album_input_revisions
    from services.native.local_album_grouper import normalize_group_value
    payload = require_current_cohort(connection, admission)
    if context != (admission.root_id, admission.album_parent) or cursor != payload['album_id']:
        raise ValidationError('Exact native cohort completion context required.')
    state = connection.execute('SELECT state,staging_state FROM library_scan_grouping_contexts '
        'WHERE run_id=? AND root_id=? AND relative_directory=?', (run_id, *context)).fetchone()
    run = connection.execute('SELECT state,requested_control FROM library_scan_runs WHERE id=?', (run_id,)).fetchone()
    if (state is None or state['state'] != 'pending' or state['staging_state'] != 'pending'
            or run is None or run['state'] not in {'indexing', 'reconciling'} or run['requested_control'] != 'none'):
        raise StaleRevisionError('Native scan or cohort completion context is not pending/running.')
    rows = [dict(row) for row in connection.execute('SELECT * FROM local_tracks WHERE local_album_id=?',
                                                  (payload['album_id'],)).fetchall()]
    if ({row['id'] for row in rows} != set(dict(admission.fingerprints))
            or any(row['manual_excluded'] or row['applied_policy'] != 'automatic'
                   or row['applied_policy_revision'] != payload['policy_revision'] for row in rows)):
        raise StaleRevisionError('Cohort completion membership or applied policy changed.')
    raw = payload['members'][0]['track']
    key = (f"{admission.root_id}:{admission.album_parent}:"
           f"{normalize_group_value(raw['tag_album_title'])}:{normalize_group_value(raw['tag_album_artist_name'])}")
    album = connection.execute('SELECT grouping_key FROM local_albums WHERE id=?', (payload['album_id'],)).fetchone()
    if album['grouping_key'] != key:
        raise StaleRevisionError('Cohort context has not passed native grouping application.')
    revision = ':'.join(album_input_revisions(rows))
    if (len(jobs) != 1 or jobs[0].local_album_id != payload['album_id'] or jobs[0].local_track_id is not None
            or jobs[0].kind != 'automatic' or jobs[0].input_revision != revision
            or jobs[0].dedupe_key != f"automatic:{payload['album_id']}:{revision}"
            or jobs[0].priority != 20 or jobs[0].requested_by_user_id is not None):
        raise ValidationError('Exact fresh native identification job required for cohort completion.')
