"""Unregistered native cohort persistence candidate; no production callers.

DDL is NOT registered in normal startup; only an explicit private opt-in
installs it with strict transactional schema checks. Native publication must
eventually supply the canonical release map and fresh filesystem fingerprints.
This slice cannot authenticate a caller's observations or authorize cleanup.
It never owns/commits a transaction and never changes grouping or manual locks.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping

import msgspec

from models.library_management_canonical import CanonicalReleaseDocument

CANDIDATE_V1_SCHEMA_SQL = """
CREATE TABLE library_import_cohort_snapshot_sources (
 id TEXT PRIMARY KEY,
 metadata_snapshot_id TEXT NOT NULL REFERENCES library_management_metadata_snapshots(id) ON DELETE RESTRICT,
 source_mode TEXT NOT NULL,
 source_id TEXT NOT NULL,
 source_generation INTEGER NOT NULL CHECK(source_generation>=0),
 source_url_sha256 TEXT NOT NULL CHECK(length(source_url_sha256)=64),
 input_hash TEXT NOT NULL,
 payload_sha256 TEXT NOT NULL,
 observed_at REAL NOT NULL
);
CREATE TRIGGER immutable_cohort_snapshot_source BEFORE UPDATE ON library_import_cohort_snapshot_sources
BEGIN SELECT RAISE(ABORT,'cohort source observations are immutable'); END;
CREATE TABLE library_import_cohort_bindings (
 id TEXT PRIMARY KEY,
 bundle_id TEXT NOT NULL UNIQUE REFERENCES library_management_import_bundles(id) ON DELETE RESTRICT,
 album_id TEXT NOT NULL REFERENCES local_albums(id) ON DELETE RESTRICT,
 version INTEGER NOT NULL CHECK(version=1),
 metadata_snapshot_id TEXT REFERENCES library_management_metadata_snapshots(id) ON DELETE RESTRICT,
 source_proof_id TEXT REFERENCES library_import_cohort_snapshot_sources(id) ON DELETE RESTRICT,
 payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
 payload_sha256 TEXT NOT NULL CHECK(length(payload_sha256)=64)
);
CREATE TABLE library_import_cohort_members (
 binding_id TEXT NOT NULL REFERENCES library_import_cohort_bindings(id) ON DELETE RESTRICT,
 local_track_id TEXT NOT NULL UNIQUE REFERENCES local_tracks(id) ON DELETE RESTRICT,
 ordinal INTEGER NOT NULL CHECK(ordinal>=0),
 PRIMARY KEY(binding_id,ordinal)
);
"""

CANDIDATE_LIFECYCLE_SCHEMA_SQL = """
CREATE TABLE library_import_cohort_schema_state (
 singleton INTEGER PRIMARY KEY CHECK(singleton=1),
 version INTEGER NOT NULL CHECK(version=2)
);
INSERT INTO library_import_cohort_schema_state VALUES(1,2);
CREATE TRIGGER immutable_cohort_schema_state_update BEFORE UPDATE ON library_import_cohort_schema_state
BEGIN SELECT RAISE(ABORT,'cohort schema state is immutable'); END;
CREATE TRIGGER immutable_cohort_schema_state_delete BEFORE DELETE ON library_import_cohort_schema_state
BEGIN SELECT RAISE(ABORT,'cohort schema state is immutable'); END;
CREATE TABLE library_import_cohort_policy_invalidations (
 binding_id TEXT PRIMARY KEY REFERENCES library_import_cohort_bindings(id) ON DELETE RESTRICT,
 payload_sha256 TEXT NOT NULL CHECK(length(payload_sha256)=64),
 recorded_policy_revision TEXT NOT NULL,
 observed_policy_revision TEXT NOT NULL,
 reason_code TEXT NOT NULL CHECK(reason_code='POLICY_BOUNDARY_CHANGED'),
 invalidated_at REAL NOT NULL
);
CREATE TRIGGER immutable_cohort_policy_invalidation_update BEFORE UPDATE ON library_import_cohort_policy_invalidations
BEGIN SELECT RAISE(ABORT,'cohort invalidation is immutable'); END;
CREATE TRIGGER immutable_cohort_policy_invalidation_delete BEFORE DELETE ON library_import_cohort_policy_invalidations
BEGIN SELECT RAISE(ABORT,'cohort invalidation is immutable'); END;
"""
CANDIDATE_SCHEMA_SQL = CANDIDATE_V1_SCHEMA_SQL + CANDIDATE_LIFECYCLE_SCHEMA_SQL

TRACK_FIELDS = ('id','local_album_id','root_id','relative_path','file_path',
                'stat_revision','stat_revision_kind','tag_revision','embedded_release_mbid',
                'embedded_release_group_mbid','embedded_recording_mbid',
                'embedded_release_track_mbid','disc_number','track_number',
                'download_task_id','source_path','ingest_source','tag_album_title',
                'tag_album_artist_name','metadata_incomplete','membership_locked','availability')
COMMITTED = {'catalog_committed','cleanup_pending','completed'}


def _cohort_schema_statements(schema=None):
    pending = ''
    for line in (CANDIDATE_SCHEMA_SQL if schema is None else schema).splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            yield pending.strip()
            pending = ''
    if pending.strip():
        raise ValueError('Incomplete private cohort schema statement.')


def _cohort_schema_objects(connection):
    return {row[0]: tuple(row[1:]) for row in connection.execute(
        "SELECT name,type,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL AND "
        "(name GLOB 'library_import_cohort_*' OR name GLOB 'immutable_cohort_*' "
        "OR tbl_name GLOB 'library_import_cohort_*')")}


def _cohort_schema_manifest(version=2):
    # executescript is safe ONLY on this isolated, empty reference database.
    # Caller migration below executes individual statements inside its savepoint.
    with closing(sqlite3.connect(':memory:')) as reference:
        reference.executescript(CANDIDATE_SCHEMA_SQL if version == 2 else CANDIDATE_V1_SCHEMA_SQL)
        core = _cohort_schema_objects(reference)
        tables = ['library_import_cohort_snapshot_sources', 'library_import_cohort_bindings', 'library_import_cohort_members']
        if version == 2:
            tables.append('library_import_cohort_policy_invalidations')
        for table in tables:
            for event in ('INSERT', 'UPDATE', 'DELETE'):
                trigger = f'trg_fk_validation_{table}_{event.casefold()}'
                reference.execute(f'CREATE TRIGGER IF NOT EXISTS "{trigger}" '
                    f'AFTER {event} ON "{table}" '
                    'WHEN (SELECT foreign_keys FROM pragma_foreign_keys) = 0 '
                    'BEGIN UPDATE library_foreign_key_validation_state '
                    'SET clean = 0, validated_at = NULL WHERE singleton = 1; END')
        native = {key: value for key, value in _cohort_schema_objects(reference).items() if key not in core}
    return core, native


def install_cohort_schema_candidate(connection):
    """Private opt-in migration; never commits, repairs, drops or backfills data."""
    if not connection.in_transaction or connection.execute('PRAGMA foreign_keys').fetchone()[0] != 1:
        raise ValueError('Native transaction with enforced foreign keys required for cohort schema.')
    parents = {'local_albums', 'local_tracks', 'library_management_import_bundles', 'library_management_metadata_snapshots'}
    existing = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not parents <= existing or connection.execute('PRAGMA foreign_key_check').fetchone() is not None:
        raise ValueError('Native cohort migration parents or foreign key integrity disagree.')
    core, native = _cohort_schema_manifest()
    actual = _cohort_schema_objects(connection)
    for name, expected in native.items():
        if name in actual:
            if actual.pop(name) != expected:
                raise ValueError('Native cohort foreign-key validation trigger changed.')
    if actual == core:
        if [tuple(row) for row in connection.execute('SELECT singleton,version FROM library_import_cohort_schema_state')] != [(1, 2)]:
            raise ValueError('Private cohort schema version marker disagrees.')
        return False
    legacy, _ = _cohort_schema_manifest(version=1)
    if actual and actual != legacy:
        raise ValueError('Partial or changed private cohort schema; no automatic repair.')
    connection.execute('SAVEPOINT private_cohort_schema_install')
    try:
        for statement in _cohort_schema_statements(CANDIDATE_LIFECYCLE_SCHEMA_SQL if actual else None):
            connection.execute(statement)
        installed = _cohort_schema_objects(connection)
        if (any(value != native[name] for name, value in installed.items() if name in native)
                or {name: value for name, value in installed.items() if name not in native} != core
                or [tuple(row) for row in connection.execute('SELECT singleton,version FROM library_import_cohort_schema_state')] != [(1, 2)]
                or connection.execute('PRAGMA foreign_key_check').fetchone() is not None):
            raise ValueError('Private cohort migration verification failed.')
    except BaseException:
        connection.execute('ROLLBACK TO private_cohort_schema_install')
        connection.execute('RELEASE private_cohort_schema_install')
        raise
    connection.execute('RELEASE private_cohort_schema_install')
    return True


@dataclass(frozen=True)
class CohortPolicyInvalidationCandidate:
    binding_id: str
    payload_sha256: str
    root_id: str
    root_path: str
    observed_policy_revision: str


@dataclass(frozen=True)
class CohortPolicyTransitionCandidate:
    transition_json: str
    configured_policy_revision: str
    root_ids: tuple[str, ...]
    bindings: tuple[CohortPolicyInvalidationCandidate, ...]


def capture_cohort_policy_transition_candidate(connection, configured_policy_revision, *, boundary_committed=False):
    """Seal one prepared boundary and its entire bounded active binding set."""
    import msgspec
    from api.v1.schemas.library_policies import TypedLibrarySettings
    from services.native.library_policy_resolver import LibraryPolicyResolver
    if not connection.in_transaction or not _cohort_lifecycle_ready(connection):
        raise ValueError('Cohort transition transaction and lifecycle required.')
    row = connection.execute('SELECT * FROM library_policy_transitions WHERE singleton=1').fetchone()
    if row is None or row['state'] != 'prepared':
        raise ValueError('Exact prepared cohort transition required.')
    previous = LibraryPolicyResolver(msgspec.json.decode(row['previous_settings_json'], type=TypedLibrarySettings))
    proposed = LibraryPolicyResolver(msgspec.json.decode(row['proposed_settings_json'], type=TypedLibrarySettings))
    if (previous.policy_revision != row['previous_policy_revision']
            or proposed.policy_revision != row['proposed_policy_revision']
            or proposed.policy_revision != configured_policy_revision):
        raise ValueError('Configured cohort transition revisions disagree.')
    expected = proposed.policy_revision if boundary_committed else previous.policy_revision
    durable = [r[0] for r in connection.execute('SELECT desired_policy_revision FROM library_policy_state')]
    if durable != [expected]:
        raise ValueError('Durable cohort transition boundary disagrees.')
    rows = connection.execute('SELECT b.* FROM library_import_cohort_bindings b '
        'LEFT JOIN library_import_cohort_policy_invalidations i ON i.binding_id=b.id '
        'WHERE i.binding_id IS NULL ORDER BY b.id LIMIT 501').fetchall()
    if len(rows) > 500:
        raise ValueError('Cohort transition binding budget exceeded.')
    roots = {root.id: root for root in previous.settings.library_roots}
    seals = []
    for binding in rows:
        if _digest(binding['payload_json']) != binding['payload_sha256']:
            raise ValueError('Cohort transition binding checksum changed.')
        payload = json.loads(binding['payload_json'])
        root = roots.get(payload['root_id'])
        if (root is None or root.path != payload['root_path'] or root.policy != 'automatic'
                or payload['policy_revision'] != previous.policy_revision):
            raise ValueError('Cohort transition previous root or policy disagrees.')
        seals.append(CohortPolicyInvalidationCandidate(binding['id'], binding['payload_sha256'],
            root.id, root.path, proposed.policy_revision))
    root_ids = tuple(sorted({root.id for root in previous.settings.library_roots + proposed.settings.library_roots}))
    return CohortPolicyTransitionCandidate(json.dumps(dict(row), sort_keys=True), configured_policy_revision,
        root_ids, tuple(seals))


def require_cohort_policy_transition_candidate(connection, seal, *, boundary_committed=False):
    if type(seal) is not CohortPolicyTransitionCandidate or capture_cohort_policy_transition_candidate(
            connection, seal.configured_policy_revision, boundary_committed=boundary_committed) != seal:
        raise ValueError('Cohort transition seal changed.')


def _cohort_lifecycle_ready(connection):
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('library_import_cohort_policy_invalidations','library_import_cohort_schema_state')")}
    return len(tables) == 2 and [tuple(row) for row in connection.execute('SELECT singleton,version FROM library_import_cohort_schema_state')] == [(1, 2)]


def invalidate_cohort_policy_candidate(connection, seal, *, invalidated_at):
    """Append disabling evidence ONLY; no catalog edits, reactivation or rebind."""
    if (not connection.in_transaction or not _cohort_lifecycle_ready(connection) or type(seal) is not CohortPolicyInvalidationCandidate
            or type(invalidated_at) not in (int, float) or not math.isfinite(invalidated_at) or invalidated_at < 0):
        raise ValueError('Exact policy invalidation evidence and transaction required.')
    row = _row(connection, 'SELECT * FROM library_import_cohort_bindings WHERE id=?', (seal.binding_id,))
    if row is None or row['payload_sha256'] != seal.payload_sha256 or _digest(row['payload_json']) != seal.payload_sha256:
        raise ValueError('Cohort invalidation binding seal changed.')
    payload = json.loads(row['payload_json'])
    policy = connection.execute('SELECT desired_policy_revision FROM library_policy_state').fetchall()
    if ((payload['root_id'], payload['root_path']) != (seal.root_id, seal.root_path)
            or len(policy) != 1 or policy[0][0] != seal.observed_policy_revision
            or not seal.observed_policy_revision or payload['policy_revision'] == seal.observed_policy_revision):
        raise ValueError('Fresh durable policy disagreement required for invalidation.')
    existing = connection.execute('SELECT payload_sha256 FROM library_import_cohort_policy_invalidations WHERE binding_id=?', (seal.binding_id,)).fetchone()
    if existing is not None:
        if existing[0] != seal.payload_sha256:
            raise ValueError('Existing invalidation seal disagrees.')
        return False
    before_changes = connection.total_changes
    connection.execute('INSERT INTO library_import_cohort_policy_invalidations VALUES(?,?,?,?,?,?)',
        (seal.binding_id, seal.payload_sha256, payload['policy_revision'], seal.observed_policy_revision,
         'POLICY_BOUNDARY_CHANGED', invalidated_at))
    after = connection.execute('SELECT * FROM library_import_cohort_policy_invalidations WHERE binding_id=?', (seal.binding_id,)).fetchone()
    if (connection.total_changes != before_changes + 1 or tuple(after) != (seal.binding_id, seal.payload_sha256,
            payload['policy_revision'], seal.observed_policy_revision, 'POLICY_BOUNDARY_CHANGED', invalidated_at)):
        raise ValueError('Policy invalidation changed during write.')
    return True


@dataclass(frozen=True)
class CohortSnapshotSeal:
    """Native snapshot correlation; optional source observation is internal evidence."""

    snapshot_id: str
    input_hash: str
    payload_sha256: str
    source_proof_id: str | None = None

    def __post_init__(self) -> None:
        if not self.snapshot_id or not _fingerprint(self.input_hash) or not _fingerprint(self.payload_sha256):
            raise ValueError('Invalid native snapshot seal')
        if self.source_proof_id is not None and not _fingerprint(self.source_proof_id):
            raise ValueError('Invalid native source observation identity')


@dataclass(frozen=True)
class ImportCohortCommitCandidate:
    """Untrusted local candidate input, never a capability or commit token."""

    binding_id: str
    root_id: str
    root_path: str
    album_parent: str
    canonical_map: tuple[tuple[str, str, int, int], ...]
    fingerprints: tuple[tuple[int, str], ...]
    snapshot: CohortSnapshotSeal | None = None

    def __post_init__(self) -> None:
        if (type(self.canonical_map) is not tuple or type(self.fingerprints) is not tuple
                or (self.snapshot is not None and type(self.snapshot) is not CohortSnapshotSeal)
                or any(type(value) is not tuple for value in (*self.canonical_map, *self.fingerprints))
                or any(type(value) is not str or not value for value in (
                    self.binding_id, self.root_id, self.root_path, self.album_parent))
                or any(len(value) != 4 or type(value[0]) is not str or not value[0]
                       or type(value[1]) is not str or not value[1]
                       or type(value[2]) is not int or value[2] < 1
                       or type(value[3]) is not int or value[3] < 1
                       for value in self.canonical_map)
                or any(len(value) != 2 or type(value[0]) is not int or not _fingerprint(value[1])
                       for value in self.fingerprints)
                or len(dict(self.fingerprints)) != len(self.fingerprints)):
            raise ValueError('Immutable distinct candidate observations required')


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _row(connection: sqlite3.Connection, sql: str, args: tuple) -> dict | None:
    cursor=connection.execute(sql,args)
    value=cursor.fetchone()
    return dict(zip((c[0] for c in cursor.description),value)) if value is not None else None


def _path(value: str, *, absolute: bool) -> PurePosixPath:
    if not isinstance(value,str) or not value or any(ord(c)<32 for c in value) or '\\' in value:
        raise ValueError('Invalid cohort path')
    path=PurePosixPath(value)
    if path.is_absolute()!=absolute or path.as_posix()!=value or '..' in path.parts or value.startswith('//') or value in ('.','/'):
        raise ValueError('Noncanonical cohort path')
    return path


def _fingerprint(value: str) -> bool:
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _verified_snapshot(connection: sqlite3.Connection, seal: CohortSnapshotSeal,
                       request: dict, canonical_map: tuple, at: float | None,
                       source_context: object | None = None) -> dict:
    row = _row(connection, 'SELECT * FROM library_management_metadata_snapshots WHERE id=?', (seal.snapshot_id,))
    if (not row or row['provider'] != 'musicbrainz' or row['entity_kind'] != 'release'
            or row['input_hash'] != seal.input_hash or row['payload_sha256'] != seal.payload_sha256
            or _digest(row['canonical_payload_json']) != seal.payload_sha256
            or any(file.get('metadata_snapshot_id') != seal.snapshot_id for file in request['files'])):
        raise ValueError('Native snapshot seal or sealed request changed')
    if at is not None:
        fetched, expires = row['fetched_at'], row['expires_at']
        if (not math.isfinite(at) or not math.isfinite(fetched) or expires is None
                or not math.isfinite(expires) or fetched > at or expires <= at or expires <= fetched):
            raise ValueError('Native snapshot is stale or has invalid clock evidence')
    document = msgspec.json.decode(row['canonical_payload_json'].encode(), type=CanonicalReleaseDocument)
    identity = document.identifiers
    releases = {(file.get('release_mbid'), file.get('release_group_mbid')) for file in request['files']}
    if (document.identity_kind != 'exact_release' or document.total_discs != len(document.media)
            or not document.media or row['entity_id'] != identity.release_mbid
            or releases != {(identity.release_mbid, identity.release_group_mbid)}):
        raise ValueError('Native snapshot release or complete medium scope disagrees')
    mapping = []
    for medium in document.media:
        if (medium.track_count != len(medium.tracks) or medium.track_count < 1
                or {track.track_number for track in medium.tracks} != set(range(1, medium.track_count + 1))):
            raise ValueError('Native snapshot projection is not a complete medium')
        for track in medium.tracks:
            ids = track.identifiers
            if (track.disc_number != medium.position or track.total_tracks != medium.track_count
                    or track.total_discs != document.total_discs
                    or (ids.release_mbid, ids.release_group_mbid) != (identity.release_mbid, identity.release_group_mbid)):
                raise ValueError('Native snapshot track scope disagrees')
            mapping.append((ids.release_track_mbid, ids.recording_mbid, track.disc_number, track.track_number))
    if sorted(mapping) != sorted(tuple(value) for value in canonical_map):
        raise ValueError('Native snapshot recording map differs from cohort')
    if seal.source_proof_id is not None:
        proof = _row(connection, 'SELECT * FROM library_import_cohort_snapshot_sources WHERE id=?', (seal.source_proof_id,))
        if (not proof or proof['metadata_snapshot_id'] != seal.snapshot_id
                or proof['input_hash'] != seal.input_hash or proof['payload_sha256'] != seal.payload_sha256):
            raise ValueError('Native source observation does not match snapshot')
        if at is not None:
            if (source_context is None or proof['observed_at'] > at
                    or (proof['source_mode'], proof['source_id'], proof['source_generation'], proof['source_url_sha256']) != (
                        source_context.source_mode, source_context.source_id, source_context.generation,
                        _digest(source_context.source_url))):
                raise ValueError('Native source context changed before cohort adoption')
    return {'snapshot_id': seal.snapshot_id, 'input_hash': seal.input_hash,
            'payload_sha256': seal.payload_sha256, 'source_proof_id': seal.source_proof_id}


def _journal_matches(journal: dict | None, member: dict, phases: set[str]) -> bool:
    track=member['track']
    return bool(journal and journal['state'] in phases and not journal['failure_code']
                and journal['destination_root_id']==track['root_id']
                and journal['destination_relative_path']==track['relative_path']
                and journal['staged_fingerprint']==member['fingerprint'])


def persist_import_cohort(connection: sqlite3.Connection, *, binding_id: str,
                          bundle_id: str, album_id: str, root_id: str,
                          root_path: str, album_parent: str,
                          canonical_map: tuple[tuple[str,str,int,int], ...],
                          members: Mapping[int,tuple[str,str]],
                          snapshot_seal: CohortSnapshotSeal | None = None,
                          observed_at: float | None = None,
                          source_context: object | None = None) -> None:
    """Candidate tx primitive: ordinal -> (catalog track ID, fresh file SHA256).

    Only a future trusted native publisher may call this after resolving ONE
    album and independently validating the complete canonical map and bytes.
    """
    if not connection.in_transaction or connection.execute('PRAGMA foreign_keys').fetchone()[0]!=1:
        raise ValueError('Caller transaction and foreign keys required')
    if not binding_id or not members or len(members)>500:
        raise ValueError('Invalid cohort size/identity')
    root=_path(root_path,absolute=True); parent=_path(album_parent,absolute=False)
    album=_row(connection,'SELECT root_id,grouping_locked,grouping_source,retired_into_album_id FROM local_albums WHERE id=?',(album_id,))
    if not album or album['root_id']!=root_id or album['grouping_locked'] or album['grouping_source']!='automatic' or album['retired_into_album_id']:
        raise ValueError('Automatic unlocked album required')
    bundle=_row(connection,'SELECT * FROM library_management_import_bundles WHERE id=?',(bundle_id,))
    if not bundle or bundle['state']!='publishing' or _digest(bundle['request_json'])!=bundle['request_hash']:
        raise ValueError('Native publishing request changed')
    request=json.loads(bundle['request_json'])
    files=request['files']
    ordinals=[f['ordinal'] for f in files]
    policy=connection.execute('SELECT desired_policy_revision FROM library_policy_state').fetchall()
    if len(policy)!=1 or policy[0][0]!=bundle['policy_revision'] or request['policy_revision']!=bundle['policy_revision']:
        raise ValueError('Policy boundary disagreement')
    if (request['origin']!='acquisition' or any(type(o) is not int or o<0 for o in ordinals)
            or any(type(o) is not int or o<0 for o in members)
            or len(ordinals)!=len(set(ordinals)) or set(ordinals)!=set(members)):
        raise ValueError('Complete acquisition membership required')
    requested=[(f.get('release_track_mbid'),f.get('recording_mbid'),f.get('medium_position'),f.get('release_track_position')) for f in files]
    if any(not a or not b or type(d) is not int or d<1 or type(p) is not int or p<1 for a,b,d,p in requested):
        raise ValueError('Complete managed identities required')
    if len({m[0] for m in canonical_map})!=len(canonical_map) or len({m[2:] for m in canonical_map})!=len(canonical_map) or sorted(requested)!=sorted(canonical_map):
        raise ValueError('Canonical recording map disagreement')
    releases={(f.get('release_mbid'),f.get('release_group_mbid')) for f in files}
    if len(releases)!=1 or not all(next(iter(releases))):
        raise ValueError('One exact release and group required')
    payload={'bundle_id':bundle_id,'album_id':album_id,'root_id':root_id,'root_path':root_path,
             'album_parent':album_parent,'request_hash':bundle['request_hash'],
             'policy_revision':bundle['policy_revision'],'canonical_map':sorted(canonical_map),'members':[]}
    if snapshot_seal is not None:
        if observed_at is None:
            raise ValueError('Native snapshot commit clock required')
        payload['native_snapshot'] = _verified_snapshot(connection, snapshot_seal, request, canonical_map, observed_at, source_context)
    for file in sorted(files,key=lambda f:f['ordinal']):
        ordinal=file['ordinal']; track_id,fingerprint=members[ordinal]
        track=_row(connection,'SELECT '+','.join(TRACK_FIELDS)+' FROM local_tracks WHERE id=?',(track_id,))
        journal=_row(connection,'SELECT * FROM library_management_import_journal WHERE bundle_id=? AND ordinal=?',(bundle_id,ordinal))
        if not track or not _fingerprint(fingerprint) or file.get('authoritative_mapping') is not True or file.get('reviewed_recording_identity',False) is not False:
            raise ValueError('Managed native publication proof required')
        relative=_path(track['relative_path'],absolute=False)
        if parent not in relative.parents or track['file_path']!=str(root/relative):
            raise ValueError('Root or album parent changed')
        actual=(track['embedded_release_track_mbid'],track['embedded_recording_mbid'],track['disc_number'],track['track_number'])
        expected=(file['release_track_mbid'],file['recording_mbid'],file['medium_position'],file['release_track_position'])
        if (track['root_id']!=root_id or file['destination_root_id']!=root_id
                or track['relative_path']!=file['destination_relative_path'] or track['local_album_id']!=album_id
                or actual!=expected or (track['embedded_release_mbid'],track['embedded_release_group_mbid'])!=next(iter(releases))
                or track['download_task_id']!=file.get('download_task_id') or track['membership_locked'] or track['availability']!='indexed'
                or track['stat_revision_kind']!='exact' or track['metadata_incomplete']
                or not track['stat_revision'] or not track['tag_revision']):
            raise ValueError('Catalog identity/provenance disagreement')
        member={'ordinal':ordinal,'track':track,'fingerprint':fingerprint}
        if not _journal_matches(journal,member,{'published'}):
            raise ValueError('Published journal disagreement')
        payload['members'].append(member)
    tag_contexts={(m['track']['tag_album_title'],m['track']['tag_album_artist_name']) for m in payload['members']}
    if len(tag_contexts)!=1 or not all(next(iter(tag_contexts))):
        raise ValueError('Conflicting or missing raw album tags')
    encoded=_json(payload)
    connection.execute('SAVEPOINT import_cohort_write')
    try:
        connection.execute('INSERT INTO library_import_cohort_bindings(id,bundle_id,album_id,version,metadata_snapshot_id,source_proof_id,payload_json,payload_sha256) VALUES(?,?,?,1,?,?,?,?)',
                           (binding_id,bundle_id,album_id,snapshot_seal.snapshot_id if snapshot_seal else None,
                            snapshot_seal.source_proof_id if snapshot_seal else None,encoded,_digest(encoded)))
        connection.executemany('INSERT INTO library_import_cohort_members VALUES(?,?,?)',
                              [(binding_id,m['track']['id'],m['ordinal']) for m in payload['members']])
        connection.execute('RELEASE import_cohort_write')
    except Exception:
        connection.execute('ROLLBACK TO import_cohort_write'); connection.execute('RELEASE import_cohort_write')
        raise


def read_consistent_cohort(connection: sqlite3.Connection, *, binding_id: str,
                           root_id: str, root_path: str, album_parent: str,
                           current_fingerprints: Mapping[str,str]) -> dict | None:
    """Read whole binding before any paging; no partial-page endorsement.

    Caller must supply fresh native filesystem observations, not model values.
    Failure means no cohort context, NEVER permission to retry/clean up.
    """
    if not connection.in_transaction:
        raise ValueError('Caller read transaction required')
    # Missing lifecycle schema is NOT permission to reuse a legacy binding.
    if not _cohort_lifecycle_ready(connection):
        return None
    if connection.execute('SELECT 1 FROM library_import_cohort_policy_invalidations WHERE binding_id=?', (binding_id,)).fetchone() is not None:
        return None
    row=_row(connection,'SELECT * FROM library_import_cohort_bindings WHERE id=?',(binding_id,))
    if not row or row['version']!=1 or _digest(row['payload_json'])!=row['payload_sha256']:
        return None
    payload=json.loads(row['payload_json'])
    if (payload['root_id'],payload['root_path'],payload['album_parent'])!=(root_id,root_path,album_parent):
        return None
    album=_row(connection,'SELECT root_id,grouping_locked,grouping_source,retired_into_album_id FROM local_albums WHERE id=?',(payload['album_id'],))
    if not album or album['root_id']!=root_id or album['grouping_locked'] or album['grouping_source']!='automatic' or album['retired_into_album_id']:
        return None
    bundle=_row(connection,'SELECT * FROM library_management_import_bundles WHERE id=?',(payload['bundle_id'],))
    if not bundle or bundle['state'] not in COMMITTED or bundle['request_hash']!=payload['request_hash'] or _digest(bundle['request_json'])!=payload['request_hash']:
        return None
    snapshot = payload.get('native_snapshot')
    if (snapshot is None) != (row['metadata_snapshot_id'] is None):
        return None
    if snapshot is None and row['source_proof_id'] is not None:
        return None
    if snapshot is not None:
        if row['metadata_snapshot_id'] != snapshot['snapshot_id']:
            return None
        if row['source_proof_id'] != snapshot.get('source_proof_id'):
            return None
        try:
            # Expiry gates initial import, not the continued existence of an
            # already committed catalog album. Content/reference still checked.
            _verified_snapshot(connection, CohortSnapshotSeal(**snapshot), json.loads(bundle['request_json']),
                               tuple(payload['canonical_map']), None)
        except (ValueError, TypeError, KeyError, msgspec.DecodeError):
            return None
    policy=connection.execute('SELECT desired_policy_revision FROM library_policy_state').fetchall()
    if len(policy)!=1 or policy[0][0]!=payload['policy_revision']:
        return None
    expected={m['track']['id'] for m in payload['members']}
    stored=connection.execute('SELECT local_track_id,ordinal FROM library_import_cohort_members WHERE binding_id=?',(binding_id,)).fetchall()
    if set(current_fingerprints)!=expected or {(r[0],r[1]) for r in stored}!={(m['track']['id'],m['ordinal']) for m in payload['members']}:
        return None
    for member in payload['members']:
        track=_row(connection,'SELECT '+','.join(TRACK_FIELDS)+' FROM local_tracks WHERE id=?',(member['track']['id'],))
        journal=_row(connection,'SELECT * FROM library_management_import_journal WHERE bundle_id=? AND ordinal=?',(payload['bundle_id'],member['ordinal']))
        if track!=member['track'] or current_fingerprints[member['track']['id']]!=member['fingerprint'] or not _journal_matches(journal,member,COMMITTED):
            return None
    return payload
