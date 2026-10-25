"""Opt-in, query-only reference inspection; never a release/cleanup capability."""
from infrastructure.persistence.import_cohort_binding import (
    _cohort_schema_manifest, _cohort_schema_objects,
)


def inspect_cohort_retention(connection, *, binding_id=None, source_proof_id=None):
    """Use the caller's read transaction. Do not create/commit/repair anything.

    Absence of a policy invalidation is NOT complete membership validity. Unbound
    source observations and invalidated bindings are NOT deletion candidates.
    """
    if (not connection.in_transaction
            or connection.execute('PRAGMA query_only').fetchone()[0] != 1):
        raise ValueError('Caller-owned query-only read transaction required')
    if ((binding_id is None) == (source_proof_id is None)
            or any(value is not None and (type(value) is not str or not value or len(value)>256)
                   for value in (binding_id, source_proof_id))):
        raise ValueError('One exact binding or source-proof identifier required')
    result = dict(read_only=True, execution_enabled=False, safe_to_release=False,
                  eligibility_proven=False, reference_scope='private_cohort_and_native_job_snapshot_refs')
    actual = _cohort_schema_objects(connection)
    if not actual:
        return dict(result,state='PRIVATE_SCHEMA_NOT_INSTALLED')
    core, native = _cohort_schema_manifest()
    if (any(actual[name] != expected for name,expected in native.items() if name in actual)
            or {key:value for key,value in actual.items() if key not in native} != core
            or [tuple(row) for row in connection.execute('SELECT singleton,version FROM library_import_cohort_schema_state')] != [(1,2)]):
        raise ValueError('Exact private lifecycle schema required; no automatic repair')
    snapshot_id = None
    if binding_id is not None:
        row = connection.execute(
            'SELECT id,bundle_id,album_id,metadata_snapshot_id,source_proof_id '
            'FROM library_import_cohort_bindings WHERE id=?',(binding_id,)).fetchone()
        if row is None:
            return dict(result,state='BINDING_NOT_FOUND')
        invalidated = connection.execute(
            'SELECT COUNT(*) FROM library_import_cohort_policy_invalidations WHERE binding_id=?',
            (binding_id,)).fetchone()[0]
        result.update(state='BOUND_WITH_POLICY_INVALIDATION_RETAINS_REFERENCES' if invalidated
                      else 'BOUND_WITHOUT_POLICY_INVALIDATION_NOT_ELIGIBILITY_PROOF',
                      binding_id=row[0],bundle_id=row[1],album_id=row[2],
                      metadata_snapshot_id=row[3],source_proof_id=row[4],
                      member_reference_count=connection.execute(
                          'SELECT COUNT(*) FROM library_import_cohort_members WHERE binding_id=?',
                          (binding_id,)).fetchone()[0])
        snapshot_id = row[3]
    else:
        row = connection.execute(
            'SELECT id,metadata_snapshot_id FROM library_import_cohort_snapshot_sources WHERE id=?',
            (source_proof_id,)).fetchone()
        if row is None:
            return dict(result,state='SOURCE_PROOF_NOT_FOUND')
        bound, invalidated = connection.execute(
            'SELECT COUNT(*),COUNT(invalidation.binding_id) FROM library_import_cohort_bindings binding '
            'LEFT JOIN library_import_cohort_policy_invalidations invalidation ON invalidation.binding_id=binding.id '
            'WHERE binding.source_proof_id=?',(source_proof_id,)).fetchone()
        result.update(state='BOUND_SOURCE_PROOF_RETAINED' if bound else 'UNBOUND_SOURCE_PROOF_RETAINED',
                      source_proof_id=row[0],metadata_snapshot_id=row[1],
                      binding_reference_count=bound,invalidated_binding_reference_count=invalidated)
        snapshot_id = row[1]
    if snapshot_id is not None:
        result['snapshot_reference_counts'] = dict(
            cohort_bindings=connection.execute('SELECT COUNT(*) FROM library_import_cohort_bindings WHERE metadata_snapshot_id=?',(snapshot_id,)).fetchone()[0],
            source_proofs=connection.execute('SELECT COUNT(*) FROM library_import_cohort_snapshot_sources WHERE metadata_snapshot_id=?',(snapshot_id,)).fetchone()[0],
            native_jobs=connection.execute('SELECT COUNT(*) FROM library_management_job_metadata_snapshots WHERE metadata_snapshot_id=?',(snapshot_id,)).fetchone()[0])
    return result
