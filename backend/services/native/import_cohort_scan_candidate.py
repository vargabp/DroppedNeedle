"""Unwired orchestration; no production constructor or scan loop enables this."""
import asyncio
import json
from contextlib import asynccontextmanager

from core.exceptions import StaleRevisionError, ValidationError
from infrastructure.persistence.import_cohort_scan import CohortScanAdmission
from services.native.import_cohort_observation import observe_cohort_bytes


async def settle_cohort_write_candidate(awaitable):
    # Keep the enclosing native root lease until an in-flight SQL task settles,
    # even if its calling scan task is cancelled. No abandoned thread commit.
    from services.native.library_management_publisher import LibraryManagementPublisher
    result, cancelled = await LibraryManagementPublisher._finish_critical_task(asyncio.create_task(awaitable))
    if cancelled:
        raise asyncio.CancelledError
    return result


@asynccontextmanager
async def observed_cohort_candidate(*, store, filesystem, resolver, binding_id,
                                    max_total_bytes, timeout_seconds):
    # Database payload is discovery only; full hash/catalog/provenance validation
    # occurs inside the commit transaction. Missing candidate schema fails closed.
    def discover(connection):
        row = connection.execute('SELECT payload_json FROM library_import_cohort_bindings WHERE id=?',
                                 (binding_id,)).fetchone()
        if row is None:
            raise StaleRevisionError('Native cohort binding is missing.')
        return json.loads(row['payload_json'])
    payload = await store._read(discover)
    roots = [root for root in resolver.settings.library_roots if root.id == payload['root_id']]
    if (len(roots) != 1 or roots[0].policy != 'automatic'
            or roots[0].path != payload['root_path'] or resolver.policy_revision != payload['policy_revision']):
        raise ValidationError('Configured automatic root/policy disagrees with cohort scan.')
    if filesystem is None:
        raise ValidationError('Native filesystem coordinator is required.')
    async with filesystem.write_many([roots[0].id]):
        fingerprints = await asyncio.to_thread(observe_cohort_bytes, root_path=roots[0].path,
            members={member['track']['id']: (member['track']['relative_path'], member['track']['stat_revision'])
                     for member in payload['members']},
            max_total_bytes=max_total_bytes, timeout_seconds=timeout_seconds)
        admission = CohortScanAdmission(binding_id=binding_id, root_id=roots[0].id,
            root_path=roots[0].path, album_parent=payload['album_parent'],
            fingerprints=tuple(fingerprints.items()))
        yield admission


async def commit_cohort_scan_candidate(*, store, filesystem, resolver, binding_id,
                                       run_id, writes, updated_at,
                                       max_total_bytes, timeout_seconds, _inventory_candidate=None):
    from infrastructure.persistence.import_cohort_scan import cohort_inventory_increments
    async with observed_cohort_candidate(store=store, filesystem=filesystem, resolver=resolver,
            binding_id=binding_id, max_total_bytes=max_total_bytes, timeout_seconds=timeout_seconds) as admission:
        return await settle_cohort_write_candidate(store.commit_scan_index_batch(run_id, writes=writes, states={}, failures=[],
            increments=(cohort_inventory_increments(_inventory_candidate) if _inventory_candidate is not None
                        else {'indexed_count': len(writes)}), updated_at=updated_at,
            _cohort_scan_candidate=admission, _cohort_inventory_candidate=_inventory_candidate))


async def index_cohort_inventory_candidate(indexer, run, batch, counts, frozen_policy_revision, checkpoint):
    """Private complete rescan only; native tag reads occur OUTSIDE write lease."""
    from infrastructure.persistence.import_cohort_scan import (
        INVENTORY_FIELDS, CohortInventorySeal, require_cohort_inventory,
    )
    from services.native.file_revision import revision_from_stat
    store = indexer._store
    resolver = indexer._cohort_resolver_getter()
    if resolver.policy_revision != frozen_policy_revision or run.kind != 'rescan_files':
        raise ValidationError('Private cohort inventory requires frozen-policy rescan_files.')
    def discover(connection):
        connection.execute('BEGIN')
        rows = connection.execute('SELECT b.id,b.payload_json FROM library_import_cohort_bindings b '
            'JOIN library_import_cohort_members m ON m.binding_id=b.id '
            'JOIN local_tracks t ON t.id=m.local_track_id WHERE t.root_id=? AND t.relative_path=?',
            (batch[0]['root_id'], batch[0]['relative_path'])).fetchall()
        if len(rows) != 1:
            raise ValidationError('Private index candidate requires one complete bound cohort.')
        payload = json.loads(rows[0]['payload_json'])
        if not 0 < len(payload['members']) <= 500:
            raise ValidationError('Private cohort inventory exceeds observation budget.')
        inventory = []
        for member in payload['members']:
            row = connection.execute('SELECT * FROM library_scan_inventory WHERE run_id=? AND root_id=? AND relative_path=?',
                (run.id, payload['root_id'], member['track']['relative_path'])).fetchone()
            if row is None:
                raise StaleRevisionError('Exact whole pending cohort inventory is required.')
            inventory.append(dict(row))
        return rows[0]['id'], inventory
    binding_id, inventory = await store._read(discover)
    seal = CohortInventorySeal(tuple(tuple(row[field] for field in INVENTORY_FIELDS) for row in inventory))
    arguments = dict(store=store, filesystem=indexer._filesystem, resolver=resolver, binding_id=binding_id,
        max_total_bytes=indexer._cohort_max_total_bytes, timeout_seconds=indexer._cohort_timeout_seconds)
    # Validate before tag I/O and again in the actual catalog transaction. Do not
    # nest native read leases beneath the non-reentrant writer lease.
    async with observed_cohort_candidate(**arguments) as admission:
        def validate(connection):
            connection.execute('BEGIN')
            require_cohort_inventory(connection, admission, seal, run.id)
        await store._read(validate)
    writes = []
    for item in inventory:
        stable = await indexer._read_stable_tags(item, run_id=run.id,
            frozen_policy_revision=frozen_policy_revision, checkpoint=checkpoint)
        counts['tag_reads'] += stable[3]
        if stable[4]:
            return None
        if any(value is None for value in stable[:3]):
            raise StaleRevisionError('Native cohort tag read was unstable.')
        tag, info, stat = stable[:3]
        tagged = dict(item, file_size_bytes=stat.st_size, file_mtime_ns=stat.st_mtime_ns,
            stat_revision=revision_from_stat(stat))
        writes.append(indexer._prepare_tagged(run.id, tagged, tag, info, now=indexer._clock()))
    if checkpoint is not None and not await checkpoint(run.id, frozen_policy_revision):
        return None
    arguments['resolver'] = indexer._cohort_resolver_getter()
    updated = await commit_cohort_scan_candidate(**arguments, run_id=run.id, writes=writes,
        updated_at=indexer._clock(), _inventory_candidate=seal)
    counts['indexed'] += len(writes)
    counts['new'] += sum(item['comparison_result'] == 'new' for item in inventory)
    counts['changed'] += sum(item['comparison_result'] == 'changed' for item in inventory)
    return updated
