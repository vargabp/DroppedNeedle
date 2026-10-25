"""Actual lifespan ownership, with unrelated startup services isolated."""
import asyncio
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch


class TargetRuntimeCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, failure):
        import target_application as target
        from core import config
        from core.dependencies import auth_providers, service_providers
        from core.task_registry import TaskRegistry
        from maintenance import automatic_upgrade

        registry = TaskRegistry()
        created = []
        order = []
        settings = SimpleNamespace(log_level='INFO', cache_dir=Path('/tmp'), shutdown_grace_period=0.2)
        preferences = SimpleNamespace(
            get_instance_id=lambda:'fixture-instance',
            get_advanced_settings=lambda:SimpleNamespace(memory_cache_cleanup_interval=60,disk_cache_cleanup_interval=60),
            get_typed_library_settings=lambda:SimpleNamespace(enabled=False,library_roots=[]),
            clear_library_scan_dirty_scopes=lambda _ids:None,
            get_library_scan_filesystem_watcher=lambda:SimpleNamespace(enabled=False))
        coordinator = SimpleNamespace(aclose=AsyncMock())
        cleanup = AsyncMock()
        entered_operational = asyncio.Event()

        @asynccontextmanager
        async def progress(*args):
            yield

        async def parked():
            await asyncio.Event().wait()

        def start(name):
            def starter(*args, **kwargs):
                if failure=='starter' and name=='identification':
                    raise RuntimeError('fixture worker start failed')
                task = registry.register(name,asyncio.create_task(parked()))
                created.append(task)
                return task
            return starter

        async def operational(**kwargs):
            entered_operational.set()
            if failure=='operational':
                raise RuntimeError('fixture operational start failed')
            if failure=='cancel':
                await asyncio.Event().wait()

        original_cancel, original_all = registry.cancel, registry.cancel_all
        async def cancel(name, *args, **kwargs):
            order.append(name)
            await original_cancel(name,*args,**kwargs)
        async def cancel_all(*args, **kwargs):
            order.append('cancel_all')
            await original_all(*args,**kwargs)

        validator = AsyncMock(side_effect=RuntimeError('fixture catalog refused') if failure=='catalog' else None)
        recovery = SimpleNamespace(examined_bundles=0,recovered_bundles=0,rolled_back_bundles=0,needs_attention_bundles=0,skipped_bundles=0)
        with ExitStack() as stack:
            replacements = [
                (target,'get_settings',lambda:settings),
                (target,'_warn_if_launcher_bypassed',AsyncMock(return_value=False)),
                (config,'migrate_legacy_config',lambda:None),
                (target,'init_app_state',AsyncMock()),
                (automatic_upgrade,'target_startup_progress',progress),
                (automatic_upgrade,'target_startup_admission_pending',lambda:False),
                (automatic_upgrade,'await_target_startup_admission',AsyncMock()),
                (target.TargetStartupValidator,'validate',validator),
                (target,'get_target_library_policy_service',lambda:SimpleNamespace(recover_pending_transition=AsyncMock())),
                (target,'get_native_library_store',lambda:SimpleNamespace(work_wakeups=object())),
                (target,'get_preferences_service',lambda:preferences),
                (auth_providers,'get_auth_service',lambda:SimpleNamespace(cleanup_expired_tokens=AsyncMock())),
                (auth_providers,'get_auth_store',lambda:object()),
                (target,'run_target_one_time_migrations',AsyncMock()),
                (target,'get_target_library_operation_supervisor',lambda:SimpleNamespace(recover=AsyncMock())),
                (target,'get_library_management_recovery_service',lambda:SimpleNamespace(recover_startup=AsyncMock(return_value=recovery))),
                (target,'get_cache',lambda:SimpleNamespace(clear=AsyncMock())),
                (target,'get_disk_cache',lambda:object()),
                (target,'get_target_consumer_composition',lambda:SimpleNamespace(covers=SimpleNamespace(disk_cache=object()))),
                (target,'_server_timezone_name',lambda:'UTC'),
                (target,'get_background_workload_gate',lambda:object()),
                (service_providers,'get_wal_checkpoint_service',lambda:object()),
                (target,'start_target_operational_runtime',operational),
                (target,'get_target_library_scan_coordinator',lambda:coordinator),
                (target,'cleanup_app_state',cleanup),
                (target.TaskRegistry,'get_instance',lambda:registry),
                (registry,'cancel',cancel),(registry,'cancel_all',cancel_all),
            ]
            starters = dict(start_cache_cleanup_task='cache',start_memory_maintenance_task='memory',
                start_disk_cache_cleanup_task='disk',start_navidrome_playlist_export_task='playlist',
                start_target_scan_supervisor='scan',start_target_identification_worker='identification',
                start_target_operation_worker='operation',start_library_contribution_verification_worker='contribution',
                start_library_filesystem_watcher='watcher',start_target_wal_checkpoint_task='wal',
                start_library_revision_poller='revision',start_target_worker_watchdog=target.TARGET_WORKER_WATCHDOG_TASK_NAME)
            replacements.extend((target,name,start(label)) for name,label in starters.items())
            for owner,name,value in replacements:
                stack.enter_context(patch.object(owner,name,value))
            try:
                if failure=='cancel':
                    context = target.production_target_lifespan(SimpleNamespace())
                    entry = asyncio.create_task(context.__aenter__())
                    await asyncio.wait_for(entered_operational.wait(),2)
                    entry.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await entry
                elif failure in ('starter','operational','catalog'):
                    with self.assertRaisesRegex(RuntimeError,'fixture'):
                        async with target.production_target_lifespan(SimpleNamespace()):
                            self.fail('Startup failure must not yield')
                else:
                    with self.assertRaisesRegex(RuntimeError,'fixture body'):
                        async with target.production_target_lifespan(SimpleNamespace()):
                            self.assertTrue(created)
                            self.assertTrue(all(not task.done() for task in created))
                            raise RuntimeError('fixture body failed')
                if failure=='catalog':
                    self.assertFalse(created)
                    cleanup.assert_not_awaited()
                    coordinator.aclose.assert_not_awaited()
                else:
                    self.assertTrue(created)
                    self.assertTrue(all(task.done() for task in created),'Started tasks survived failed startup')
                    self.assertEqual(registry.get_all(),{})
                    self.assertEqual(order[:2],[target.TARGET_WORKER_WATCHDOG_TASK_NAME,'cancel_all'])
                    coordinator.aclose.assert_awaited_once()
                    cleanup.assert_awaited_once()
            finally:
                # Test-only orphan prevention even when the before-code assertions fail.
                await original_all(grace_period=0.2)

    async def test_worker_starter_failure_drains_partial_runtime(self):
        await self.exercise('starter')

    async def test_operational_await_failure_drains_runtime(self):
        await self.exercise('operational')

    async def test_startup_cancellation_drains_runtime(self):
        await self.exercise('cancel')

    async def test_normal_body_failure_preserves_shutdown_order(self):
        await self.exercise('body')

    async def test_catalog_refusal_never_starts_runtime(self):
        await self.exercise('catalog')
