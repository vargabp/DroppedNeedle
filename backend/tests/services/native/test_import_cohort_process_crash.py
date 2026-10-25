"""SIGKILL only fresh test children; never a production process or container."""
import asyncio
import importlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import unittest

import msgspec


_CHILD = r"""
import asyncio,importlib,json,os,signal,sys,threading,types
data=json.loads(sys.stdin.buffer.read())
if data.get('backend_root'):
    sys.path.insert(0,data['backend_root'])
import infrastructure.persistence.native_library_schema
for key,name in data['modules']:
    module=types.ModuleType(name);sys.modules[name]=module
    exec(compile(data['sources'][key],name,'exec'),module.__dict__)
from core.config import Settings
from services.preferences_service import PreferencesService
from infrastructure.persistence.native_library_store import NativeLibraryStore
from services.native.library_filesystem_coordinator import LibraryFilesystemCoordinator
from services.native.library_policy_resolver import LibraryPolicyResolver
from services.native.library_policy_service import LibraryPolicyService
from services.native.library_policy_reconciliation_service import LibraryPolicyReconciliationService
from services.native.target_library_policy_service import TargetLibraryPolicyService
from api.v1.schemas.library_policies import TypedLibrarySettings
from pathlib import Path
import msgspec
def crash():
    os.kill(os.getpid(),signal.SIGKILL)
async def run():
    path=Path(data['config'])
    prefs=PreferencesService(Settings(config_file_path=path,root_app_dir=path.parent))
    store=NativeLibraryStore(Path(data['database']),threading.Lock(),
        _cohort_schema_candidate_enabled=True,_cohort_policy_invalidation_enabled=True,
        _cohort_policy_transition_enabled=True)
    resolver=lambda: LibraryPolicyResolver(prefs.get_typed_library_settings_raw())
    settings=LibraryPolicyService(prefs,None,resolver,lambda:None)
    async def forbidden(*args,**kwargs):
        raise AssertionError('No scan authorized')
    reconciliation=LibraryPolicyReconciliationService(store,resolver,types.SimpleNamespace(request_run=forbidden))
    service=TargetLibraryPolicyService(settings,reconciliation,store,transition_lock=asyncio.Lock(),
        _cohort_settings_candidate_enabled=True,_cohort_recovery_candidate_enabled=True,
        _cohort_filesystem=LibraryFilesystemCoordinator())
    phase=data['phase']
    if phase=='prepared':
        original=reconciliation.prepare_boundary
        async def cut(**kwargs):
            await original(**kwargs)
            crash()
        reconciliation.prepare_boundary=cut
    elif phase=='config':
        original=settings.persist_settings
        def cut(*args,**kwargs):
            original(*args,**kwargs)
            crash()
        settings.persist_settings=cut
    elif phase in ('uncommitted_event','recover_uncommitted_event'):
        module=importlib.import_module('infrastructure.persistence.import_cohort_binding')
        original=module.invalidate_cohort_policy_candidate
        def cut(*args,**kwargs):
            original(*args,**kwargs)
            crash()
        module.invalidate_cohort_policy_candidate=cut
    elif phase=='committed':
        original=store.commit_policy_transition
        async def cut(**kwargs):
            result=await original(**kwargs)
            crash()
            return result
        store.commit_policy_transition=cut
    elif phase=='recover_aborted':
        original=store.abort_policy_transition
        async def cut(**kwargs):
            await original(**kwargs)
            crash()
        store.abort_policy_transition=cut
    else:
        raise AssertionError('Unknown test checkpoint')
    if phase.startswith('recover_'):
        await service.recover_pending_transition()
    else:
        await service.save_settings(msgspec.convert(data['proposed'],type=TypedLibrarySettings),
            expected_policy_revision=data['previous_revision'])
    raise AssertionError('Child did not terminate at the selected checkpoint')
asyncio.run(run())
"""


class CohortProcessCrashTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        module = importlib.import_module('tests.services.native.test_import_cohort_preferences_recovery')
        self.base = module.CohortPreferencesRecoveryTests()
        await self.base.asyncSetUp()
        self.addCleanup(self.base.doCleanups)

    async def crash(self, phase):
        main = sys.modules['__main__']
        if hasattr(main, 'payload') and hasattr(main, 'modules'):
            modules = main.modules
            sources = {key: main.payload[key] for key, _ in modules}
            backend_root = None
        else:
            # Ordinary repository pytest uses its real checkout, never the
            # deployed image's old runtime modules. No skipped crash checks.
            modules, sources = [], {}
            backend_root = str(Path(__file__).resolve().parents[3])
        data = dict(modules=modules, sources=sources, backend_root=backend_root,
            phase=phase, config=str(self.base.config), database=str(self.base.fixture.path),
            proposed=msgspec.to_builtins(self.base.proposed),
            previous_revision=self.base.fixture.old.policy_revision)
        process = subprocess.Popen([sys.executable, '-c', _CHILD], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            _, error = await asyncio.to_thread(process.communicate, json.dumps(data).encode(), timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.communicate)
            self.fail('Disposable child failed to reach SIGKILL checkpoint within20seconds')
        finally:
            if process.poll() is None:
                process.kill()
                await asyncio.to_thread(process.communicate)
        self.assertEqual(process.returncode, -signal.SIGKILL,
            'Child did not stop by checkpoint SIGKILL: ' + error.decode(errors='replace')[-1000:])
        self.base.reopen()

    async def assert_recovered_once(self):
        self.assertTrue(await self.base.service.recover_pending_transition())
        self.assertEqual(self.base.revision(), self.base.fixture.current.policy_revision)
        self.assertEqual((await self.base.store.get_policy_transition())['state'], 'completed')
        events = self.base.fixture.events()
        self.assertEqual(len(events), 1)
        self.base.reopen()
        self.assertFalse(await self.base.service.recover_pending_transition())
        self.assertEqual(self.base.fixture.events(), events)

    async def test_sigkill_after_prepare_aborts_without_config_or_catalog_change(self):
        before = self.base.fixture.rows()
        await self.crash('prepared')
        self.assertEqual((await self.base.store.get_policy_transition())['state'], 'prepared')
        self.assertEqual(self.base.revision(), self.base.fixture.old.policy_revision)
        self.assertTrue(await self.base.service.recover_pending_transition())
        self.assertEqual((await self.base.store.get_policy_transition())['state'], 'aborted')
        self.assertEqual(self.base.fixture.events(), [])
        self.assertEqual(self.base.fixture.rows(), before)

    async def test_sigkill_after_config_file_save_finishes_once(self):
        before = self.base.fixture.rows()
        await self.crash('config')
        self.assertEqual(self.base.revision(), self.base.fixture.current.policy_revision)
        self.assertEqual(self.base.fixture.events(), [])
        self.assertEqual(self.base.fixture.rows(), before)
        await self.assert_recovered_once()

    async def test_sigkill_after_uncommitted_event_rolls_back_then_recovers(self):
        before = self.base.fixture.rows()
        await self.crash('uncommitted_event')
        self.assertEqual((await self.base.store.get_policy_transition())['state'], 'prepared')
        self.assertEqual((await self.base.store.get_pending_policy())['desired_policy_revision'], self.base.fixture.old.policy_revision)
        self.assertEqual(self.base.fixture.events(), [])
        self.assertEqual(self.base.fixture.rows(), before)
        await self.assert_recovered_once()

    async def test_sigkill_after_committed_boundary_does_not_duplicate_event(self):
        await self.crash('committed')
        self.assertEqual(self.base.revision(), self.base.fixture.current.policy_revision)
        self.assertEqual((await self.base.store.get_policy_transition())['state'], 'completed')
        events = self.base.fixture.events()
        self.assertEqual(len(events), 1)
        self.assertFalse(await self.base.service.recover_pending_transition())
        self.assertEqual(self.base.fixture.events(), events)

    async def test_sigkill_during_recovery_event_rolls_back_for_next_recovery(self):
        await self.base.prepare(persist=True)
        await self.crash('recover_uncommitted_event')
        self.assertEqual((await self.base.store.get_policy_transition())['state'], 'prepared')
        self.assertEqual(self.base.fixture.events(), [])
        await self.assert_recovered_once()

    async def test_sigkill_after_committed_recovery_abort_is_idempotent(self):
        await self.base.prepare()
        await self.crash('recover_aborted')
        self.assertEqual((await self.base.store.get_policy_transition())['state'], 'aborted')
        self.assertEqual(self.base.revision(), self.base.fixture.old.policy_revision)
        self.assertFalse(await self.base.service.recover_pending_transition())
        self.assertEqual(self.base.fixture.events(), [])
