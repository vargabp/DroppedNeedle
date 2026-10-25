"""Real coordinator/SQL tests for publisher-shaped child-task lease scopes."""
import asyncio
import importlib
import threading
import unittest

from services.native.library_filesystem_coordinator import LibraryFilesystemCoordinator


class CohortLeaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_publisher_child_can_borrow_held_root_without_deadlock(self):
        coordinator = LibraryFilesystemCoordinator()
        async def critical():
            async with coordinator._read_for_cohort_candidate('root'):
                self.assertTrue(coordinator._state('root').writer_active)
        async with coordinator.write_many(['root']):
            await asyncio.wait_for(asyncio.create_task(critical()), 1)

    async def test_native_preparation_and_commit_keep_competing_writer_excluded(self):
        module = importlib.import_module('tests.infrastructure.test_import_cohort_atomic_commit')
        fixture = module.CohortAtomicCommitTests()
        await fixture.asyncSetUp()
        self.addCleanup(fixture.doCleanups)
        coordinator = fixture.fixture.service._filesystem
        writer_entered = asyncio.Event()
        started = asyncio.Event()
        async def competitor():
            started.set()
            async with coordinator.write('root'):
                writer_entered.set()
        async def critical():
            fixture.writes, _, _ = await fixture.fixture.prepare()
            self.assertFalse(writer_entered.is_set())
            identifiers = await fixture.commit()
            self.assertEqual(len(identifiers), 2)
            self.assertFalse(writer_entered.is_set())
        async with coordinator.write_many(['root']):
            competing = asyncio.create_task(competitor())
            await started.wait()
            await asyncio.wait_for(asyncio.create_task(critical()), 2)
            self.assertFalse(writer_entered.is_set())
        await asyncio.wait_for(competing, 1)
        self.assertTrue(writer_entered.is_set())

    async def test_inherited_scope_is_revoked_after_writer_exit(self):
        coordinator = LibraryFilesystemCoordinator()
        go = asyncio.Event()
        started = asyncio.Event()
        entered = asyncio.Event()
        async def old_child():
            await go.wait()
            started.set()
            async with coordinator._read_for_cohort_candidate('root'):
                entered.set()
        async with coordinator.write('root'):
            child = asyncio.create_task(old_child())
        async with coordinator.write('root'):
            go.set()
            await started.wait()
            self.assertFalse(entered.is_set())
        await asyncio.wait_for(child, 1)
        self.assertTrue(entered.is_set())

    async def _escaped_borrow(self, *, cancel_writer):
        coordinator = LibraryFilesystemCoordinator()
        borrowing = asyncio.Event()
        release = asyncio.Event()
        body_exited = asyncio.Event()
        competitor_started = asyncio.Event()
        competitor_entered = asyncio.Event()
        children = []
        async def child():
            async with coordinator._read_for_cohort_candidate('root'):
                borrowing.set()
                await release.wait()
        async def parent():
            async with coordinator.write('root'):
                children.append(asyncio.create_task(child()))
                await borrowing.wait()
                body_exited.set()
        async def competitor():
            competitor_started.set()
            async with coordinator.write('root'):
                competitor_entered.set()
        owner = asyncio.create_task(parent())
        await body_exited.wait()
        competing = asyncio.create_task(competitor())
        await competitor_started.wait()
        if cancel_writer:
            owner.cancel()
        self.assertFalse(owner.done())
        self.assertFalse(competitor_entered.is_set())
        release.set()
        if cancel_writer:
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(owner, 1)
        else:
            await asyncio.wait_for(owner, 1)
        await asyncio.wait_for(asyncio.gather(*children, competing), 1)
        self.assertTrue(competitor_entered.is_set())

    async def test_writer_drains_existing_borrow_before_releasing_root(self):
        await self._escaped_borrow(cancel_writer=False)

    async def test_cancellation_cannot_release_writer_while_child_borrows(self):
        await self._escaped_borrow(cancel_writer=True)

    async def test_root_scope_does_not_authorize_another_root(self):
        coordinator = LibraryFilesystemCoordinator()
        held = asyncio.Event()
        release = asyncio.Event()
        started = asyncio.Event()
        entered = asyncio.Event()
        async def holder():
            async with coordinator.write('other'):
                held.set()
                await release.wait()
        task = asyncio.create_task(holder())
        await held.wait()
        async def child():
            started.set()
            async with coordinator._read_for_cohort_candidate('other'):
                entered.set()
        async with coordinator.write('root'):
            reader = asyncio.create_task(child())
            await started.wait()
            self.assertFalse(entered.is_set())
        release.set()
        await asyncio.wait_for(asyncio.gather(task, reader), 1)

    async def test_scope_is_specific_to_coordinator_instance(self):
        first, second = LibraryFilesystemCoordinator(), LibraryFilesystemCoordinator()
        async with first.write('root'):
            async with second._read_for_cohort_candidate('root'):
                self.assertEqual(second._state('root').readers, 1)
                self.assertFalse(second._state('root').writer_active)

    async def test_normal_read_api_is_not_made_reentrant(self):
        coordinator = LibraryFilesystemCoordinator()
        started, entered = asyncio.Event(), asyncio.Event()
        async def reader():
            started.set()
            async with coordinator.read('root'):
                entered.set()
        async with coordinator.write('root'):
            child = asyncio.create_task(reader())
            await started.wait()
            self.assertFalse(entered.is_set())
        await asyncio.wait_for(child, 1)

    async def test_context_copy_into_worker_loop_cannot_borrow_parent_lease(self):
        coordinator = LibraryFilesystemCoordinator()
        started, entered = threading.Event(), threading.Event()
        async def worker_read():
            started.set()
            async with coordinator._read_for_cohort_candidate('root'):
                entered.set()
        async with coordinator.write('root'):
            worker = asyncio.create_task(asyncio.to_thread(lambda: asyncio.run(worker_read())))
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            self.assertFalse(entered.is_set())
        await asyncio.wait_for(worker, 1)
        self.assertTrue(entered.is_set())
