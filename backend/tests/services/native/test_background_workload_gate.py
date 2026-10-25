import asyncio
from time import monotonic

import pytest

from services.native.background_workload_gate import BackgroundWorkloadGate


@pytest.mark.asyncio
async def test_waits_for_interactive_quiet_period() -> None:
    gate = BackgroundWorkloadGate(
        interactive_quiet_period=0.04,
        interactive_max_deferral=1.0,
    )
    # A completed request: begin/end arms the quiet window without holding admission.
    gate.begin_interactive_request()
    gate.end_interactive_request()

    started = monotonic()
    await gate.wait_until_available()

    assert monotonic() - started >= 0.03
    assert gate.deferred_waits == 1
    assert gate.forced_passes == 0


@pytest.mark.asyncio
async def test_active_request_blocks_past_quiet_period_until_request_finishes() -> None:
    gate = BackgroundWorkloadGate(
        interactive_quiet_period=0.03,
        interactive_max_deferral=1.0,
    )
    gate.begin_interactive_request()
    waiter = asyncio.create_task(gate.wait_until_available())

    await asyncio.sleep(0.06)
    assert not waiter.done()

    gate.end_interactive_request()
    await asyncio.sleep(0.01)
    assert not waiter.done()
    await asyncio.wait_for(waiter, timeout=0.08)


@pytest.mark.asyncio
async def test_active_request_cannot_starve_bounded_maintenance_forever() -> None:
    gate = BackgroundWorkloadGate(
        interactive_quiet_period=0.2,
        interactive_max_deferral=0.04,
    )
    gate.begin_interactive_request()

    await asyncio.wait_for(gate.wait_until_available(), timeout=0.1)
    gate.end_interactive_request()

    assert gate.forced_passes == 1


@pytest.mark.asyncio
async def test_continuous_activity_gets_one_bounded_fairness_pass() -> None:
    gate = BackgroundWorkloadGate(
        interactive_quiet_period=0.05,
        interactive_max_deferral=0.12,
    )
    gate.begin_interactive_request()
    waiter = asyncio.create_task(gate.wait_until_available())

    while not waiter.done():
        await asyncio.sleep(0.015)
        gate.begin_interactive_request()
        gate.end_interactive_request()

    await waiter

    assert gate.total_deferred_seconds >= 0.1
    assert gate.total_deferred_seconds < 0.3
    assert gate.forced_passes == 1


@pytest.mark.asyncio
async def test_scan_remains_a_strict_gate_after_fairness_deadline() -> None:
    gate = BackgroundWorkloadGate(
        interactive_quiet_period=0.01,
        interactive_max_deferral=0.02,
    )
    gate.set_scan_active(True)
    waiter = asyncio.create_task(gate.wait_until_available())

    await asyncio.sleep(0.04)
    assert not waiter.done()

    gate.set_scan_active(False)
    await asyncio.wait_for(waiter, timeout=0.1)


@pytest.mark.asyncio
async def test_wait_is_cancellable() -> None:
    gate = BackgroundWorkloadGate()
    gate.set_scan_active(True)
    waiter = asyncio.create_task(gate.wait_until_available())

    await asyncio.sleep(0)
    waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter


@pytest.mark.asyncio
async def test_proactive_warmer_slots_do_not_overlap() -> None:
    gate = BackgroundWorkloadGate()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first() -> None:
        async with gate.warmer_slot():
            first_entered.set()
            await release_first.wait()

    async def second() -> None:
        async with gate.warmer_slot():
            second_entered.set()

    first_task = asyncio.create_task(first())
    await first_entered.wait()
    second_task = asyncio.create_task(second())
    await asyncio.sleep(0)
    assert not second_entered.is_set()

    release_first.set()
    await asyncio.gather(first_task, second_task)

    assert second_entered.is_set()


@pytest.mark.asyncio
async def test_triggered_warmer_waits_for_periodic_warmer_slot() -> None:
    gate = BackgroundWorkloadGate()
    periodic_entered = asyncio.Event()
    release_periodic = asyncio.Event()
    triggered_entered = asyncio.Event()

    async def periodic() -> None:
        async with gate.warmer_slot():
            periodic_entered.set()
            await release_periodic.wait()

    async def triggered() -> None:
        triggered_entered.set()

    periodic_task = asyncio.create_task(periodic())
    await periodic_entered.wait()
    triggered_task = asyncio.create_task(gate.run_warmer_unit(triggered))
    await asyncio.sleep(0)
    assert not triggered_entered.is_set()

    release_periodic.set()
    await asyncio.gather(periodic_task, triggered_task)

    assert triggered_entered.is_set()


@pytest.mark.asyncio
async def test_admitted_warmer_unit_does_not_redefer_at_internal_boundaries() -> None:
    gate = BackgroundWorkloadGate(
        interactive_quiet_period=0.04,
        interactive_max_deferral=0.1,
    )
    nested_wait_seconds = 1.0

    async def operation() -> None:
        nonlocal nested_wait_seconds
        gate.begin_interactive_request()
        started = monotonic()
        try:
            await gate.wait_until_available()
        finally:
            nested_wait_seconds = monotonic() - started
            gate.end_interactive_request()

    await gate.run_warmer_unit(operation)

    assert nested_wait_seconds < 0.02
    assert gate.deferred_waits == 0

    # Admission is scoped to the bounded unit and must not leak to later callers.
    started = monotonic()
    await gate.wait_until_available()
    assert monotonic() - started >= 0.03
    assert gate.deferred_waits == 1


@pytest.mark.asyncio
async def test_queued_warmer_rechecks_interactive_admission_after_slot() -> None:
    gate = BackgroundWorkloadGate(
        interactive_quiet_period=0.03,
        interactive_max_deferral=1.0,
    )
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first() -> None:
        first_entered.set()
        await release_first.wait()

    async def second() -> None:
        second_entered.set()

    first_task = asyncio.create_task(gate.run_warmer_unit(first))
    await first_entered.wait()
    second_task = asyncio.create_task(gate.run_warmer_unit(second))
    await asyncio.sleep(0)

    gate.begin_interactive_request()
    release_first.set()
    await first_task
    await asyncio.sleep(0.02)
    assert not second_entered.is_set()

    gate.end_interactive_request()
    await asyncio.sleep(0.01)
    assert not second_entered.is_set()
    await asyncio.wait_for(second_task, timeout=0.08)
