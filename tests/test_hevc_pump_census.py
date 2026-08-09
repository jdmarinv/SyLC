"""The presentation token must be observable, because losing it is silent.

Bounded delivery gives the GUI a single credit: the decoder marks one
presentation pending, emits, and waits for presentation_consumed() before it
will emit again.  Nothing in the pipeline notices when that acknowledgement
fails to come back -- the look-ahead tap sits BEFORE the drop, so the scout
keeps reporting full cadence while not one frame reaches a window, the depth
service starves, and the picture stops with every log line looking healthy.

These tests pin the census that makes the lost token visible.  They do not
pin the freeze itself: at the time of writing the cause of a lost
acknowledgement is not established, only its signature.
"""

import logging
import time

import pytest

from sylc.hevc_decode_thread import HevcDecodeThread


@pytest.fixture
def pump():
    thread = HevcDecodeThread()
    thread._bounded_delivery = True
    return thread


def _take_token(pump):
    """What the emit branch of run() does, minus the Qt signal."""
    pump._presentation_pending = True
    pump._pending_since = time.perf_counter()
    pump._pump_emitted += 1


def test_token_is_free_before_any_frame(pump):
    stats = pump.sync_stats()
    assert stats['presentation_pending'] is False
    assert stats['pending_ms'] == 0.0


def test_outstanding_token_ages(pump):
    _take_token(pump)
    time.sleep(0.02)
    stats = pump.sync_stats()
    assert stats['presentation_pending'] is True
    assert stats['pending_ms'] >= 15.0, "the age must track wall time"


def test_acknowledgement_frees_the_token_and_records_its_age(pump):
    _take_token(pump)
    time.sleep(0.02)
    pump.presentation_consumed()
    stats = pump.sync_stats()
    assert stats['presentation_pending'] is False
    assert stats['pending_ms'] == 0.0
    assert stats['acked'] == 1
    # The peak survives the release: a freeze that resolves on its own leaves
    # no other trace, and that near-miss is the whole diagnostic value.
    assert stats['pending_max_ms'] >= 15.0


def test_peak_keeps_the_worst_not_the_last(pump):
    _take_token(pump)
    time.sleep(0.03)
    pump.presentation_consumed()
    worst = pump.sync_stats()['pending_max_ms']
    _take_token(pump)
    pump.presentation_consumed()
    assert pump.sync_stats()['pending_max_ms'] == worst


def test_configure_resets_the_census(pump):
    _take_token(pump)
    pump.presentation_consumed()
    pump._sync_drop_count = 7
    pump._backpressure_drop_count = 9
    pump.configure(source=object(), mode=None, half=False, inverted=False,
                   bounded_delivery=True)
    stats = pump.sync_stats()
    assert stats['emitted'] == 0 and stats['acked'] == 0
    assert stats['sync_drops'] == 0 and stats['backpressure_drops'] == 0
    assert stats['pending_max_ms'] == 0.0
    assert stats['presentation_pending'] is False


def test_a_stuck_token_warns_and_a_healthy_one_does_not(pump, caplog):
    # Health: the census is quiet outside its 30 s window.
    pump._pump_log_at = time.perf_counter()
    with caplog.at_level(logging.WARNING):
        pump._pump_census()
    assert not caplog.records

    # Past the alarm threshold the expiry should already have run, so this
    # speaks immediately whatever the window says -- it is the guard's guard.
    caplog.clear()
    pump._presentation_pending = True
    pump._pending_since = time.perf_counter() - 4.0
    with caplog.at_level(logging.WARNING):
        pump._pump_census()
    assert any("l'expiration est fixee" in r.message for r in caplog.records)


def test_a_stuck_token_does_not_flood_the_log(pump, caplog):
    pump._presentation_pending = True
    pump._pending_since = time.perf_counter() - 4.0
    with caplog.at_level(logging.WARNING):
        for _ in range(200):          # a decode thread calls this per frame
            pump._pump_census()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, "the 5 s rate limit must hold under a freeze"


# --- the guard itself ----------------------------------------------------
# Measured 2026-08-09: emitted stuck at 361 against acked 360 for 26 s while
# 576 decoded frames went to back-pressure. One lost acknowledgement froze the
# player for good. These pin the recovery AND its accounting -- the recovery
# is only acceptable because the loss stays counted.

def test_a_fresh_token_is_never_expired(pump):
    _take_token(pump)
    assert pump._expire_stale_token() is False
    assert pump.sync_stats()['presentation_pending'] is True
    assert pump.sync_stats()['token_expired'] == 0


def test_a_token_held_past_the_deadline_is_reclaimed(pump, caplog):
    _take_token(pump)
    pump._pending_since = time.perf_counter() - (pump.TOKEN_EXPIRY_S + 0.5)
    with caplog.at_level(logging.WARNING):
        assert pump._expire_stale_token() is True
    assert pump.sync_stats()['presentation_pending'] is False
    assert pump.sync_stats()['token_expired'] == 1
    assert any("expire" in r.message for r in caplog.records), (
        "a silent recovery would hide the very defect it works around")


def test_expiry_does_not_forge_an_acknowledgement(pump):
    # `acked` must keep counting real returns only: the gap between emitted
    # and acked IS the measurement of the loss, and an expiry that closed it
    # would erase the evidence.
    _take_token(pump)
    pump._pending_since = time.perf_counter() - (pump.TOKEN_EXPIRY_S + 0.5)
    pump._expire_stale_token()
    stats = pump.sync_stats()
    assert stats['emitted'] == 1 and stats['acked'] == 0


def test_a_lost_acknowledgement_costs_one_expiry_not_a_freeze(pump):
    """The whole point: playback resumes, and it resumes exactly once."""
    _take_token(pump)                       # this delivery is never acked
    pump._pending_since = time.perf_counter() - (pump.TOKEN_EXPIRY_S + 0.5)
    assert pump._expire_stale_token() is True
    _take_token(pump)                       # the next frame goes out
    pump.presentation_consumed()            # and is acknowledged normally
    assert pump._expire_stale_token() is False
    stats = pump.sync_stats()
    assert stats['token_expired'] == 1, "one loss must cost exactly one expiry"
    assert stats['presentation_pending'] is False
