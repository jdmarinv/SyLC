import math
from pathlib import Path

import cv2
import numpy as np

from sylc.synth3d_matting_service import (
    MatAnyone2Runtime,
    MatAnyone2Service,
    MatteAdvector,
    MatteFrame,
)

def _advector_with_horizon(horizon_ms, monkeypatch):
    monkeypatch.setenv(
        "SYLC_SYNTH3D_CONTOUR_LOCK_MAX_AGE_MS", repr(horizon_ms))
    return MatteAdvector()


def _runtime_stub():
    root = Path("unused")
    return MatAnyone2Runtime(root, root, root, root, root)


def test_matanyone_defaults_keep_full_yuv_and_cap_model_at_512(monkeypatch):
    monkeypatch.delenv("SYLC_MATANYONE2_AUTO_CAP", raising=False)
    monkeypatch.delenv("SYLC_MATANYONE2_TRANSPORT", raising=False)
    service = MatAnyone2Service(
        _runtime_stub(), target_fps=23.976, short_side=808)
    assert service.short_side == 512
    assert service.auto_cap == 512
    assert service.transport == "yuv420"


def test_direct_partition_percentile_matches_numpy():
    rng = np.random.default_rng(4207)
    values = rng.normal(size=(96, 160)).astype(np.float32)
    mask = rng.random(values.shape) > 0.37
    for q in (0.0, 50.0, 65.0, 75.0, 90.0, 100.0):
        sampled = values[::2, ::2][mask[::2, ::2]]
        if sampled.size > 4096:
            sampled = sampled[::int(np.ceil(sampled.size / 4096.0))]
        expected = float(np.percentile(sampled, q))
        actual = MatteAdvector._masked_percentile(values, mask, q, -1.0)
        assert actual == expected


def _advector_with_latency(latencies):
    """Feed first-sight ages so the advector measures a delivery latency."""
    advector = MatteAdvector()
    alpha = np.zeros((8, 8), dtype=np.uint8)
    for index, latency in enumerate(latencies):
        matte = MatteFrame(index + 1, 1, 1000.0, alpha, 0.0, False, False)
        advector._note_arrival(matte, 1000.0 + latency)
    return advector


def test_age_grace_defaults_to_the_historical_constant():
    # No delivered matte yet: fail closed on the pre-existing 120 ms grace.
    assert MatteAdvector()._age_grace_ms() == MatteAdvector.MIN_AGE_GRACE_MS


def test_fast_pipeline_keeps_the_historical_age_penalty():
    # 54-70 ms inference (the measured 1080p path) is already inside the
    # historical grace, so the score must be untouched by the indexing.
    advector = _advector_with_latency([54.0, 58.0, 70.8, 62.0])
    assert advector._age_grace_ms() == MatteAdvector.MIN_AGE_GRACE_MS


def test_age_grace_follows_measured_latency_and_is_capped():
    advector = _advector_with_latency([313.0, 350.0, 420.0, 380.0])
    grace = advector._age_grace_ms()
    # 65% of the hard horizon leaves evidence margin at 420 ms while the age
    # term still demotes a matte at the 520 ms expiry boundary.
    assert grace == (advector.AGE_GRACE_HORIZON_RATIO *
                     advector._max_transport_ms)
    assert 313.0 < grace < 350.0
    # A single late outlier must not move a median-based estimate.
    assert _advector_with_latency(
        [54.0, 58.0, 500.0, 62.0])._age_grace_ms() == (
            MatteAdvector.MIN_AGE_GRACE_MS)


def test_first_arrival_self_indexes_but_cannot_bypass_the_cap():
    normal = _advector_with_latency([313.0])
    assert normal._age_grace_ms() == 313.0
    assert math.exp(-max(
        0.0, 313.0 - normal._age_grace_ms()) / normal.AGE_DECAY_MS) == 1.0

    late = _advector_with_latency([420.0])
    assert late._age_grace_ms() == (
        late.AGE_GRACE_HORIZON_RATIO * late._max_transport_ms)
    assert late._age_grace_ms() < 420.0
    assert math.exp(-max(
        0.0, 420.0 - late._age_grace_ms()) / late.AGE_DECAY_MS) < 1.0


def test_contour_gate_stays_reachable_at_4k_latency():
    """The premium path must not be closed by arithmetic on 4K/10-bit.

    With a fixed 120 ms grace the age term alone capped the score at 0.632 for
    the measured 313 ms inference, so no amount of image evidence could reach
    the promotion threshold on exactly the content that motivated the
    transport.  The horizon, not the score, must be the binding constraint.
    """
    advector = _advector_with_latency([313.0, 350.0, 420.0, 380.0])
    grace = advector._age_grace_ms()
    threshold = MatteAdvector.CONTOUR_MIN_CONFIDENCE
    decay = MatteAdvector.AGE_DECAY_MS
    ceiling = lambda age: math.exp(-max(0.0, age - grace) / decay)

    assert ceiling(313.0) > threshold
    # Merely exceeding the threshold is not useful: retain enough headroom for
    # a plausible 15% photometric/FB evidence loss at the measured upper bound.
    assert ceiling(420.0) * 0.85 > threshold
    # And the decision itself, not just the ceiling, must be available there.
    assert 313.0 < advector._contour_age_limit_ms(grace)
    assert 420.0 < advector._contour_age_limit_ms(grace)
    # Still monotone, and still demoted rather than accepted at the horizon.
    horizon = advector._max_transport_ms
    assert ceiling(horizon) < threshold
    assert ceiling(horizon) > advector._min_confidence


def test_expiring_matte_is_never_promoted_at_any_horizon(monkeypatch):
    """Sweep the band, do not sample it.

    The float defect this replaced showed up at horizons 418..462 — 420 and
    462 among them, neither of which is a boundary of anything.  A handful of
    hand-picked singular points would have missed it entirely, so the only
    honest shape for this test is a sweep of the whole configurable range at a
    step finer than the effect.
    """
    step = 0.25
    horizon = 80.0
    checked = 0
    while horizon <= 700.0:
        advector = _advector_with_horizon(horizon, monkeypatch)
        assert advector._max_transport_ms == horizon
        for latency in (0.0, 120.0, 313.0, 420.0, 0.65 * horizon, horizon):
            if latency:
                advector._arrivals.clear()
                advector._arrivals.append(latency)
            grace = advector._age_grace_ms()
            limit = advector._contour_age_limit_ms(grace)
            # The invariant, stated exactly as the transport applies it.
            assert not (horizon < limit), (
                f"horizon={horizon!r} latency={latency!r} "
                f"grace={grace!r} limit={limit!r}")
            checked += 1
        horizon += step
    assert checked > 10_000


def test_promotion_limit_never_exceeds_the_horizon_under_random_config(
        monkeypatch):
    # Same invariant, off the grid: a step size can always be unlucky.
    rng = np.random.default_rng(20260807)
    for horizon in rng.uniform(80.0, 700.0, 20_000):
        advector = _advector_with_horizon(float(horizon), monkeypatch)
        advector._arrivals.append(float(rng.uniform(0.0, horizon)))
        grace = advector._age_grace_ms()
        assert advector._contour_age_limit_ms(grace) <= advector._max_transport_ms


def test_promotion_travels_as_a_decision_not_a_score():
    # The renderer holds no threshold: what it reads is the boolean.  A fresh
    # network observation stays promotable, a codec-MV projection never is.
    alpha = np.zeros((8, 8), dtype=np.uint8)
    fresh = MatteFrame(1, 1, 0.0, alpha, 0.0, False, False)
    assert fresh.allow_contour is True
    assert 'allow_contour' in MatteAdvector().status()


def test_parallel_contour_flow_is_identical_to_sequential(monkeypatch):
    rng = np.random.default_rng(1949)
    base = rng.integers(16, 236, size=(96, 160), dtype=np.uint8)
    current = np.empty_like(base)
    current[:, :3] = base[:, :1]
    current[:, 3:] = base[:, :-3]
    alpha = np.zeros((96, 160), dtype=np.uint8)
    cv2.ellipse(alpha, (80, 48), (24, 35), 0, 0, 360, 255, -1)
    matte = MatteFrame(1, 1, 1000.0, alpha, 0.0, False, False)

    outputs = {}
    statuses = {}
    advectors = []
    try:
        for mode in ("bidirectional", "parallel"):
            monkeypatch.setenv("SYLC_SYNTH3D_CONTOUR_LOCK_MODE", mode)
            advector = MatteAdvector()
            advectors.append(advector)
            advector.note_frame(base, 1000.0)
            advector.note_frame(current, 1041.7)
            outputs[mode] = advector.advect(matte, 1041.7)
            statuses[mode] = advector.status()

        assert (outputs["parallel"] is None) == (
            outputs["bidirectional"] is None)
        if outputs["parallel"] is not None:
            assert np.array_equal(
                outputs["parallel"].alpha, outputs["bidirectional"].alpha)
            assert np.array_equal(
                outputs["parallel"].reliability,
                outputs["bidirectional"].reliability)
        for field in ("kind", "confidence", "local_reject_pct", "sparse_pct"):
            assert statuses["parallel"][field] == statuses["bidirectional"][field]
    finally:
        for advector in advectors:
            if advector._flow_pool is not None:
                advector._flow_pool.shutdown()
