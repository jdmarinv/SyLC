"""The scout must see a low-contrast hard cut, and must not see a flash.

The scout published on a bare threshold crossing at 0.42, so it could only be
made safe against flashes by keeping the bar above them -- and a real cut that
lands below the bar was simply invisible. Measured on the x264 encode: the cut
at 96.97 s reads TV 0.296 against ambient 0.031, unmistakable by shape and
invisible by magnitude, and 7 of 16 hard cuts in that window were missed.

These tests pin the shape discriminator that replaced the bare threshold, so
neither half of the trade can silently regress.
"""

import numpy as np
import pytest

from sylc.lookahead_scout import CUT_TV, LookAheadScout


FRAME_MS = 1000.0 / 23.976


def _frame(level, size=128, seed=0):
    """A flat-ish frame at a given luma level, with a little structure."""
    rng = np.random.default_rng(seed)
    base = np.full((size, size), level, dtype=np.int16)
    base += rng.integers(-4, 5, size=(size, size))
    return np.clip(base, 0, 255).astype(np.uint8)


def _run(levels):
    """Push one frame per level; return the published cut pts."""
    scout = LookAheadScout(reorder_depth=0)
    for index, level in enumerate(levels):
        scout.push(_frame(level, seed=level), index * FRAME_MS)
    return list(scout._cut_events)


def test_hard_cut_is_published_one_frame_late():
    # Steady, one jump, steady again: the cut is dated at the NEW shot's
    # first frame even though it is confirmed on the frame after.
    events = _run([40, 40, 40, 200, 200, 200])
    assert events == [pytest.approx(3 * FRAME_MS)]


def test_flash_is_not_published():
    # Into the bright frame and back out: two exceedances, and the content
    # after is the content from before.  A bare threshold cannot tell this
    # from a cut, which is the whole reason the bar had to sit at 0.42.
    assert _run([40, 40, 200, 40, 40, 40]) == []


def _pair_with_tv(fraction, seed=7):
    """Two frames whose TV distance tracks the fraction of moved pixels.

    Two flat levels have disjoint histograms and score 1.0; a real
    low-contrast cut keeps most of its tonal mass and moves a fraction of it.
    """
    from sylc.lookahead_scout import _subsample, _tv_distance, CUT_SIDE
    a = _frame(96, seed=1)
    rng = np.random.default_rng(seed)
    b = a.copy()
    moved = rng.random(a.shape) < fraction
    b[moved] = _frame(180, seed=2)[moved]
    mini = lambda f: _subsample(f, CUT_SIDE).astype(np.float32) / 255.0
    return a, b, _tv_distance(mini(a), mini(b))


def _publishes(a, b, cut_tv=None):
    scout = (LookAheadScout(reorder_depth=0) if cut_tv is None
             else LookAheadScout(reorder_depth=0, cut_tv=cut_tv))
    for index, frame in enumerate((a, a, b, b, b)):
        scout.push(frame, index * FRAME_MS)
    return scout.cuts_published


def test_a_cut_just_above_the_bar_is_published():
    # Pinned against the CONFIGURED bar, not a literal: this test broke when
    # the default moved back to 0.42 because it hardcoded the straddle.
    a, b, tv = _pair_with_tv(min(0.95, CUT_TV + 0.08))
    assert tv >= CUT_TV, f"fixture below the configured bar: {tv} < {CUT_TV}"
    assert _publishes(a, b) == 1


def test_the_measured_low_contrast_cut_needs_a_lower_bar():
    """Pins the known miss AND its remedy, depending on neither default.

    The x264 cut at 96.97 s reads TV 0.296 -- 9.6x its ambient, a hard cut by
    shape.  Any bar above it is blind to that class.  Restoring 0.42 was a
    live-observed trade against judder, not a claim that 0.296 is not a cut,
    so both halves are recorded here rather than in a comment that rots.
    """
    a, b, tv = _pair_with_tv(0.30)
    assert 0.25 <= tv <= 0.40, f"fixture drifted out of the band: {tv}"
    assert _publishes(a, b, cut_tv=0.28) == 1      # seen with the lower bar
    assert _publishes(a, b, cut_tv=0.42) == 0      # invisible at the default


def test_arming_threshold_stays_far_above_ambient_motion():
    # Ambient frame-to-frame distance on real content sits near 0.03; the
    # bar must keep an order of magnitude of headroom over it.
    assert CUT_TV >= 8.0 * 0.031


def test_static_content_publishes_nothing():
    assert _run([40] * 8) == []
