// CutGate: spike-then-calm confirmation for the source-histogram scene-cut
// signal, paired with an instantaneous pass-through for an already-
// authoritative depth-residual cut.
//
// SharedDepthService's histogram-distance detector alone fires on flashes and
// fast fades, not just hard cuts: a single frame's luma histogram can spike
// and fall back within one cycle.  Something must separate the two.
//
// This gate used to require a SECOND CONSECUTIVE exceedance to confirm.  That
// rule describes a flash, not a cut, and it was measured to never fire:
// tools_dev/cut_trace.py over 4500 frames of real 2160p content found 28 hard
// cuts, EVERY ONE of them a single-frame exceedance (the next frame's distance
// lands at 0.002-0.03 because it is the same new shot), and the old rule
// confirmed 0 of 28.  Live playback agreed exactly -- the service's `cut_src=`
// counter stayed at 0 for a whole session while `cut_bnd=` and `cut_depth=`
// climbed.  One of the three legs of shot-cut detection had never worked.
//
// The correct discriminator is the one the lookahead already used, and which
// confirmed 28 of 28 on the same content: a cut is a spike followed by QUIET,
// a flash is a spike followed by a spike BACK.  Applied retrospectively it
// needs no future frame, so it works when the lookahead is unavailable.
//
// Cost: the confirmation arrives on the cycle AFTER the cut frame, so the
// stabilizer resets one map late instead of never.  Two exceedances back to
// back stay ambiguous (flash, or a two-frame shot in fast montage) and are
// refused -- same semantics as the lookahead verdict, and the same fail-closed
// bias as the 2026-08-06 depth-pumping work, which established that a false
// cut is more costly than a late one.
//
// Pure math, header-only, no ORT/D3D; single-owner-thread use, same idiom as
// DepthStabilizer (see depth_stabilizer.h).  tests/test_cut_gate.py drives
// this class directly through the pybind11 binding.
#pragma once

class CutGate {
public:
    float histogram_threshold = 0.42f;
    // A frame is "calm" below this fraction of the threshold.  Matches the
    // lookahead's own 0.5 factor so both legs answer the same question.
    float calm_fraction = 0.5f;

    // depth_cut: an already-authoritative cut this cycle (e.g. a crossed shot
    // boundary recorded by the scout) -- always confirms immediately and
    // disarms, so an instant verdict never also produces a late one.
    // histogram_distance: the source-image histogram distance for this cycle.
    // Returns true on the cycle a cut is confirmed, which for the histogram
    // path is the cycle AFTER the cut frame.
    bool update(bool depth_cut, float histogram_distance) {
        if (depth_cut) {
            armed_ = false;
            previous_exceeded_ = false;
            return true;
        }
        const bool exceeded = histogram_distance >= histogram_threshold;
        const bool calm =
            histogram_distance < calm_fraction * histogram_threshold;
        const bool confirmed = armed_ && calm;
        // Arm only on an ISOLATED exceedance.  A flash is TWO exceedances --
        // into the bright frame and back out of it -- and the content after it
        // is the content from before, so "spike then calm" alone would confirm
        // on the way out and fire a false cut on every flash.  Requiring the
        // previous cycle to have been below threshold refuses that second
        // exceedance the right to arm, which is what separates a flash from a
        // cut using only the scalar distance.  A two-frame shot in fast
        // montage is refused by the same rule; that is the documented
        // fail-closed bias, a late reset costing less than a false one.
        armed_ = exceeded && !previous_exceeded_;
        previous_exceeded_ = exceeded;
        return confirmed;
    }

    // True when the previous cycle exceeded the threshold and this cycle will
    // decide whether that was a cut or a flash.
    bool pending() const { return armed_; }

private:
    bool armed_ = false;
    bool previous_exceeded_ = false;
};
