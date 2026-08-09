"""Trace the source-histogram cut detector over real film content.

Why this exists: the shot-cut path has three legs and none of them could be
observed.  `SharedDepthService` computes a 32-bin luma histogram distance,
routes it through `CutGate`'s two-consecutive-frame confirmation, and ORs the
result with a lookahead verdict and the depth residual.  When the reset stops
happening there is no way to tell WHICH leg failed from the outside, and
reading the code only produces hypotheses -- one of which was already refuted
by experiment.

This reproduces the two source-image legs exactly, offline, on real frames:

  * the histogram and its total-variation distance, same 32 bins, same
    Rec.709 luma reconstruction as shared_depth_service.cpp,
  * `CutGate` (two consecutive exceedances confirm; a single spike is dropped),
  * `cut_ahead` (spike at this frame AND calm to the next).

It cannot reproduce the depth-residual leg, which needs the model.  What it
CAN answer is the structural question those two legs turn on: at a real hard
cut, how many CONSECUTIVE frames actually exceed the threshold?

Usage:
    .venv\\Scripts\\python.exe tools_dev\\cut_trace.py <media> --start-frame 30000 --count 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# shared_depth_service.cpp: kHistogramBins, DepthStabilizer::scene_cut_threshold
HISTOGRAM_BINS = 32
SCENE_CUT_THRESHOLD = 0.42
# The service bins the luma of the model input grid, not the full frame.
GRID_W, GRID_H = 756, 322


class CutGate:
    """Exact port of include/cut_gate.h as shipped: two CONSECUTIVE
    exceedances confirm."""

    def __init__(self, histogram_threshold: float = SCENE_CUT_THRESHOLD):
        self.histogram_threshold = histogram_threshold
        self.pending = False

    def update(self, depth_cut: bool, histogram_distance: float) -> bool:
        if depth_cut:
            self.pending = False
            return True
        if histogram_distance >= self.histogram_threshold:
            if self.pending:
                self.pending = False  # confirmed: re-arm for the next shot
                return True
            self.pending = True
            return False
        self.pending = False
        return False


class SpikeThenCalmGate:
    """Proposed replacement: confirm on spike-then-calm, one frame late.

    A flash is a spike and a spike BACK; a cut is a spike and then quiet.
    Waiting for a second consecutive exceedance therefore describes a flash,
    not a cut, which is why the shipped gate confirms 0 of the hard cuts in
    real content while the lookahead's spike-then-calm test confirms all of
    them.  This is that same test applied retrospectively, so it needs no
    future frame: it costs one frame of latency instead of never firing.

    Returns True on the cycle AFTER the cut frame.  Two exceedances back to
    back stay ambiguous (flash or two-frame shot) and are refused, matching
    the lookahead's semantics and the fail-closed bias that the 2026-08-06
    depth-pumping work established.
    """

    def __init__(self, histogram_threshold: float = SCENE_CUT_THRESHOLD,
                 calm_fraction: float = 0.5):
        self.histogram_threshold = histogram_threshold
        self.calm_fraction = calm_fraction
        self.armed = False
        self.previous_exceeded = False

    def update(self, depth_cut: bool, histogram_distance: float) -> bool:
        if depth_cut:
            self.armed = False
            self.previous_exceeded = False
            return True
        exceeded = histogram_distance >= self.histogram_threshold
        calm = histogram_distance < self.calm_fraction * self.histogram_threshold
        confirmed = self.armed and calm
        # Only an ISOLATED exceedance may arm: a flash exceeds twice, into the
        # bright frame and back out, and would otherwise confirm on the way out.
        self.armed = exceeded and not self.previous_exceeded
        self.previous_exceeded = exceeded
        return confirmed


def frame_histogram(bgr: np.ndarray) -> np.ndarray:
    """Same luma and binning as shared_depth_service.cpp."""
    small = cv2.resize(bgr, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
    rgb = small[:, :, ::-1].astype(np.float32) / 255.0
    y = np.clip(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] +
                0.0722 * rgb[..., 2], 0.0, 1.0)
    bins = np.minimum(HISTOGRAM_BINS - 1,
                      (y * HISTOGRAM_BINS).astype(np.int32))
    hist = np.bincount(bins.ravel(),
                       minlength=HISTOGRAM_BINS).astype(np.float64)
    return hist / float(y.size)


def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path)
    parser.add_argument("--start-frame", type=int, default=30000)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=SCENE_CUT_THRESHOLD)
    parser.add_argument("--sweep", action="store_true",
                        help="scan arming thresholds instead of scoring one")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.media))
    if not capture.isOpened():
        print(f"FAIL: cannot open {args.media}")
        return 2
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    distances: list[float] = []
    previous: np.ndarray | None = None
    for _ in range(args.count):
        ok, frame = capture.read()
        if not ok:
            break
        histogram = frame_histogram(frame)
        distances.append(0.0 if previous is None
                         else total_variation(histogram, previous))
        previous = histogram
    capture.release()
    if len(distances) < 3:
        print("FAIL: not enough frames decoded")
        return 2

    if args.sweep:
        # The absolute distance is a weak discriminator: the cut measured at
        # 96.97 s of the x264 encode reads 0.280 against a 0.42 threshold and
        # is invisible to both source legs, yet the frame after it reads 0.004
        # -- a 70:1 shape that no flash and no motion produces.  Scan what
        # lowering the ARMING threshold costs while keeping the calm
        # confirmation, so the trade is chosen from data rather than nerve.
        print(f"{'seuil':>7} {'confirmations':>14} {'ratio median pic/suivant':>26}")
        for threshold in (0.42, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10):
            gate = SpikeThenCalmGate(threshold)
            hits = [i for i, d in enumerate(distances) if gate.update(False, d)]
            ratios = sorted(
                distances[i - 1] / max(distances[i], 1e-6)
                for i in hits if i >= 1)
            median = ratios[len(ratios) // 2] if ratios else float("nan")
            print(f"{threshold:>7.2f} {len(hits):>14} {median:>26.1f}")
        return 0

    # Leg 1: CutGate confirmation.  Leg 2: spike here, calm to the next frame.
    gate = CutGate(args.threshold)
    source_cut = [gate.update(False, d) for d in distances]
    cut_ahead = [
        distances[i] >= args.threshold and
        distances[i + 1] < 0.5 * args.threshold
        for i in range(len(distances) - 1)] + [False]

    exceed = [d >= args.threshold for d in distances]
    # Length of every run of consecutive exceedances: this is the number the
    # two-frame confirmation depends on.
    runs: list[int] = []
    run = 0
    for flag in exceed:
        if flag:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)

    frames = len(distances)
    print(f"frames analysees   : {frames} (depuis {args.start_frame})")
    print(f"seuil              : {args.threshold}")
    print(f"depassements       : {sum(exceed)}")
    print(f"salves de depassements consecutifs : {len(runs)}")
    if runs:
        one = sum(1 for r in runs if r == 1)
        print(f"   longueur 1 frame : {one}/{len(runs)}  "
              f"({100.0 * one / len(runs):.0f}%)")
        print(f"   longueur >=2     : {len(runs) - one}/{len(runs)}")
        print(f"   longueur max     : {max(runs)}")
    print()
    missed = sum(1 for i in range(frames)
                 if exceed[i] and not source_cut[i] and not cut_ahead[i])
    proposed_gate = SpikeThenCalmGate(args.threshold)
    proposed = [proposed_gate.update(False, d) for d in distances]
    print(f"leg 1  source_cut confirme : {sum(source_cut)}")
    print(f"leg 2  cut_ahead           : {sum(cut_ahead)}")
    print(f"coupes franches manquees par les DEUX legs source : {missed}")
    print()
    # The proposed gate confirms one frame late, so a hit at i+1 answers the
    # cut at i.  Count agreement against the lookahead's verdict, which is the
    # only leg measured to work.
    agreed = sum(1 for i in range(frames - 1) if cut_ahead[i] and proposed[i + 1])
    spurious = sum(1 for i in range(1, frames)
                   if proposed[i] and not cut_ahead[i - 1])
    print(f"PROPOSE  spike-then-calm retrospectif : {sum(proposed)} confirmations")
    print(f"   d'accord avec cut_ahead (leg 2)    : {agreed}/{sum(cut_ahead)}")
    print(f"   confirmations sans coupe leg 2     : {spurious}")

    top = sorted(range(frames), key=lambda i: distances[i], reverse=True)[:12]
    print("\nplus fortes transitions :")
    print(f"{'frame':>8} {'TV':>7} {'TV suivante':>12} {'leg1':>6} {'leg2':>6}")
    for i in sorted(top):
        nxt = distances[i + 1] if i + 1 < frames else float("nan")
        print(f"{args.start_frame + i:>8} {distances[i]:>7.3f} {nxt:>12.3f} "
              f"{str(source_cut[i]):>6} {str(cut_ahead[i]):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
