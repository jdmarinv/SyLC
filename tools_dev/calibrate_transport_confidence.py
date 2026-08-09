"""Recalibration protocol for the transport confidence terms.

The score gates one decision: may this transported matte drive the full
contour reconstruction?  Its four terms are combined with fixed weights, and
at the measured operating point fb_edge alone spends 77% of the penalty budget
while the direct photometric evidence reports a 2.3% registration error.  A
weight cannot be argued about from a score -- only against what the score is
supposed to predict.

GROUND TRUTH.  MatAnyone2 infers a matte at frame A and, later, at frame B.
Transporting A's alpha to B's timestamp produces a prediction; B's own
inferred alpha is what the pipeline would have shown had it been fast enough.
The disagreement between them, measured IN THE CONTOUR BAND where the white
fringe lives, is the error the confidence claims to predict.

Limitation, stated because it bounds the conclusion: MatAnyone2 is temporal,
so B carries memory of the frames between A and B and is not an independent
oracle.  It is nonetheless the exact target the transport is approximating,
and no better reference exists without hand-labelled mattes.

Emits one row per (A, B) pair: the four terms, the age, and the measured
transport error.  What the weights SHOULD be is then a question about which
terms predict that error -- answerable from the table, not from taste.

    .venv\\Scripts\\python.exe tools_dev\\calibrate_transport_confidence.py <media> \\
        --seek 90 --seconds 60 --json calib.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sylc.synth3d_matting_service import (  # noqa: E402
    MatAnyone2Runtime,
    MatAnyone2Service,
    MatteAdvector,
)


def planar_i420(bgr):
    height, width = bgr.shape[:2]
    flat = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    plane = height * width
    quarter = plane // 4
    half = (height // 2, width // 2)
    return (flat[:plane].reshape(height, width).copy(),
            flat[plane:plane + quarter].reshape(half).copy(),
            flat[plane + quarter:].reshape(half).copy())


def band_error(predicted, truth):
    """Disagreement where the fringe lives: the contour band of the truth."""
    if predicted.shape != truth.shape:
        predicted = cv2.resize(predicted, (truth.shape[1], truth.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
    foreground = (truth >= 128).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    band = cv2.dilate(foreground, kernel) != cv2.erode(foreground, kernel)
    band = cv2.dilate(band.astype(np.uint8), np.ones((7, 7), np.uint8)) != 0
    if not np.any(band):
        return None
    difference = np.abs(predicted.astype(np.float32) -
                        truth.astype(np.float32)) / 255.0
    intersection = float(np.sum((predicted >= 128) & (truth >= 128)))
    union = float(np.sum((predicted >= 128) | (truth >= 128)))
    return {
        "band_mae": float(difference[band].mean()),
        "band_p95": float(np.percentile(difference[band], 95.0)),
        "iou": intersection / union if union > 0.0 else 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path)
    parser.add_argument("--seek", type=float, default=90.0)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    runtime = MatAnyone2Runtime.discover(ROOT)
    if runtime is None:
        raise SystemExit("complete offline MatAnyone2 runtime not found")
    capture = cv2.VideoCapture(str(args.media))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.media}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 23.976
    capture.set(cv2.CAP_PROP_POS_MSEC, args.seek * 1000.0)

    service = MatAnyone2Service(runtime, target_fps=fps, short_side=2160)
    if not service.start():
        raise SystemExit(f"worker did not start: {service.status()}")
    deadline = time.monotonic() + 40.0
    while time.monotonic() < deadline:
        if service.status()["state"] in ("ready", "running"):
            break
        if service.status()["state"] == "error":
            raise SystemExit(service.status()["error"])
        time.sleep(0.05)

    advector = MatteAdvector()
    # Pairing must happen INLINE: the luma ring holds MAX_FRAMES (64) frames,
    # so a pass over collected mattes at the end finds their source frames long
    # evicted.  Doing it as each matte lands is also the live path exactly.
    recent = []          # mattes still inside the transport horizon
    rows = []
    seen_identity = None
    index = 0
    delivered = 0
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.seconds:
            ok, bgr = capture.read()
            if not ok:
                break
            planes = planar_i420(bgr)
            pts_ms = args.seek * 1000.0 + index * 1000.0 / fps
            index += 1
            advector.note_frame(planes[0], pts_ms)
            service.configure_media(fps=fps, short_side=planes[0].shape[0])
            service.submit_yuv(planes, pts_ms)
            matte = service.latest_for_pts(pts_ms)
            if matte is None:
                continue
            identity = (matte.sequence, matte.generation)
            if identity == seen_identity:
                continue
            seen_identity = identity
            delivered += 1
            horizon = advector._max_transport_ms
            for base in recent:
                age = float(matte.pts_ms) - float(base.pts_ms)
                if age <= 1.0 or age > horizon:
                    continue
                advector._cache_key = None
                advector._last_terms = {}
                handled, out = advector._advect_with_luma(
                    base, float(matte.pts_ms))
                terms = dict(advector._last_terms)
                if not handled or out is None or out is base or not terms:
                    continue
                error = band_error(out.alpha, matte.alpha)
                if error is None:
                    continue
                rows.append({"age_ms": age,
                             "confidence": float(out.tracking_confidence),
                             **terms, **error})
            recent.append(matte)
            recent = [m for m in recent
                      if float(matte.pts_ms) - float(m.pts_ms) <= horizon]
    finally:
        service.stop(timeout=5.0)
    capture.release()

    print(f"{delivered} mattes livres, {index} frames")

    if not rows:
        print("aucune paire exploitable")
        return 1
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    data = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]}
    truth = data["band_mae"]
    print(f"\n{len(rows)} paires transportees\n")
    print(f"{'terme':>14} {'correlation avec l erreur reelle':>34}")
    for name in ("coverage", "age", "photo_edge", "photo_global", "fb_edge",
                 "evidence", "confidence"):
        if name not in data:
            continue
        column = data[name]
        if column.std() < 1e-12:
            print(f"{name:>14} {'constant':>34}")
            continue
        print(f"{name:>14} {np.corrcoef(column, truth)[0, 1]:>34.3f}")

    print(f"\n{'':>14} {'mediane':>10} {'p95':>10}")
    for name in ("band_mae", "iou", "fb_edge", "photo_edge", "age_ms"):
        column = data[name]
        print(f"{name:>14} {np.percentile(column, 50):>10.4f} "
              f"{np.percentile(column, 95):>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
