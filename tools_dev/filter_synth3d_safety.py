"""Build a conservative flow-transported release envelope for stereo safety."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _transport(previous: np.ndarray, packed_flow: np.ndarray) -> np.ndarray:
    flow = packed_flow[..., :2].astype(np.float32) / 65535.0
    flow = (flow - 0.5) * 128.0
    height, width = previous.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    return cv2.remap(previous.astype(np.float32),
                     xx - flow[..., 0], yy - flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rise-per-map", type=float, default=0.012)
    parser.add_argument("--depth-sequence", type=Path)
    parser.add_argument("--focus-time", type=float, default=134.128)
    parser.add_argument("--sub-roi", nargs=4, type=int,
                        default=(140, 180, 300, 340))
    parser.add_argument("--window", type=float, default=0.75)
    args = parser.parse_args()

    data = np.load(args.capture)
    geometry = data["geometry"]
    transport = data["transport"]
    surface = data["surface"]
    times = data["times"]
    depth = (geometry & np.uint32(0xffff)).astype(np.uint16)
    if args.depth_sequence:
        depth = np.load(args.depth_sequence)["depth"]
    safety = (geometry >> np.uint32(16)).astype(np.float32) / 65535.0
    filtered = np.empty_like(safety)
    filtered[0] = safety[0]
    for index in range(1, len(safety)):
        carried = _transport(filtered[index - 1], transport[index])
        # Instant conservative attack; bounded release. min(current, ...) is
        # the safety proof: filtering can only flatten more than the current
        # ownership analysis, never restore disparity that it rejected.
        filtered[index] = np.minimum(
            safety[index], carried + max(0.0, args.rise_per_map))

    meta = json.loads(args.capture.with_suffix(".json").read_text("utf-8"))
    count, grid_h, grid_w = safety.shape
    full_w, full_h = meta["source_size"]
    roi_x, roi_y = meta["roi"][:2]
    sx0, sy0, sx1, sy1 = args.sub_roi
    gx0 = max(0, min(grid_w - 1, round((roi_x + sx0) * grid_w / full_w)))
    gy0 = max(0, min(grid_h - 1, round((roi_y + sy0) * grid_h / full_h)))
    gx1 = max(gx0 + 1, min(grid_w, round((roi_x + sx1) * grid_w / full_w)))
    gy1 = max(gy0 + 1, min(grid_h, round((roi_y + sy1) * grid_h / full_h)))
    raw_p95 = []
    filtered_p95 = []
    retained = []
    for index in range(1, count):
        if abs(float(times[index]) - args.focus_time) > args.window:
            continue
        luma = surface[index, ..., 1].astype(np.float32) / 65535.0
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        mask[gy0:gy1, gx0:gx1] = True
        mask &= luma < (112.0 / 255.0)
        if mask.sum() < 8:
            continue
        raw_delta = np.abs(
            safety[index] - _transport(safety[index - 1], transport[index]))
        filtered_delta = np.abs(
            filtered[index] - _transport(filtered[index - 1], transport[index]))
        raw_p95.append(float(np.quantile(raw_delta[mask], .95)))
        filtered_p95.append(float(np.quantile(filtered_delta[mask], .95)))
        retained.append(float(np.mean(filtered[index][mask] /
                                      np.maximum(1e-5, safety[index][mask]))))

    safety_u16 = np.round(filtered * 65535.0).astype(np.uint16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, depth=depth, safety=safety_u16, times=times)
    print(json.dumps({
        "grid_sub_roi": [gx0, gy0, gx1, gy1],
        "raw_safety_change_p95": float(np.median(raw_p95)),
        "filtered_safety_change_p95": float(np.median(filtered_p95)),
        "mean_safety_retained_ratio": float(np.mean(retained)),
        "rise_per_map": args.rise_per_map,
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
