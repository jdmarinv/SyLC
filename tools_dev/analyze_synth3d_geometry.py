"""Compare raw and final synth3d geometry with the pipeline's own flow."""

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
    parser.add_argument("--focus-time", type=float, required=True)
    parser.add_argument("--sub-roi", nargs=4, type=int, required=True,
                        metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--window", type=float, default=0.75)
    parser.add_argument("--heatmap", type=Path)
    args = parser.parse_args()

    data = np.load(args.capture)
    meta = json.loads(args.capture.with_suffix(".json").read_text("utf-8"))
    geometry = data["geometry"]
    surface = data["surface"]
    transport = data["transport"]
    times = data["times"]
    grid_h, grid_w = geometry.shape[1:]
    full_w, full_h = meta["source_size"]
    roi_x, roi_y = meta["roi"][:2]
    sx0, sy0, sx1, sy1 = args.sub_roi
    gx0 = max(0, min(grid_w - 1, round((roi_x + sx0) * grid_w / full_w)))
    gy0 = max(0, min(grid_h - 1, round((roi_y + sy0) * grid_h / full_h)))
    gx1 = max(gx0 + 1, min(grid_w, round((roi_x + sx1) * grid_w / full_w)))
    gy1 = max(gy0 + 1, min(grid_h, round((roi_y + sy1) * grid_h / full_h)))

    raw_residual_p50 = []
    raw_residual_p95 = []
    final_residual_p50 = []
    final_residual_p95 = []
    safety_residual_p95 = []
    raw_frame_median = []
    final_frame_median = []
    reliability_mean = []
    raw_delta_maps = []
    final_delta_maps = []
    heat_luma = []
    unique_maps = 0
    previous_raw = None
    previous_final = None
    previous_signature = None

    for index, time_value in enumerate(times):
        raw_u16 = surface[index, ..., 0]
        final_u16 = (geometry[index] & np.uint32(0xffff)).astype(np.uint16)
        signature = (int(raw_u16[::17, ::19].sum(dtype=np.uint64)),
                     int(final_u16[::17, ::19].sum(dtype=np.uint64)))
        if previous_signature == signature and previous_raw is not None and \
                np.array_equal(raw_u16, previous_raw) and \
                np.array_equal(final_u16, previous_final):
            continue
        unique_maps += 1
        raw = raw_u16.astype(np.float32) / 65535.0
        final = final_u16.astype(np.float32) / 65535.0
        luma = surface[index, ..., 1].astype(np.float32) / 65535.0
        safety = (geometry[index] >> np.uint32(16)).astype(np.float32) / 65535.0
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        mask[gy0:gy1, gx0:gx1] = True
        mask &= luma < (112.0 / 255.0)
        in_window = abs(float(time_value) - args.focus_time) <= args.window
        if in_window and mask.sum() >= 8:
            raw_frame_median.append(float(np.median(raw[mask])))
            final_frame_median.append(float(np.median(final[mask])))
            reliability_mean.append(float(np.mean(
                transport[index, ..., 2][mask].astype(np.float32) / 65535.0)))
            if previous_raw is not None:
                raw_previous = previous_raw.astype(np.float32) / 65535.0
                final_previous = previous_final.astype(np.float32) / 65535.0
                raw_delta = np.abs(
                    raw - _transport(raw_previous, transport[index]))
                final_delta = np.abs(
                    final - _transport(final_previous, transport[index]))
                previous_safety = (previous_geometry >> np.uint32(16)).astype(
                    np.float32) / 65535.0
                safety_delta = np.abs(
                    safety - _transport(previous_safety, transport[index]))
                raw_residual_p50.append(float(np.median(raw_delta[mask])))
                raw_residual_p95.append(float(np.quantile(raw_delta[mask], 0.95)))
                final_residual_p50.append(float(np.median(final_delta[mask])))
                final_residual_p95.append(float(np.quantile(final_delta[mask], 0.95)))
                safety_residual_p95.append(
                    float(np.quantile(safety_delta[mask], 0.95)))
                raw_delta_maps.append(raw_delta)
                final_delta_maps.append(final_delta)
                heat_luma.append(luma)
        previous_raw = raw_u16.copy()
        previous_final = final_u16.copy()
        previous_geometry = geometry[index].copy()
        previous_signature = signature

    def summary(values: list[float]) -> float:
        return float(np.median(values)) if values else float("nan")

    result = {
        "captures": int(len(times)),
        "unique_maps": unique_maps,
        "grid_sub_roi": [gx0, gy0, gx1, gy1],
        "raw_depth_change_p50": summary(raw_residual_p50),
        "raw_depth_change_p95": summary(raw_residual_p95),
        "final_depth_change_p50": summary(final_residual_p50),
        "final_depth_change_p95": summary(final_residual_p95),
        "safety_change_p95": summary(safety_residual_p95),
        "raw_frame_median_range": float(
            np.ptp(raw_frame_median)) if raw_frame_median else float("nan"),
        "final_frame_median_range": float(
            np.ptp(final_frame_median)) if final_frame_median else float("nan"),
        "flow_reliability_mean": float(
            np.mean(reliability_mean)) if reliability_mean else float("nan"),
    }
    print(json.dumps(result, indent=2))

    if args.heatmap and raw_delta_maps:
        raw_p95 = np.quantile(np.stack(raw_delta_maps), 0.95, axis=0)
        final_p95 = np.quantile(np.stack(final_delta_maps), 0.95, axis=0)
        base = cv2.cvtColor(
            np.round(heat_luma[len(heat_luma) // 2] * 255.0).astype(np.uint8),
            cv2.COLOR_GRAY2BGR)
        tiles = []
        for label, delta in (("raw q16", raw_p95), ("final Geometry", final_p95)):
            heat = cv2.applyColorMap(
                np.round(np.clip(delta / 0.05, 0.0, 1.0) * 255.0).astype(np.uint8),
                cv2.COLORMAP_TURBO)
            tile = cv2.addWeighted(base, 0.48, heat, 0.52, 0.0)
            cv2.rectangle(tile, (gx0, gy0), (gx1, gy1), (255, 255, 255), 1)
            cv2.putText(tile, f"p95 {label} (0..0.05)", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        output = np.hstack(tiles)
        args.heatmap.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.heatmap), output):
            raise RuntimeError(f"cannot write heatmap: {args.heatmap}")
        print(f"heatmap={args.heatmap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
