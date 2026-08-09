"""Offline causal multiscale temporal filter for captured synth3d depth."""

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


def _smoothstep(lo: float, hi: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - lo) / max(1e-8, hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bands(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low1 = cv2.GaussianBlur(depth, (0, 0), 1.15, borderType=cv2.BORDER_REPLICATE)
    low2 = cv2.GaussianBlur(low1, (0, 0), 3.2, borderType=cv2.BORDER_REPLICATE)
    return depth - low1, low1 - low2, low2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus-time", type=float, default=134.128)
    parser.add_argument("--sub-roi", nargs=4, type=int,
                        default=(140, 180, 300, 340))
    parser.add_argument("--window", type=float, default=0.75)
    args = parser.parse_args()

    data = np.load(args.capture)
    meta = json.loads(args.capture.with_suffix(".json").read_text("utf-8"))
    surface = data["surface"]
    transport = data["transport"]
    times = data["times"]
    raw = surface[..., 0].astype(np.float32) / 65535.0
    reliability = transport[..., 2].astype(np.float32) / 65535.0
    count, grid_h, grid_w = raw.shape

    filtered = np.empty_like(raw)
    high_state, mid_state, low_state = _bands(raw[0])
    filtered[0] = np.clip(high_state + mid_state + low_state, 0.0, 1.0)
    temporal_before = []
    temporal_after = []

    for index in range(1, count):
        previous_high = _transport(high_state, transport[index])
        previous_mid = _transport(mid_state, transport[index])
        previous_low = _transport(low_state, transport[index])
        previous_reconstructed = previous_high + previous_mid + previous_low
        # tone_lo/tone_hi changes are an arbitrary global gauge transform, not
        # new scene geometry. Remove their scale/offset on reliable interiors
        # before asking the multiscale bands to decide what changed locally.
        confidence = surface[index, ..., 2].astype(np.float32) / 65535.0
        boundary = surface[index, ..., 3].astype(np.float32) / 65535.0
        fit_mask = (reliability[index] >= 0.65) & (confidence >= 0.30) & \
                   (boundary < 0.25)
        x = raw[index][fit_mask].astype(np.float64)
        y = previous_reconstructed[fit_mask].astype(np.float64)
        if x.size >= 256:
            xm, ym = float(x.mean()), float(y.mean())
            variance = float(np.mean((x - xm) ** 2))
            covariance = float(np.mean((x - xm) * (y - ym)))
            scale = covariance / max(1e-8, variance)
            scale = float(np.clip(scale, 0.82, 1.22))
            shift = float(np.clip(ym - scale * xm, -0.10, 0.10))
        else:
            scale, shift = 1.0, 0.0
        aligned = np.clip(scale * raw[index] + shift, 0.0, 1.0)
        current_high, current_mid, current_low = _bands(aligned)
        # Ordinary confidence around 0.84 should not weaken the filter. Only a
        # genuinely untracked/disoccluded cell approaches the current map.
        uncertain = _smoothstep(0.28, 0.62, 1.0 - reliability[index])

        def update(current: np.ndarray, previous: np.ndarray,
                   base_alpha: float, change_lo: float,
                   change_hi: float) -> np.ndarray:
            change = _smoothstep(change_lo, change_hi,
                                 np.abs(current - previous))
            alpha = base_alpha + (1.0 - base_alpha) * np.maximum(
                uncertain, change)
            return previous + alpha * (current - previous)

        low_state = update(current_low, previous_low, 0.045, 0.014, 0.050)
        mid_state = update(current_mid, previous_mid, 0.16, 0.010, 0.038)
        high_state = update(current_high, previous_high, 0.58, 0.008, 0.030)
        filtered[index] = np.clip(
            low_state + mid_state + high_state, 0.0, 1.0)
        temporal_before.append(np.abs(
            raw[index] - _transport(raw[index - 1], transport[index])))
        temporal_after.append(np.abs(
            filtered[index] - _transport(filtered[index - 1], transport[index])))

    full_w, full_h = meta["source_size"]
    roi_x, roi_y = meta["roi"][:2]
    sx0, sy0, sx1, sy1 = args.sub_roi
    gx0 = max(0, min(grid_w - 1, round((roi_x + sx0) * grid_w / full_w)))
    gy0 = max(0, min(grid_h - 1, round((roi_y + sy0) * grid_h / full_h)))
    gx1 = max(gx0 + 1, min(grid_w, round((roi_x + sx1) * grid_w / full_w)))
    gy1 = max(gy0 + 1, min(grid_h, round((roi_y + sy1) * grid_h / full_h)))
    before_p95 = []
    after_p95 = []
    raw_median = []
    filtered_median = []
    gradient_ratio = []
    for index in range(1, count):
        if abs(float(times[index]) - args.focus_time) > args.window:
            continue
        luma = surface[index, ..., 1].astype(np.float32) / 65535.0
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        mask[gy0:gy1, gx0:gx1] = True
        mask &= luma < (112.0 / 255.0)
        if mask.sum() < 8:
            continue
        before_p95.append(float(np.quantile(temporal_before[index - 1][mask], .95)))
        after_p95.append(float(np.quantile(temporal_after[index - 1][mask], .95)))
        raw_median.append(float(np.median(raw[index][mask])))
        filtered_median.append(float(np.median(filtered[index][mask])))
        gx_raw = cv2.Sobel(raw[index], cv2.CV_32F, 1, 0, ksize=3)
        gy_raw = cv2.Sobel(raw[index], cv2.CV_32F, 0, 1, ksize=3)
        gx_filtered = cv2.Sobel(filtered[index], cv2.CV_32F, 1, 0, ksize=3)
        gy_filtered = cv2.Sobel(filtered[index], cv2.CV_32F, 0, 1, ksize=3)
        raw_edge = np.hypot(gx_raw, gy_raw)[mask]
        filtered_edge = np.hypot(gx_filtered, gy_filtered)[mask]
        gradient_ratio.append(float(
            np.quantile(filtered_edge, .90) /
            max(1e-8, np.quantile(raw_edge, .90))))

    depth_u16 = np.round(filtered * 65535.0).astype(np.uint16)
    safety_u16 = (data["geometry"] >> np.uint32(16)).astype(np.uint16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, depth=depth_u16, safety=safety_u16, times=times)
    print(json.dumps({
        "grid_sub_roi": [gx0, gy0, gx1, gy1],
        "raw_change_p95": float(np.median(before_p95)),
        "filtered_change_p95": float(np.median(after_p95)),
        "raw_median_range": float(np.ptp(raw_median)),
        "filtered_median_range": float(np.ptp(filtered_median)),
        "edge_p90_retained_ratio": float(np.median(gradient_ratio)),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
