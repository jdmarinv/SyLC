"""Measure the exact shader-commanded stereo motion from packed provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _backward_warp(previous: np.ndarray, current_source: np.ndarray,
                   previous_source: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        current_source, previous_source, None,
        0.5, 5, 31, 4, 7, 1.5, 0)
    height, width = current_source.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    return cv2.remap(previous.astype(np.float32),
                     xx + flow[..., 0], yy + flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _decode(packed: np.ndarray, full_width: int,
            include_fill: bool) -> tuple[np.ndarray, np.ndarray]:
    base = (packed & np.uint32(0xffff)).astype(np.float32)
    base *= float(full_width) / 65535.0
    fill = ((packed >> np.uint32(24)) & np.uint32(0x7f)).astype(np.float32)
    fill /= 127.0
    if include_fill:
        delta = ((packed >> np.uint32(16)) & np.uint32(0xff)).astype(np.int16)
        delta -= 128
        base += delta.astype(np.float32) * fill
    return base, fill


def _forward_position(inverse_source_x: np.ndarray, x0: int) -> np.ndarray:
    """Invert destination->source x into source->destination x, per row."""
    height, width = inverse_source_x.shape
    destination = x0 + np.arange(width, dtype=np.float32) + 0.5
    source = x0 + np.arange(width, dtype=np.float32) + 0.5
    forward = np.empty_like(inverse_source_x, dtype=np.float32)
    for y in range(height):
        owner = inverse_source_x[y]
        order = np.argsort(owner, kind="stable")
        forward[y] = np.interp(
            source, owner[order], destination[order],
            left=destination[0], right=destination[-1])
    return forward


def _separation(left: np.ndarray, right: np.ndarray,
                full_width: int, x0: int,
                include_fill: bool) -> tuple[np.ndarray, np.ndarray]:
    left_owner, left_fill = _decode(left, full_width, include_fill)
    right_owner, right_fill = _decode(right, full_width, include_fill)
    left_position = _forward_position(left_owner, x0)
    right_position = _forward_position(right_owner, x0)
    return left_position - right_position, np.maximum(left_fill, right_fill)


def _stats(capture: np.lib.npyio.NpzFile, full_width: int, x0: int,
           focus: float, sub_roi: tuple[int, int, int, int],
           window: float, include_fill: bool) -> dict[str, float]:
    source = capture["source"].astype(np.uint8, copy=False)
    left = capture["provenance_left"]
    right = capture["provenance_right"]
    times = capture["times"]
    x_lo, y_lo, x_hi, y_hi = sub_roi
    separations = []
    fills = []
    temporal_p50 = []
    temporal_p95 = []
    frame_separation_p50 = []
    frame_separation_p10 = []
    frame_separation_p90 = []
    fill_mean = []
    fill_active = []
    previous_separation = None
    previous_source = None
    for index in range(len(times)):
        separation, fill = _separation(
            left[index], right[index], full_width, x0, include_fill)
        if abs(float(times[index]) - focus) <= window:
            mask = np.zeros_like(source[index], dtype=bool)
            mask[max(0, y_lo):min(mask.shape[0], y_hi),
                 max(0, x_lo):min(mask.shape[1], x_hi)] = True
            mask &= source[index] < 112
            if mask.sum() >= 16:
                separations.extend(separation[mask].tolist())
                fills.extend(fill[mask].tolist())
                frame_separation_p10.append(
                    float(np.quantile(separation[mask], 0.10)))
                frame_separation_p50.append(float(np.median(separation[mask])))
                frame_separation_p90.append(
                    float(np.quantile(separation[mask], 0.90)))
                fill_mean.append(float(np.mean(fill[mask])))
                fill_active.append(float(np.mean(fill[mask] > 0.08)))
                if previous_separation is not None:
                    transported = _backward_warp(
                        previous_separation, source[index], previous_source)
                    delta = np.abs(separation - transported)
                    temporal_p50.append(float(np.median(delta[mask])))
                    temporal_p95.append(float(np.quantile(delta[mask], 0.95)))
        previous_separation = separation
        previous_source = source[index]
    values = np.asarray(separations)
    return {
        "separation_p10_px": float(np.quantile(values, 0.10)),
        "separation_p50_px": float(np.median(values)),
        "separation_p90_px": float(np.quantile(values, 0.90)),
        "frame_median_range_px": float(
            np.max(frame_separation_p50) - np.min(frame_separation_p50)),
        "frame_median_std_px": float(np.std(frame_separation_p50)),
        "frame_p10_range_px": float(
            np.max(frame_separation_p10) - np.min(frame_separation_p10)),
        "frame_p90_range_px": float(
            np.max(frame_separation_p90) - np.min(frame_separation_p90)),
        "temporal_change_p50_px": float(np.median(temporal_p50)),
        "temporal_change_p95_px": float(np.median(temporal_p95)),
        "fill_mean": float(np.mean(fill_mean)),
        "fill_active_pct": 100.0 * float(np.mean(fill_active)),
    }


def _write_heatmap(capture: np.lib.npyio.NpzFile, full_width: int, x0: int,
                   focus: float, window: float, output: Path) -> None:
    source = capture["source"].astype(np.uint8, copy=False)
    left = capture["provenance_left"]
    right = capture["provenance_right"]
    times = capture["times"]
    deltas = []
    previous_separation = None
    previous_source = None
    selected_sources = []
    for index in range(len(times)):
        separation, _ = _separation(
            left[index], right[index], full_width, x0, True)
        if (previous_separation is not None and
                abs(float(times[index]) - focus) <= window):
            transported = _backward_warp(
                previous_separation, source[index], previous_source)
            deltas.append(np.abs(separation - transported))
            selected_sources.append(source[index])
        previous_separation = separation
        previous_source = source[index]
    temporal_p95 = np.quantile(np.stack(deltas), 0.95, axis=0)
    scale = np.clip(temporal_p95 / 4.0, 0.0, 1.0)
    heat = cv2.applyColorMap(
        np.round(scale * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    base = cv2.cvtColor(
        selected_sources[len(selected_sources) // 2], cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.52, heat, 0.48, 0.0)
    cv2.putText(overlay, "p95 variation separation stereo (0..4 px)",
                (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (255, 255, 255), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), overlay):
        raise RuntimeError(f"cannot write heatmap: {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--focus-time", type=float, required=True)
    parser.add_argument("--sub-roi", nargs=4, type=int, required=True,
                        metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--window", type=float, default=0.75)
    parser.add_argument("--heatmap", type=Path)
    args = parser.parse_args()
    capture = np.load(args.capture)
    if "provenance_left" not in capture:
        parser.error("capture has no provenance maps")
    meta = json.loads(args.capture.with_suffix(".json").read_text("utf-8"))
    full_width = int(meta["source_size"][0])
    x0 = int(meta["roi"][0])
    sub_roi = tuple(args.sub_roi)
    print("base inverse warp (depth displacement only)")
    print(json.dumps(_stats(
        capture, full_width, x0, args.focus_time, sub_roi,
        args.window, False), indent=2))
    print("final inverse warp (including temporal/disocclusion fill)")
    print(json.dumps(_stats(
        capture, full_width, x0, args.focus_time, sub_roi,
        args.window, True), indent=2))
    if args.heatmap:
        _write_heatmap(capture, full_width, x0, args.focus_time,
                       args.window, args.heatmap)
        print(f"heatmap={args.heatmap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
