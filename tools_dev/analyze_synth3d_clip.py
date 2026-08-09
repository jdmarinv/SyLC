"""Quantify temporal depth/disparity on a replay_synth3d_clip capture."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _u8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint16:
        return (array / 257.0).astype(np.uint8)
    return array.astype(np.uint8, copy=False)


def _ridge_mask(luma: np.ndarray) -> np.ndarray:
    y = luma.astype(np.float32) / 255.0
    c = y[2:-2, 2:-2]
    scores = []
    for a, b in ((y[2:-2, :-4], y[2:-2, 4:]),
                 (y[:-4, 2:-2], y[4:, 2:-2]),
                 (y[:-4, :-4], y[4:, 4:]),
                 (y[:-4, 4:], y[4:, :-4])):
        da, db = c - a, c - b
        scores.append(np.where(da * db > 0.0, np.minimum(abs(da), abs(db)), 0.0))
    ridge = np.maximum.reduce(scores)
    out = np.zeros_like(luma, dtype=bool)
    out[2:-2, 2:-2] = ridge >= 0.055
    return out


def _backward_warp(previous: np.ndarray, current_source: np.ndarray,
                   previous_source: np.ndarray) -> np.ndarray:
    # Flow from current coordinates into the previous source frame gives the
    # exact remap needed to compare a transported previous signal to current.
    flow = cv2.calcOpticalFlowFarneback(
        current_source, previous_source, None,
        0.5, 5, 31, 4, 7, 1.5, 0)
    height, width = current_source.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    return cv2.remap(previous.astype(np.float32),
                     xx + flow[..., 0], yy + flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _same_frame_stereo(source: np.ndarray, left: np.ndarray,
                       right: np.ndarray, mask: np.ndarray):
    kwargs = dict(pyr_scale=0.5, levels=5, winsize=31, iterations=5,
                  poly_n=7, poly_sigma=1.5, flags=0)
    fl = cv2.calcOpticalFlowFarneback(source, left, None, **kwargs)
    fr = cv2.calcOpticalFlowFarneback(source, right, None, **kwargs)
    stereo = fl[..., 0] - fr[..., 0]
    values = stereo[mask]
    if values.size < 16:
        return stereo, np.nan, np.nan
    return stereo, float(np.median(values)), float(np.quantile(values, 0.90))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("warp", type=Path)
    parser.add_argument("depth", type=Path)
    parser.add_argument("--montage", type=Path)
    parser.add_argument("--focus-time", type=float)
    parser.add_argument("--sub-roi", nargs=4, type=int,
                        metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--window", type=float, default=0.75,
                        help="seconds around focus-time for sub-ROI metrics")
    args = parser.parse_args()

    warp = np.load(args.warp)
    depth = np.load(args.depth)
    source = _u8(depth["source"])
    depth_luma = _u8(depth["left"])
    warp_source = _u8(warp["source"])
    left, right = _u8(warp["left"]), _u8(warp["right"])
    count = min(len(source), len(depth_luma), len(left), len(right))

    depth_wire_residual = []
    depth_wire_p95 = []
    depth_wire_changed = []
    source_wire_residual = []
    depth_fog_residual = []
    stereo_median = []
    stereo_p90 = []
    stereo_maps = []
    wire_counts = []
    for index in range(count):
        wire = _ridge_mask(source[index])
        # Suppress the four-pixel border where transport has no valid support.
        valid = np.zeros_like(wire)
        valid[6:-6, 6:-6] = True
        wire &= valid
        fog = valid & ~cv2.dilate(wire.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        wire_counts.append(int(wire.sum()))
        stereo, med, p90 = _same_frame_stereo(
            warp_source[index], left[index], right[index], wire)
        stereo_maps.append(stereo)
        stereo_median.append(med)
        stereo_p90.append(p90)
        if index == 0:
            continue
        transported_depth = _backward_warp(
            depth_luma[index - 1], source[index], source[index - 1])
        transported_source = _backward_warp(
            source[index - 1], source[index], source[index - 1])
        depth_delta = abs(depth_luma[index].astype(np.float32) - transported_depth)
        source_delta = abs(source[index].astype(np.float32) - transported_source)
        depth_wire_residual.append(float(np.median(depth_delta[wire])))
        depth_wire_p95.append(float(np.quantile(depth_delta[wire], 0.95)))
        depth_wire_changed.append(float(np.mean(depth_delta[wire] > 4.0)))
        source_wire_residual.append(float(np.median(source_delta[wire])))
        depth_fog_residual.append(float(np.median(depth_delta[fog])))

    print(f"shape={source.shape} wire_pixels_median={np.median(wire_counts):.0f}")
    print("transported temporal residual (8-bit luma levels, median per frame)")
    print(f"  source/wire median={np.median(source_wire_residual):.3f} "
          f"p95={np.quantile(source_wire_residual, .95):.3f}")
    print(f"  depth/wire  median={np.median(depth_wire_residual):.3f} "
          f"p95={np.quantile(depth_wire_residual, .95):.3f}")
    print(f"  depth/wire within-frame p95 median={np.median(depth_wire_p95):.3f} "
          f"max={np.max(depth_wire_p95):.3f} "
          f"changed>4 mean={100*np.mean(depth_wire_changed):.2f}%")
    print(f"  depth/fog   median={np.median(depth_fog_residual):.3f} "
          f"p95={np.quantile(depth_fog_residual, .95):.3f}")
    finite_med = np.asarray(stereo_median)[np.isfinite(stereo_median)]
    finite_p90 = np.asarray(stereo_p90)[np.isfinite(stereo_p90)]
    print("same-frame wire disparity proxy (left flow - right flow, pixels)")
    print(f"  median range={finite_med.min():.3f}..{finite_med.max():.3f} "
          f"std={finite_med.std():.3f}")
    print(f"  p90    range={finite_p90.min():.3f}..{finite_p90.max():.3f} "
          f"std={finite_p90.std():.3f}")
    stereo_wire_p50 = []
    stereo_wire_p95 = []
    stereo_wire_changed = []
    for index in range(1, count):
        wire = _ridge_mask(source[index])
        wire[:6] = False
        wire[-6:] = False
        wire[:, :6] = False
        wire[:, -6:] = False
        transported = _backward_warp(
            stereo_maps[index - 1], source[index], source[index - 1])
        delta = abs(stereo_maps[index] - transported)
        stereo_wire_p50.append(float(np.median(delta[wire])))
        stereo_wire_p95.append(float(np.quantile(delta[wire], 0.95)))
        stereo_wire_changed.append(float(np.mean(delta[wire] > 0.75)))
    print("transported stereo-disparity change on wires (pixels)")
    print(f"  p50 median={np.median(stereo_wire_p50):.3f} "
          f"p95 median={np.median(stereo_wire_p95):.3f} "
          f"p95 max={np.max(stereo_wire_p95):.3f} "
          f"changed>0.75 mean={100*np.mean(stereo_wire_changed):.2f}%")

    if args.sub_roi:
        x0, y0, x1, y1 = args.sub_roi
        focus = (float(args.focus_time) if args.focus_time is not None
                 else float(depth["times"][count // 2]))
        depth_center_p50 = []
        depth_center_p95 = []
        stereo_center_p50 = []
        stereo_center_p95 = []
        for index in range(1, count):
            if abs(float(depth["times"][index]) - focus) > args.window:
                continue
            mask = np.zeros_like(source[index], dtype=bool)
            mask[max(0, y0):min(mask.shape[0], y1),
                 max(0, x0):min(mask.shape[1], x1)] = True
            # The hub is the dark coherent machinery, not the translucent fog
            # passing across the same rectangle.
            mask &= source[index] < 112
            transported_depth = _backward_warp(
                depth_luma[index - 1], source[index], source[index - 1])
            dd = abs(depth_luma[index].astype(np.float32) - transported_depth)
            transported_stereo = _backward_warp(
                stereo_maps[index - 1], source[index], source[index - 1])
            sd = abs(stereo_maps[index] - transported_stereo)
            if mask.sum() < 16:
                continue
            depth_center_p50.append(float(np.median(dd[mask])))
            depth_center_p95.append(float(np.quantile(dd[mask], 0.95)))
            stereo_center_p50.append(float(np.median(sd[mask])))
            stereo_center_p95.append(float(np.quantile(sd[mask], 0.95)))
        print(f"dark hub sub-ROI {args.sub_roi} around {focus:.3f}s")
        print(f"  depth change p50={np.median(depth_center_p50):.3f} "
              f"p95={np.median(depth_center_p95):.3f} /255")
        print(f"  stereo change p50={np.median(stereo_center_p50):.3f} "
              f"p95={np.median(stereo_center_p95):.3f} px")

    if args.montage:
        focus = (float(args.focus_time) if args.focus_time is not None
                 else float(depth["times"][count // 2]))
        center = int(np.argmin(abs(depth["times"][:count] - focus)))
        first = max(0, min(count - 6, center - 2))
        ids = np.arange(first, first + min(6, count))
        args.montage.parent.mkdir(parents=True, exist_ok=True)
        for label, series in (("depth", depth_luma),
                              ("source", source),
                              ("left", left),
                              ("right", right)):
            tiles = []
            for index in ids:
                tile = cv2.cvtColor(series[index], cv2.COLOR_GRAY2BGR)
                cv2.putText(tile, f"{float(depth['times'][index]):.3f}s",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (255, 255, 255), 2, cv2.LINE_AA)
                tiles.append(tile)
            rows = [np.hstack(tiles[:3])]
            if len(tiles) > 3:
                rows.append(np.hstack(tiles[3:6]))
            target = args.montage.with_name(
                f"{args.montage.stem}_{label}{args.montage.suffix}")
            cv2.imwrite(str(target), np.vstack(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
