"""GPU proof that the calibrated envelope changes and bounds displayed disparity.

This probe uses a constant near-depth field, reads the production provenance
sidecar and compares the same 3% artistic warp with the envelope disabled and
enabled.  It requires Windows, D3D11 and a freshly built native module.
"""

from __future__ import annotations

import ctypes
import json

import numpy as np

from probe_stereo_lab_rivalry_gpu import _hidden_hwnd, native


WIDTH, HEIGHT = 640, 360
SOFT_PCT, HARD_PCT = 0.30, 0.50


def _source_planes():
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    luma = np.clip(
        32.0 + 0.21 * x + 19.0 * np.sin(0.027 * x + 0.019 * y),
        16.0, 235.0).astype(np.uint8)
    chroma_shape = (HEIGHT // 2, WIDTH // 2)
    return (np.ascontiguousarray(luma),
            np.full(chroma_shape, 128, np.uint8),
            np.full(chroma_shape, 128, np.uint8))


def _render(hwnd: int, comfort_enabled: bool):
    renderer = native.NativeRenderer()
    try:
        if not renderer.initialize(hwnd, WIDTH, HEIGHT, False):
            raise RuntimeError(renderer.last_error())
        if not renderer.set_synth3d(
                True, 3.0, 0.0, False,
                side=WIDTH, grid_width=WIDTH, grid_height=HEIGHT,
                stereo_lab=True,
                comfort_enabled=comfort_enabled,
                comfort_soft_pct=SOFT_PCT,
                comfort_hard_pct=HARD_PCT):
            raise RuntimeError(renderer.last_error())
        renderer.synth3d_set_test_depth(
            np.full((HEIGHT, WIDTH), 65535, dtype=np.uint16))
        if not renderer.set_yuv_frame(*_source_planes()):
            raise RuntimeError(renderer.last_error())
        if not renderer.present(0):
            raise RuntimeError(renderer.last_error())
        provenance = np.asarray(renderer.synth3d_read_plane(6)).copy()
        return provenance, renderer.synth3d_status()
    finally:
        renderer.shutdown()


def _median_disparity_pct(provenance: np.ndarray) -> float:
    base_x = (provenance & np.uint32(0xffff)).astype(np.float64) / 65535.0
    destination_x = ((np.arange(WIDTH, dtype=np.float64) + 0.5) /
                     float(WIDTH))[None, :]
    fill = (provenance >> np.uint32(24)) & np.uint32(0x7f)
    margin = max(32, WIDTH // 10)
    valid = np.zeros_like(provenance, dtype=bool)
    valid[:, margin:-margin] = True
    valid &= fill == 0
    full_disparity_pct = 200.0 * np.abs(destination_x - base_x)
    return float(np.median(full_disparity_pct[valid]))


def main() -> int:
    hwnd = _hidden_hwnd()
    try:
        raw, raw_status = _render(hwnd, False)
        calibrated, calibrated_status = _render(hwnd, True)
    finally:
        ctypes.windll.user32.DestroyWindow(hwnd)

    raw_pct = _median_disparity_pct(raw)
    calibrated_pct = _median_disparity_pct(calibrated)
    report = {
        "raw_disparity_pct": raw_pct,
        "calibrated_disparity_pct": calibrated_pct,
        "soft_pct": SOFT_PCT,
        "hard_pct": HARD_PCT,
        "raw_status": raw_status,
        "calibrated_status": calibrated_status,
    }
    print(json.dumps(report, indent=2))

    if raw_pct <= 2.0:
        raise SystemExit("raw 3% test warp did not produce the expected disparity")
    if not (SOFT_PCT < calibrated_pct < HARD_PCT + 0.03):
        raise SystemExit("calibrated disparity escaped its physical envelope")
    if calibrated_pct >= 0.40 * raw_pct:
        raise SystemExit("calibrated warp was not materially compressed")
    if "comfort=calibrated" not in calibrated_status:
        raise SystemExit("native status did not acknowledge calibrated comfort")
    if "comfort=off" not in raw_status:
        raise SystemExit("comfort-off rollback status is incorrect")
    print("[COMFORT-GPU] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
