r"""Replay a real movie excerpt through the complete native 2D->3D path.

This is a diagnostic companion to profile_synth3d_e2e.py.  It decodes the
actual source with OpenCV, paces it at media cadence, and captures a normalized
ROI from the source and both synthesized luma planes on selected frames.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build_py314" / "python" / "Release"))
_DLL_HANDLES = [
    os.add_dll_directory(str(path))
    for path in (ROOT / "runtime", ROOT / "ort_tensorrt")
    if path.exists()
]

import mvc_demuxer_cpp as native
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget


def _parse_status(line: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in line.split() if "=" in token)


def _i420_planes(bgr: np.ndarray):
    height, width = bgr.shape[:2]
    packed = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    y = np.ascontiguousarray(packed[:height, :])
    u = np.ascontiguousarray(
        packed[height:height + height // 4, :].reshape(height // 2, width // 2))
    v = np.ascontiguousarray(
        packed[height + height // 4:, :].reshape(height // 2, width // 2))
    return y, u, v


def _open_at(path: Path, seconds: float):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    return capture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--warmup", type=float, default=6.0)
    parser.add_argument("--capture-every", type=int, default=3)
    parser.add_argument("--roi", nargs=4, type=float,
                        default=(0.60, 0.03, 1.0, 0.58),
                        metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--strength", type=float, default=1.4)
    parser.add_argument("--convergence", type=float, default=0.52)
    parser.add_argument("--depth-view", action="store_true")
    parser.add_argument("--no-temporal-fill", action="store_true")
    parser.add_argument("--no-stereo-lab", action="store_true")
    parser.add_argument("--geometry-capture", action="store_true")
    parser.add_argument("--test-depth-sequence", type=Path)
    parser.add_argument("--test-geometry-sequence", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    test_depth_sequence = None
    test_geometry_sequence = None
    test_geometry_zeros = None
    if args.test_depth_sequence and args.test_geometry_sequence:
        parser.error("choose test-depth-sequence or test-geometry-sequence")
    if args.test_depth_sequence:
        test_depth_sequence = np.load(args.test_depth_sequence)["depth"]
    if args.test_geometry_sequence:
        sequence_data = np.load(args.test_geometry_sequence)
        test_geometry_sequence = (
            sequence_data["depth"], sequence_data["safety"])

    model = ROOT / "models" / "da3_base_756x322.onnx"
    ort_dir = ROOT / "ort_tensorrt"
    if not model.exists():
        parser.error(f"model is missing: {model}")

    capture = _open_at(args.video, args.start)
    ok, first_bgr = capture.read()
    if not ok:
        raise RuntimeError("cannot decode first frame")
    height, width = first_bgr.shape[:2]
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24000.0 / 1001.0)
    period = 1.0 / fps

    app = QApplication.instance() or QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("SyLC 2D->3D real-clip diagnostic")
    window.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
    window.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
    window.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, True)
    window.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
    window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
    window.resize(640, max(160, round(640 * height / width)))
    window.move(20, 20)
    window.show()
    app.processEvents()

    renderer = native.NativeRenderer()
    if not renderer.initialize(int(window.winId()), window.width(), window.height()):
        raise RuntimeError(renderer.last_error())
    renderer.set_source_aspect(width / float(height))
    if not renderer.set_synth3d(
            True, args.strength, args.convergence, args.depth_view,
            str(model), str(ort_dir), False, 756, 756, 322,
            0.0, 0.0, False, not args.no_temporal_fill,
            not args.no_stereo_lab):
        raise RuntimeError(renderer.last_error())
    if test_depth_sequence is not None:
        grid_width, grid_height = renderer.synth3d_grid()
        expected = (grid_height, grid_width)
        if test_depth_sequence.ndim != 3 or tuple(test_depth_sequence.shape[1:]) != expected:
            raise RuntimeError(
                f"test depth sequence shape {test_depth_sequence.shape} != (*,{expected})")
        renderer.synth3d_set_test_depth(test_depth_sequence[0])
    if test_geometry_sequence is not None:
        depth_sequence, safety_sequence = test_geometry_sequence
        grid_width, grid_height = renderer.synth3d_grid()
        expected = (grid_height, grid_width)
        if (depth_sequence.ndim != 3 or safety_sequence.shape != depth_sequence.shape or
                tuple(depth_sequence.shape[1:]) != expected):
            raise RuntimeError(
                f"test geometry shapes {depth_sequence.shape}/{safety_sequence.shape} "
                f"!= (*,{expected})")
        test_geometry_zeros = np.zeros(expected, dtype=np.uint16)
        renderer.synth3d_set_test_geometry(
            depth_sequence[0], depth_sequence[0], safety_sequence[0],
            test_geometry_zeros)

    def submit(bgr: np.ndarray, media_ms: float):
        y, u, v = _i420_planes(bgr)
        renderer.set_video_time_ms(media_ms)
        if not renderer.set_yuv_frame(y, u, v):
            raise RuntimeError(renderer.last_error())
        if not renderer.present(0):
            raise RuntimeError(renderer.last_error())
        app.processEvents()
        return y

    source_roi = []
    left_roi = []
    right_roi = []
    provenance_left_roi = []
    provenance_right_roi = []
    geometry_maps = []
    surface_maps = []
    transport_maps = []
    times = []
    statuses = []
    final_status = ""
    try:
        # Build/warm the asynchronous ORT service without consuming the clip.
        # Media time approaches the real start monotonically, matching playback.
        warm_frames = max(1, round(args.warmup * fps))
        warm_start = time.perf_counter()
        for index in range(warm_frames):
            submit(first_bgr,
                   (args.start - args.warmup + index * period) * 1000.0)
            target = warm_start + (index + 1) * period
            delay = target - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)

        capture.release()
        capture = _open_at(args.video, args.start)
        frame_count = max(1, round(args.seconds * fps))
        run_start = time.perf_counter()
        x0 = max(0, min(width - 1, round(args.roi[0] * width)))
        y0 = max(0, min(height - 1, round(args.roi[1] * height)))
        x1 = max(x0 + 1, min(width, round(args.roi[2] * width)))
        y1 = max(y0 + 1, min(height, round(args.roi[3] * height)))
        for index in range(frame_count):
            ok, bgr = capture.read()
            if not ok:
                break
            media_s = args.start + index * period
            if test_depth_sequence is not None:
                renderer.synth3d_set_test_depth(
                    test_depth_sequence[min(index, len(test_depth_sequence) - 1)])
            if test_geometry_sequence is not None:
                depth_sequence, safety_sequence = test_geometry_sequence
                sequence_index = min(index, len(depth_sequence) - 1)
                renderer.synth3d_set_test_geometry(
                    depth_sequence[sequence_index], depth_sequence[sequence_index],
                    safety_sequence[sequence_index], test_geometry_zeros)
            y = submit(bgr, media_s * 1000.0)
            if index % max(1, args.capture_every) == 0:
                source_roi.append(y[y0:y1, x0:x1].copy())
                if args.geometry_capture:
                    geometry_maps.append(np.asarray(
                        renderer.synth3d_read_plane(8)).copy())
                    surface_maps.append(np.asarray(
                        renderer.synth3d_read_plane(9)).copy())
                    transport_maps.append(np.asarray(
                        renderer.synth3d_read_plane(10)).copy())
                else:
                    left = np.asarray(renderer.synth3d_read_plane(0)).copy()
                    right = np.asarray(renderer.synth3d_read_plane(3)).copy()
                    left_roi.append(left[y0:y1, x0:x1].copy())
                    right_roi.append(right[y0:y1, x0:x1].copy())
                    if not args.depth_view and not args.no_stereo_lab:
                        provenance_left = np.asarray(
                            renderer.synth3d_read_plane(6)).copy()
                        provenance_right = np.asarray(
                            renderer.synth3d_read_plane(7)).copy()
                        provenance_left_roi.append(
                            provenance_left[y0:y1, x0:x1].copy())
                        provenance_right_roi.append(
                            provenance_right[y0:y1, x0:x1].copy())
                times.append(media_s)
                status = renderer.synth3d_status()
                statuses.append(status)
                fields = _parse_status(status)
                print(f"{media_s:8.3f}s stable={fields.get('stable', '?')} "
                      f"history={fields.get('history', '?')} "
                      f"motion={fields.get('motion', '?')} "
                      f"age={fields.get('age_ms', '?')}")
            target = run_start + (index + 1) * period
            delay = target - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
        final_status = renderer.synth3d_status()
    finally:
        capture.release()
        renderer.set_synth3d(False)
        renderer.shutdown()
        window.close()
        app.processEvents()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        source=np.asarray(source_roi),
        left=np.asarray(left_roi),
        right=np.asarray(right_roi),
        times=np.asarray(times, dtype=np.float64),
        status=np.asarray(statuses),
    )
    if provenance_left_roi:
        payload["provenance_left"] = np.asarray(provenance_left_roi)
        payload["provenance_right"] = np.asarray(provenance_right_roi)
    if geometry_maps:
        payload["geometry"] = np.asarray(geometry_maps)
        payload["surface"] = np.asarray(surface_maps)
        payload["transport"] = np.asarray(transport_maps)
    np.savez_compressed(args.output, **payload)
    meta = {
        "video": str(args.video),
        "start": args.start,
        "seconds": args.seconds,
        "fps": fps,
        "source_size": [width, height],
        "roi": [x0, y0, x1, y1],
        "depth_view": args.depth_view,
        "test_depth_sequence": (str(args.test_depth_sequence)
                                if args.test_depth_sequence else None),
        "test_geometry_sequence": (str(args.test_geometry_sequence)
                                   if args.test_geometry_sequence else None),
        "captures": len(times),
        "final_status": final_status,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
