"""Projection-room calibration and comfort activation regressions."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from sylc.stereo_calibration_dialog import (
    CALIBRATION_SCHEMA_VERSION,
    CALIBRATION_VERSION_KEY,
    StereoCalibrationWizard,
    calibration_is_complete,
    load_calibration_profile,
    save_calibration,
)


def test_default_profile_is_enabled_and_physically_calibrated():
    profile = load_calibration_profile({}, environ={})

    assert profile.enabled is True
    assert profile.screen_height_m == 1.7
    assert profile.envelope.geometry.screen_width_m == 2.5
    assert profile.envelope.geometry.viewing_distance_m == 3.5
    assert profile.envelope.soft_pixels == pytest.approx(30.96576)
    assert profile.envelope.hard_pixels == pytest.approx(51.6096)


def test_environment_overrides_persisted_room_profile():
    profile = load_calibration_profile(
        {
            "synth3d_comfort_enabled": False,
            "synth3d_comfort_screen_width_m": 2.0,
        },
        environ={
            "SYLC_COMFORT_ENABLED": "1",
            "SYLC_COMFORT_SCREEN_WIDTH_M": "4.0",
        },
    )

    assert profile.enabled is True
    assert profile.envelope.geometry.screen_width_m == 4.0


def test_save_marks_current_install_complete_and_validates_values():
    settings = {}
    values = {
        "synth3d_comfort_enabled": True,
        "synth3d_comfort_screen_width_m": 3.2,
        "synth3d_comfort_screen_height_m": 1.8,
        "synth3d_comfort_horizontal_pixels": 3840,
        "synth3d_comfort_viewing_distance_m": 4.1,
        "synth3d_comfort_ipd_m": 0.063,
        "synth3d_comfort_soft_vac_d": 0.18,
        "synth3d_comfort_hard_vac_d": 0.30,
    }

    profile = save_calibration(settings, values)

    assert profile.enabled is True
    assert settings[CALIBRATION_VERSION_KEY] == CALIBRATION_SCHEMA_VERSION
    assert calibration_is_complete(settings)
    assert profile.envelope.geometry.horizontal_pixels == 3840
    assert profile.envelope.geometry.viewing_distance_m == 4.1


def test_invalid_geometry_is_never_marked_complete():
    settings = {}
    with pytest.raises(ValueError):
        save_calibration(settings, {
            "synth3d_comfort_screen_width_m": 0.0,
        })

    assert not calibration_is_complete(settings)


def test_wizard_round_trips_room_measurements_and_updates_summary():
    app = QApplication.instance() or QApplication([])
    wizard = StereoCalibrationWizard({})

    wizard.screen_width.setValue(4.25)
    wizard.viewing_distance.setValue(5.5)
    wizard.horizontal_pixels.setValue(3840)
    wizard.ipd_mm.setValue(63.5)
    values = wizard.calibration_values()

    assert values["synth3d_comfort_screen_width_m"] == 4.25
    assert values["synth3d_comfort_viewing_distance_m"] == 5.5
    assert values["synth3d_comfort_horizontal_pixels"] == 3840
    assert values["synth3d_comfort_ipd_m"] == pytest.approx(0.0635)
    assert "Soft knee" in wizard.summary.text()

    wizard.close()
    wizard.deleteLater()
    app.processEvents()


def test_native_path_passes_and_applies_comfort_once():
    root = Path(__file__).resolve().parents[1]
    native = (root / "mvc_realtime_demuxer" / "src" /
              "native_renderer.cpp").read_text(encoding="utf-8")
    shader = (root / "mvc_realtime_demuxer" / "shaders" /
              "synth3d.hlsl").read_text(encoding="utf-8")

    assert "p.comfort_enabled = comfort_enabled;" in native
    assert "p.comfort_enabled = false;" not in native
    assert "float project_comfort_disparity(float disparity)" in shader
    assert "return project_comfort_disparity(artistic);" in shader

