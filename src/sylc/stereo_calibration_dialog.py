"""First-run projection-room calibration for Stereo Lab.

The persisted values are deliberately independent from Qt so the loading and
validation contract can be exercised without constructing the player window.
Environment variables keep precedence for lab/profiling runs, while the wizard
writes the ordinary per-install profile used by the application.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, MutableMapping, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from .synth3d_stereo_comfort import (
    StereoComfortEnvelope,
    StereoDisplayGeometry,
)


CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_VERSION_KEY = "synth3d_comfort_calibration_version"

DEFAULT_CALIBRATION = {
    "synth3d_comfort_enabled": True,
    "synth3d_comfort_screen_width_m": 2.5,
    "synth3d_comfort_screen_height_m": 1.7,
    "synth3d_comfort_horizontal_pixels": 1920,
    "synth3d_comfort_viewing_distance_m": 3.5,
    "synth3d_comfort_ipd_m": 0.064,
    "synth3d_comfort_soft_vac_d": 0.18,
    "synth3d_comfort_hard_vac_d": 0.30,
}

_ENVIRONMENT_KEYS = {
    "synth3d_comfort_enabled": "SYLC_COMFORT_ENABLED",
    "synth3d_comfort_screen_width_m": "SYLC_COMFORT_SCREEN_WIDTH_M",
    "synth3d_comfort_screen_height_m": "SYLC_COMFORT_SCREEN_HEIGHT_M",
    "synth3d_comfort_horizontal_pixels": "SYLC_COMFORT_HORIZONTAL_PIXELS",
    "synth3d_comfort_viewing_distance_m": "SYLC_COMFORT_VIEWING_DISTANCE_M",
    "synth3d_comfort_ipd_m": "SYLC_COMFORT_IPD_M",
    "synth3d_comfort_soft_vac_d": "SYLC_COMFORT_SOFT_VAC_D",
    "synth3d_comfort_hard_vac_d": "SYLC_COMFORT_HARD_VAC_D",
}


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {
            "0", "false", "off", "disabled", "no", "n",
        }
    return bool(value)


def calibration_is_complete(settings: Mapping) -> bool:
    """Return whether this installation completed the current wizard."""
    try:
        return int(settings.get(CALIBRATION_VERSION_KEY, 0)) >= \
            CALIBRATION_SCHEMA_VERSION
    except (TypeError, ValueError, OverflowError):
        return False


def editable_calibration_values(settings: Mapping) -> dict:
    """Return safe field values for the wizard, ignoring environment overrides."""
    values = dict(DEFAULT_CALIBRATION)
    for key, default in DEFAULT_CALIBRATION.items():
        raw = settings.get(key, default)
        try:
            values[key] = _as_bool(raw) if isinstance(default, bool) else float(raw)
        except (TypeError, ValueError, OverflowError):
            values[key] = default
    values["synth3d_comfort_horizontal_pixels"] = int(round(
        values["synth3d_comfort_horizontal_pixels"]))
    return values


@dataclass(frozen=True)
class StereoCalibrationProfile:
    screen_height_m: float
    envelope: StereoComfortEnvelope
    enabled: bool = True


def load_calibration_profile(
        settings: Mapping,
        environ: Optional[Mapping[str, str]] = None) -> StereoCalibrationProfile:
    """Build the active calibrated envelope (environment > settings > defaults)."""
    env = os.environ if environ is None else environ

    def value(key):
        env_key = _ENVIRONMENT_KEYS[key]
        return env[env_key] if env_key in env else settings.get(
            key, DEFAULT_CALIBRATION[key])

    enabled = _as_bool(value("synth3d_comfort_enabled"))
    screen_height_m = float(value("synth3d_comfort_screen_height_m"))
    if screen_height_m <= 0.0:
        raise ValueError("screen height must be positive")
    geometry = StereoDisplayGeometry(
        screen_width_m=float(value("synth3d_comfort_screen_width_m")),
        horizontal_pixels=int(round(float(value(
            "synth3d_comfort_horizontal_pixels")))),
        viewing_distance_m=float(value(
            "synth3d_comfort_viewing_distance_m")),
        interpupillary_distance_m=float(value("synth3d_comfort_ipd_m")),
    )
    envelope = StereoComfortEnvelope(
        geometry,
        soft_vac_diopters=float(value("synth3d_comfort_soft_vac_d")),
        hard_vac_diopters=float(value("synth3d_comfort_hard_vac_d")),
    )
    return StereoCalibrationProfile(screen_height_m, envelope, enabled)


def save_calibration(
        settings: MutableMapping, values: Mapping) -> StereoCalibrationProfile:
    """Validate, persist into ``settings``, and mark the current wizard complete."""
    staged = dict(settings)
    for key in DEFAULT_CALIBRATION:
        if key in values:
            staged[key] = values[key]
    profile = load_calibration_profile(staged, environ={})
    settings.update({key: staged.get(key, default)
                     for key, default in DEFAULT_CALIBRATION.items()})
    settings[CALIBRATION_VERSION_KEY] = CALIBRATION_SCHEMA_VERSION
    return profile


class StereoCalibrationWizard(QWizard):
    """Two-step first-run assistant for physical projection geometry."""

    def __init__(self, settings: Mapping, parent=None):
        super().__init__(parent)
        self.setObjectName("stereoCalibrationWizard")
        self.setWindowTitle("Stereo Lab — Projection Room Calibration")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.setButtonText(QWizard.WizardButton.FinishButton, "Save calibration")
        self.setMinimumWidth(560)

        intro = QWizardPage(self)
        intro.setTitle("Calibrate Stereo Lab for this room")
        intro.setSubTitle(
            "This one-time setup converts rendered pixel disparity into the "
            "physical binocular demand at your viewing position.")
        intro_layout = QVBoxLayout(intro)
        explanation = QLabel(
            "Measure the illuminated image—not the wall or projector panel. "
            "Stereo Lab will use these dimensions to preserve the intended 3D "
            "effect below the comfort knee and progressively limit only "
            "excessive disparity. You can reopen this assistant at any time "
            "from the 2D→3D menu.", intro)
        explanation.setWordWrap(True)
        explanation.setTextFormat(Qt.TextFormat.PlainText)
        intro_layout.addWidget(explanation)
        intro_layout.addStretch(1)
        self.addPage(intro)

        values = editable_calibration_values(settings)
        room = QWizardPage(self)
        room.setTitle("Projection room")
        room.setSubTitle("Enter the geometry at the normal viewing position.")
        room_layout = QVBoxLayout(room)
        form = QFormLayout()

        self.screen_width = self._double_spin(
            "screenWidthMetres", 0.30, 30.0, 2, " m",
            values["synth3d_comfort_screen_width_m"])
        self.screen_height = self._double_spin(
            "screenHeightMetres", 0.20, 20.0, 2, " m",
            values["synth3d_comfort_screen_height_m"])
        self.viewing_distance = self._double_spin(
            "viewingDistanceMetres", 0.40, 50.0, 2, " m",
            values["synth3d_comfort_viewing_distance_m"])
        self.horizontal_pixels = QSpinBox(room)
        self.horizontal_pixels.setObjectName("horizontalPixels")
        self.horizontal_pixels.setRange(640, 16384)
        self.horizontal_pixels.setSingleStep(128)
        self.horizontal_pixels.setSuffix(" px")
        self.horizontal_pixels.setValue(
            values["synth3d_comfort_horizontal_pixels"])
        self.ipd_mm = self._double_spin(
            "viewerIpdMillimetres", 45.0, 80.0, 1, " mm",
            values["synth3d_comfort_ipd_m"] * 1000.0)
        self.comfort_enabled = QCheckBox(
            "Enable calibrated comfort correction", room)
        self.comfort_enabled.setObjectName("comfortCorrectionEnabled")
        self.comfort_enabled.setChecked(
            values["synth3d_comfort_enabled"])

        form.addRow("Illuminated image width", self.screen_width)
        form.addRow("Illuminated image height", self.screen_height)
        form.addRow("Viewing distance", self.viewing_distance)
        form.addRow("Horizontal resolution", self.horizontal_pixels)
        form.addRow("Viewer IPD", self.ipd_mm)
        room_layout.addLayout(form)
        room_layout.addWidget(self.comfort_enabled)

        self.summary = QLabel(room)
        self.summary.setObjectName("calibrationSummary")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        room_layout.addWidget(self.summary)
        room_layout.addStretch(1)
        self.addPage(room)

        for widget in (
                self.screen_width, self.screen_height,
                self.viewing_distance, self.horizontal_pixels, self.ipd_mm):
            widget.valueChanged.connect(self._update_summary)
        self.comfort_enabled.toggled.connect(self._update_summary)
        self._soft_vac = values["synth3d_comfort_soft_vac_d"]
        self._hard_vac = values["synth3d_comfort_hard_vac_d"]
        self._update_summary()

    @staticmethod
    def _double_spin(name, minimum, maximum, decimals, suffix, value):
        spin = QDoubleSpinBox()
        spin.setObjectName(name)
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSuffix(suffix)
        spin.setValue(value)
        return spin

    def calibration_values(self) -> dict:
        return {
            "synth3d_comfort_enabled": self.comfort_enabled.isChecked(),
            "synth3d_comfort_screen_width_m": self.screen_width.value(),
            "synth3d_comfort_screen_height_m": self.screen_height.value(),
            "synth3d_comfort_horizontal_pixels": self.horizontal_pixels.value(),
            "synth3d_comfort_viewing_distance_m": self.viewing_distance.value(),
            "synth3d_comfort_ipd_m": self.ipd_mm.value() / 1000.0,
            "synth3d_comfort_soft_vac_d": self._soft_vac,
            "synth3d_comfort_hard_vac_d": self._hard_vac,
        }

    def _update_summary(self, *_args):
        try:
            profile = load_calibration_profile(
                self.calibration_values(), environ={})
            envelope = profile.envelope
            state = "enabled" if profile.enabled else "disabled"
            self.summary.setText(
                f"Comfort correction: {state}. Soft knee at "
                f"{envelope.soft_pixels:.1f} px "
                f"({envelope.soft_percent:.2f}% of image width); "
                f"asymptotic limit {envelope.hard_pixels:.1f} px "
                f"({envelope.hard_percent:.2f}%).")
        except (TypeError, ValueError, OverflowError):
            self.summary.setText("The current geometry is not valid.")


__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "CALIBRATION_VERSION_KEY",
    "DEFAULT_CALIBRATION",
    "StereoCalibrationProfile",
    "StereoCalibrationWizard",
    "calibration_is_complete",
    "editable_calibration_values",
    "load_calibration_profile",
    "save_calibration",
]
