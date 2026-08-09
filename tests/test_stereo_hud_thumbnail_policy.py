"""FramePack must not suppress hover thumbnails in the main window."""

from sylc.stereo_hud import StereoHudController


def test_main_window_thumbnail_remains_native_with_external_framepack():
    # Active stereo HUD + external FramePack means hide_native=False: keep the
    # ordinary main-window tooltip and optionally paint the HUD copy too.
    assert StereoHudController._thumbnail_uses_hud(True, False) is False


def test_embedded_stereo_uses_hud_thumbnail_instead_of_tool_window():
    assert StereoHudController._thumbnail_uses_hud(True, True) is True
    assert StereoHudController._thumbnail_uses_hud(False, True) is False
