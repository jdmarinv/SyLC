"""Characterization tests for PlayerWindow's Synth3D coordination."""

import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

from sylc import synth3d_coordination_mixin as coordination_module
from sylc.synth3d_coordination_mixin import Synth3DCoordinationMixin


class _Harness(Synth3DCoordinationMixin):
    def __init__(self):
        self._app_settings = {}
        self._synth3d_active = False
        self._synth3d_preset = 'comfort'
        self._synth3d_strength = 0.8
        self._synth3d_convergence = 0.62
        self._synth3d_pending_cuts = []
        self._synth3d_matte_cut_seen_ms = -math.inf
        self._synth3d_matte_floor_pts_ms = -math.inf
        self._synth3d_aspect_override = None
        self._synth3d_aspect_unavailable_key = None
        self._synth3d_matte_service = None
        self.mvc_mode_active = False
        self._hevc_mode_active = False
        self.mvc_decoder_thread = None
        self.hevc_thread = None
        self.has_media = True
        self.saved = 0
        self.pushed = 0
        self.cleared = 0
        self.remembered = []
        self.notifications = []
        self.native_lost = 0

    def _save_app_settings(self):
        self.saved += 1

    def _push_synth3d_to_widgets(self):
        self.pushed += 1

    def _remember_for_file(self, **fields):
        self.remembered.append(fields)

    def show_3d_notification(self, message, success=False):
        self.notifications.append((message, success))

    def _update_synth3d_menu_state(self):
        pass

    def _synth3d_clear_human_matte(self):
        self.cleared += 1

    def _content_is_3d(self):
        return False

    def _display_widgets(self):
        return []

    def _synth3d_on_native_path_lost(self):
        self.native_lost += 1


class _MatteService:
    def __init__(self):
        self.resets = []

    def reset(self, reason):
        self.resets.append(reason)


def test_media_fps_snaps_integer_millisecond_quantization():
    assert Synth3DCoordinationMixin._snap_media_fps(23.81) == 23.976
    assert Synth3DCoordinationMixin._snap_media_fps(59.8) == 59.94
    assert Synth3DCoordinationMixin._snap_media_fps(17.0) == 17.0
    assert Synth3DCoordinationMixin._snap_media_fps(float('nan')) is None


def test_cut_boundary_queues_every_distinct_future_cut():
    """The regression this replaced: a LATER cut arriving while an earlier one
    waits for its presentation used to be dropped, silently.  The scout leads
    by up to 12 decoded frames, so two cuts inside that window is ordinary on
    short shots -- and it cost 23% of matte epoch advances, measured live
    (30 boundaries consumed by the depth path against 23 matte resets)."""
    player = _Harness()

    assert player._synth3d_note_cut_boundary(1200.0) is True
    assert player._synth3d_note_cut_boundary(1500.0) is True   # was dropped
    assert player._synth3d_note_cut_boundary(900.0) is True
    assert player._synth3d_pending_cuts == [900.0, 1200.0, 1500.0]

    # Duplicates within half a millisecond are the same cut reported twice.
    assert player._synth3d_note_cut_boundary(1200.3) is False

    player._synth3d_matte_cut_seen_ms = 900.0
    assert player._synth3d_note_cut_boundary(900.2) is False
    assert player._synth3d_note_cut_boundary(-1) is False


def test_pending_cut_queue_is_bounded_and_counts_what_it_drops():
    player = _Harness()
    for index in range(player.SYNTH3D_MAX_PENDING_CUTS + 3):
        assert player._synth3d_note_cut_boundary(1000.0 + index * 50.0) is True
    assert len(player._synth3d_pending_cuts) == player.SYNTH3D_MAX_PENDING_CUTS
    # Silent loss is what hid the original defect; a drop must leave a trace.
    assert player._synth3d_cuts_dropped == 3
    # The IMMINENT cut must survive: the advisory pump reports the nearest one
    # every 100 ms, so evicting it re-adds and re-evicts it at pump cadence --
    # measured at 10 drops/second before the eviction end was corrected.
    assert player._synth3d_pending_cuts[0] == 1000.0


def test_re_noting_the_imminent_cut_never_churns_the_queue():
    player = _Harness()
    for index in range(player.SYNTH3D_MAX_PENDING_CUTS):
        assert player._synth3d_note_cut_boundary(1000.0 + index * 50.0) is True
    nearest = player._synth3d_pending_cuts[0]
    # What the 100 ms pump does: re-report the nearest cut, over and over.
    for _ in range(50):
        assert player._synth3d_note_cut_boundary(nearest) is False
    assert getattr(player, '_synth3d_cuts_dropped', 0) == 0
    assert player._synth3d_pending_cuts[0] == nearest


def test_matte_epoch_advances_only_when_cut_reaches_presentation():
    player = _Harness()
    service = _MatteService()
    player._synth3d_pending_cuts = [1000.0]

    assert player._synth3d_apply_matte_cut_if_due(999.4, service) is False
    assert service.resets == []

    assert player._synth3d_apply_matte_cut_if_due(999.5, service) is True
    assert service.resets == ['shot boundary @1000.000 ms']
    assert player._synth3d_pending_cuts == []
    assert player._synth3d_matte_cut_seen_ms == 1000.0
    assert player._synth3d_matte_floor_pts_ms == 1000.0
    assert player.cleared == 1


def test_media_pacing_reports_short_side_and_authored_rate():
    player = _Harness()
    player.mvc_decoder_thread = SimpleNamespace(target_frame_time=0.042)
    luma = SimpleNamespace(shape=(808, 1920))

    fps, short_side = player._synth3d_media_pacing((luma,))

    assert fps == 23.976
    assert short_side == 808


def test_strength_and_convergence_are_clamped_and_mark_custom():
    player = _Harness()

    player.set_synth3d_strength(100, persist=False)
    assert player._synth3d_strength == 3.0
    assert player._synth3d_preset == 'custom'

    player.set_synth3d_convergence(-50, persist=False)
    assert player._synth3d_convergence == 0.0
    player.set_synth3d_convergence(250, persist=False)
    assert player._synth3d_convergence == 1.0
    assert player.saved == 0
    assert player.pushed == 3


def test_named_preset_updates_settings_and_renderer():
    player = _Harness()

    player.apply_synth3d_preset('cinema')

    assert player._synth3d_preset == 'cinema'
    assert (player._synth3d_strength, player._synth3d_convergence) == (1.4, 0.52)
    assert player._app_settings['synth3d_preset'] == 'cinema'
    assert player.saved == 1
    assert player.pushed == 1
    assert player.notifications[-1] == ('2D->3D preset: Cinema', True)


def test_retired_immersion_preset_migrates_to_cinema_and_is_not_selectable():
    assert Synth3DCoordinationMixin._normalise_synth3d_preset(
        'immersion') == 'cinema'
    assert Synth3DCoordinationMixin._normalise_synth3d_preset(
        'unknown') == 'custom'
    assert set(Synth3DCoordinationMixin._SYNTH3D_PRESETS) == {
        'comfort', 'cinema'}

    player = _Harness()
    before = (player._synth3d_preset, player._synth3d_strength,
              player._synth3d_convergence, player.saved, player.pushed)
    player.apply_synth3d_preset('immersion')
    after = (player._synth3d_preset, player._synth3d_strength,
             player._synth3d_convergence, player.saved, player.pushed)
    assert after == before


def test_decoder_restart_does_not_report_native_path_loss():
    player = _Harness()

    player._synth3d_handle_decoder_stop(True)
    assert player.native_lost == 0

    player._synth3d_handle_decoder_stop(False)
    assert player.native_lost == 1


def test_eligibility_requires_loaded_2d_native_session():
    player = _Harness()
    assert player._synth3d_eligible() is False

    player.mvc_mode_active = True
    assert player._synth3d_eligible() is True

    player.has_media = False
    assert player._synth3d_eligible() is False


def test_status_parser_keeps_key_value_diagnostics():
    fields = Synth3DCoordinationMixin._synth3d_status_fields(
        'state=running provider=TensorRT fps=23.9 ignored clients=2')
    assert fields == {
        'state': 'running',
        'provider': 'TensorRT',
        'fps': '23.9',
        'clients': '2',
    }


def test_native_support_gate_is_injected_without_main_module_import():
    player = _Harness()
    with tempfile.TemporaryDirectory() as root:
        model = Path(root) / 'model.onnx'
        runtime = Path(root) / 'runtime'
        model.touch()
        runtime.mkdir()
        (runtime / 'onnxruntime.dll').touch()
        player._synth3d_model_path = lambda: (str(model), 756)
        player._synth3d_ort_dir = lambda _model=None: str(runtime)
        original = coordination_module.NATIVE_RENDER_AVAILABLE
        try:
            coordination_module.configure_synth3d_support(False)
            assert player._synth3d_unsupported_reason() == 'renderer'
            coordination_module.configure_synth3d_support(True)
            assert player._synth3d_unsupported_reason() is None
        finally:
            coordination_module.configure_synth3d_support(original)


if __name__ == '__main__':
    import unittest

    unittest.main()


def test_queued_cuts_drain_one_per_presentation():
    player = _Harness()
    service = _MatteService()
    player._synth3d_note_cut_boundary(1000.0)
    player._synth3d_note_cut_boundary(1200.0)

    assert player._synth3d_apply_matte_cut_if_due(1000.0, service) is True
    assert player._synth3d_pending_cuts == [1200.0]
    assert player._synth3d_apply_matte_cut_if_due(1200.0, service) is True
    assert service.resets == ['shot boundary @1000.000 ms',
                              'shot boundary @1200.000 ms']
    assert player._synth3d_pending_cuts == []
    assert getattr(player, '_synth3d_cuts_coalesced', 0) == 0


def test_cuts_overtaken_by_presentation_coalesce_and_are_counted():
    """Several due at once means their shots went by faster than the pump ran.
    One reset is all that remains possible, but it must not look clean."""
    player = _Harness()
    service = _MatteService()
    player._synth3d_note_cut_boundary(1000.0)
    player._synth3d_note_cut_boundary(1200.0)

    assert player._synth3d_apply_matte_cut_if_due(1500.0, service) is True
    assert service.resets == ['shot boundary @1200.000 ms']
    assert player._synth3d_cuts_coalesced == 1
    assert player._synth3d_matte_cut_seen_ms == 1200.0
