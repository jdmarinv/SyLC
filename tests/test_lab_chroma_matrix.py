from copy import deepcopy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools_dev" / "lab_chroma_matrix.py"
SPEC = importlib.util.spec_from_file_location("lab_chroma_matrix", MODULE_PATH)
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


def _baseline():
    protocol = matrix.current_protocol()
    return {
        "protocol": deepcopy(protocol),
        "rows": [{"seed": seed} for seed in protocol["seeds"]],
    }, protocol


def test_baseline_contract_accepts_exact_protocol_and_rows():
    baseline, protocol = _baseline()
    assert matrix.validate_baseline_contract(baseline, protocol) == []


def test_baseline_contract_rejects_deleted_known_red_topology():
    baseline, protocol = _baseline()
    protocol["seeds"] = [
        seed for seed in protocol["seeds"]
        if seed not in (20260809, 20260811)
    ]
    failures = matrix.validate_baseline_contract(baseline, protocol)
    assert failures
    assert any("protocol differs" in failure for failure in failures)
    assert any("extra=[20260809, 20260811]" in failure for failure in failures)


def test_baseline_contract_rejects_missing_and_duplicate_rows():
    baseline, protocol = _baseline()
    baseline["rows"].pop(2)
    baseline["rows"].append({"seed": protocol["seeds"][0]})
    failures = matrix.validate_baseline_contract(baseline, protocol)
    assert any("duplicate seeds" in failure for failure in failures)
    assert any("missing=[20260809]" in failure for failure in failures)


def test_baseline_contract_rejects_geometry_drift():
    baseline, protocol = _baseline()
    protocol["strength"] = 2.5
    failures = matrix.validate_baseline_contract(baseline, protocol)
    assert any("protocol differs" in failure for failure in failures)


def test_main_rejects_deleted_known_red_topologies_before_gpu(monkeypatch, capsys):
    monkeypatch.setattr(
        matrix,
        "SEEDS",
        [seed for seed in matrix.SEEDS if seed not in (20260809, 20260811)],
    )
    monkeypatch.setattr(matrix.sys, "argv", ["lab_chroma_matrix.py"])

    def gpu_must_not_run(_seed):
        raise AssertionError("contract validation must run before the GPU probe")

    monkeypatch.setattr(matrix, "run_seed", gpu_must_not_run)

    assert matrix.main() == 2
    output = capsys.readouterr().out
    assert "BASELINE CONTRACT FAIL" in output
    assert "20260809" in output
    assert "20260811" in output
