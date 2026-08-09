"""Reproducible Stereo Lab rivalry matrix over a recorded topology set.

Why this exists: ``probe_stereo_lab_rivalry_gpu.py`` measures ONE topology of
ONE binary and answers one question — is the Lab better than raw right now.
Release claims were written from ad-hoc runs whose topology set was never
recorded, so they could not be re-measured and drifted away from the tree.
This matrix fixes the protocol in code and keeps a committed baseline, so the
next commit is compared against a number that exists rather than a memory.

Three criteria are scored separately because they answer different questions
and one of them used to hide the other two:

  1. lab_vs_raw   — the probe's own gate: labC p95 must not exceed rawC p95 by
                    more than 0.50 code values.  Read it as "the Lab is not
                    materially worse", NOT as "the Lab is better": a topology
                    where the Lab costs +57% relative still passes when the
                    absolute cost is 0.37 code values.  The direction is
                    therefore reported separately, as a signed delta and a
                    ratio, and summarised as a better/worse/neutral count.
                    A tolerance gate answers "did this build break"; the
                    direction answers "is the Lab earning its place".
  2. old_vs_new   — did THIS change move any metric against the committed
                    baseline?  The only criterion that can attribute a
                    regression to a commit; needs the baseline file.
  3. strong_error — p99, the >8 / >16 code-value rates, and the amplitude of
                    what the Lab actually WROTE.  The zone percentiles cannot
                    see this class at all: the Lab leaves 97-98% of the zone
                    bit-identical, so a p95 over the whole zone is set by
                    untouched samples, and when ~40 touched samples move to the
                    top of the ranking they displace the entire tail — the p95
                    then reports that rank shift as if it were a degradation at
                    the p95 level.  Excluding the touched samples makes raw and
                    Lab p95 identical to four decimals on both topologies that
                    fail the p95 gate, which is the proof.  The touched_*
                    columns are restricted to samples the Lab wrote, so they
                    move if and only if the image changed.  Two correct shader
                    fixes were scored as failures before this existed.

Usage:
    .venv\\Scripts\\python.exe tools_dev\\lab_chroma_matrix.py
    .venv\\Scripts\\python.exe tools_dev\\lab_chroma_matrix.py --update-baseline

Requires Windows, D3D11 and a freshly built ``mvc_demuxer_cpp``.  Exit code 0
means no drift against the baseline AND no seed newly failing criterion 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tools_dev" / "probe_stereo_lab_rivalry_gpu.py"
BASELINE = ROOT / "tools_dev" / "lab_chroma_baseline.json"

# The recorded protocol.  Changing any of it invalidates the baseline and must
# be done together with --update-baseline and a note in RELEASE_NOTES.md.
SEEDS = [20260807 + index for index in range(8)]
GEOMETRY = {"width": 640, "height": 360, "strength": 2.4, "convergence": 0.5}

# GPU/driver differences move these by well under a code value; a real
# regression moves them by whole code values.  Absolute floor plus a relative
# term so the large-residual topologies are not held to an unreachable
# absolute tolerance.
DRIFT_ABS = 0.05
DRIFT_REL = 0.03

METRICS = ("p95_code", "p99_code", "over_8_code_pct", "over_16_code_pct")
# Amplitude of what the Lab wrote.  touched_count is included on purpose: a
# "fix" that silently stops correcting would improve every amplitude metric.
TOUCHED_METRICS = ("touched_count", "improved", "degraded",
                   "median_delta_code", "worst_delta_code",
                   "degraded_sum_code")


def current_protocol() -> dict:
    """Return the complete baseline identity, including ordered topologies."""
    return {"seeds": list(SEEDS), **GEOMETRY}


def validate_baseline_contract(baseline: dict, protocol: dict) -> list[str]:
    """Reject a baseline that does not describe exactly this matrix.

    Extra baseline rows are as dangerous as missing ones: without this check a
    developer could remove an inconvenient known-red seed from ``SEEDS`` and
    obtain a green run because the comparison loop only visits current rows.
    Ordering is part of the protocol so reports remain stable and auditable.
    """
    failures = []
    recorded = baseline.get("protocol")
    if recorded != protocol:
        failures.append(
            "baseline protocol differs from the recorded matrix protocol: "
            f"baseline={recorded!r}, current={protocol!r}")

    rows = baseline.get("rows")
    if not isinstance(rows, list):
        failures.append("baseline rows must be a list")
        return failures

    row_seeds = [row.get("seed") for row in rows if isinstance(row, dict)]
    if len(row_seeds) != len(rows):
        failures.append("every baseline row must be an object with a seed")
    if len(row_seeds) != len(set(row_seeds)):
        failures.append(f"baseline contains duplicate seeds: {row_seeds!r}")
    expected = protocol.get("seeds", [])
    if row_seeds != expected:
        missing = [seed for seed in expected if seed not in row_seeds]
        extra = [seed for seed in row_seeds if seed not in expected]
        failures.append(
            "baseline row seeds differ from the ordered protocol: "
            f"missing={missing!r}, extra={extra!r}, "
            f"baseline={row_seeds!r}, current={expected!r}")
    return failures


def run_seed(seed: int) -> dict:
    """One probe run.  Exit 1 is the probe's verdict, not a crash."""
    proc = subprocess.run(
        [sys.executable, str(PROBE), "--seed", str(seed),
         "--width", str(GEOMETRY["width"]), "--height", str(GEOMETRY["height"]),
         "--strength", str(GEOMETRY["strength"]),
         "--convergence", str(GEOMETRY["convergence"])],
        cwd=str(ROOT), capture_output=True, text=True)
    text = proc.stdout.lstrip()
    if not text.startswith("{"):
        raise RuntimeError(
            f"seed {seed}: probe produced no result\n{proc.stdout}\n{proc.stderr}")
    result, _ = json.JSONDecoder().raw_decode(text)
    # SystemExit("...") in the probe prints its verdict on stderr; stdout holds
    # only the JSON, whose last line is the closing brace.
    verdict = ("" if proc.returncode == 0 else proc.stderr.strip())
    return {
        "seed": seed,
        "probe_returncode": proc.returncode,
        "probe_verdict": verdict,
        "raw": {k: result["raw"][k] for k in METRICS},
        "lab": {k: result["lab"][k] for k in METRICS},
        "raw_chroma": {k: result["raw_chroma"][k] for k in METRICS},
        "lab_chroma": {k: result["lab_chroma"][k] for k in METRICS},
        "chroma_touched": {k: result["chroma_touched"][k]
                           for k in TOUCHED_METRICS},
        # If this ever goes false the rank-shift reading above stops holding
        # and the zone percentiles have to be re-interpreted, not just re-read.
        "untouched_p95_match": result["chroma_touched"]["untouched_p95_match"],
        "chroma_sample_count": result["lab_chroma"]["sample_count"],
        "changed_pixel_pct": result["changed_pixel_pct"],
        "max_change_code": result["max_change_code"],
    }


def drifted(new: float, old: float) -> bool:
    return abs(new - old) > max(DRIFT_ABS, DRIFT_REL * abs(old))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-baseline", action="store_true",
                        help="record the current run as the new reference")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    protocol = current_protocol()

    # Validate the committed contract BEFORE spending time on eight D3D11
    # runs.  --update-baseline is the one explicit operation allowed to change
    # the topology set and protocol; ordinary qualification must fail closed.
    baseline = None
    if not args.update_baseline:
        if not BASELINE.exists():
            print(f"FAIL: no baseline at {BASELINE}; run --update-baseline first")
            return 2
        try:
            baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"FAIL: invalid baseline {BASELINE}: {exc}")
            return 2
        contract_failures = validate_baseline_contract(baseline, protocol)
        if contract_failures:
            print("BASELINE CONTRACT FAIL")
            for entry in contract_failures:
                print(f"  {entry}")
            print("Run --update-baseline only if the protocol change is intentional.")
            return 2

    rows = [run_seed(seed) for seed in SEEDS]
    report = {"protocol": protocol, "rows": rows}

    if args.update_baseline:
        known = {}
        if BASELINE.exists():
            known = {
                str(r["seed"]): r.get("known_failing", "")
                for r in json.loads(BASELINE.read_text())["rows"]}
        for row in rows:
            note = known.get(str(row["seed"]), "")
            if note:
                row["known_failing"] = note
        BASELINE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"baseline written: {BASELINE}")
        return 0

    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    base_rows = {r["seed"]: r for r in baseline["rows"]}

    print(f"{'seed':>10} {'rawC p95':>9} {'labC p95':>9} {'labC p99':>9} "
          f"| {'wrote':>6} {'+':>4} {'-':>4} {'med':>7} {'worst':>7} {'degsum':>8} "
          f"| {'crit1':>5} {'crit2':>5}")
    failures = []
    better = worse = neutral = 0
    for row in rows:
        seed = row["seed"]
        base = base_rows.get(seed)
        # Criterion 1: the probe's own Lab-vs-raw verdict in THIS build.
        crit1 = "PASS" if row["probe_returncode"] == 0 else "FAIL"
        # Criterion 2: any metric moved against the committed baseline.
        moved = []
        if base is not None:
            for block in ("raw", "lab", "raw_chroma", "lab_chroma"):
                for metric in METRICS:
                    if drifted(row[block][metric], base[block][metric]):
                        moved.append(f"{block}.{metric} "
                                     f"{base[block][metric]:.4f}->"
                                     f"{row[block][metric]:.4f}")
            for metric in TOUCHED_METRICS:
                old = base.get("chroma_touched", {}).get(metric)
                if old is not None and drifted(
                        row["chroma_touched"][metric], old):
                    moved.append(f"chroma_touched.{metric} "
                                 f"{old:.4f}->{row['chroma_touched'][metric]:.4f}")
        if not row["untouched_p95_match"]:
            failures.append(
                f"seed {seed}: the Lab changed samples outside its own write "
                f"set; the zone percentiles no longer mean what this matrix "
                f"documents")
        crit2 = "DRIFT" if moved else "same"
        # Direction, independent of the tolerance gate above.
        raw_p95 = row["raw_chroma"]["p95_code"]
        lab_p95 = row["lab_chroma"]["p95_code"]
        delta = lab_p95 - raw_p95
        ratio = lab_p95 / max(raw_p95, 1e-9)
        if delta == 0.0:
            neutral += 1
        elif delta < 0.0:
            better += 1
        else:
            worse += 1
        t = row["chroma_touched"]
        print(f"{seed:>10} {raw_p95:>9.4f} {lab_p95:>9.4f} "
              f"{row['lab_chroma']['p99_code']:>9.4f} "
              f"| {t['touched_count']:>6} {t['improved']:>4} {t['degraded']:>4} "
              f"{t['median_delta_code']:>+7.2f} {t['worst_delta_code']:>+7.2f} "
              f"{t['degraded_sum_code']:>8.1f} "
              f"| {crit1:>5} {crit2:>5}")
        for entry in moved:
            print(f"           drift: {entry}")
            failures.append(f"seed {seed}: {entry}")
        # A seed the baseline records as failing stays failing without hiding
        # a NEW failure elsewhere: only an unexpected FAIL is a matrix failure.
        if base is not None and crit1 == "FAIL" and base["probe_returncode"] == 0:
            failures.append(f"seed {seed}: newly fails Lab-vs-raw "
                            f"({row['probe_verdict']})")
        if base is not None and base.get("known_failing") and crit1 == "FAIL":
            print(f"           known: {base['known_failing']}")

    # Criterion 3 is reported per seed above and summarised here: a localised
    # stripe barely moves p95 but shows up in the strong-error rates.
    worst = max(rows, key=lambda r: r["lab_chroma"]["p99_code"])
    print(f"\nstrongest lab chroma p99 = {worst['lab_chroma']['p99_code']:.4f} "
          f"(seed {worst['seed']}, "
          f">16 code in {worst['lab_chroma']['over_16_code_pct']:.2f}% of zone)")
    known_red = sum(1 for r in rows if r["probe_returncode"] != 0)
    print(f"lab-vs-raw tolerance gate: {len(rows) - known_red}/{len(rows)} pass, "
          f"{known_red} known-red")
    # Reported apart from the gate on purpose: passing the tolerance is not
    # the same as helping, and conflating them is what let a "improved in six"
    # claim survive a set where the Lab is worse in six.
    print(f"lab-vs-raw direction:      {better} better, {worse} worse, "
          f"{neutral} neutral (paired chroma p95, no tolerance)")

    if failures:
        print("\nMATRIX FAIL")
        for entry in failures:
            print(f"  {entry}")
        return 1
    print("\nMATRIX PASS (no drift vs baseline, no new Lab-vs-raw failure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
