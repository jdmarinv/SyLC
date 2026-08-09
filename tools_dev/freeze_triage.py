"""Locate a 2D->3D freeze: which stage stopped, and which merely starved.

The freeze looks the same from every angle -- image still, audio fine, worker
alive -- because the stages are chained by back-pressure.  Three readings taken
TOGETHER separate them, and no two of them suffice:

  age_ms        Age of the PUBLISHED depth map.  This is the freeze itself:
                the picture stops being refreshed when this starts climbing,
                whatever the cause.  Anchor every window on it.

  taps / busy   taps counts every wants_input() call, busy the subset refused
                because the worker has not taken the previous map yet.  That
                refusal moves no grant and no submit, so before `busy` existed
                a stalled worker and a stalled renderer produced byte-identical
                counters.  taps flat = the renderer is not asking; busy ~ taps
                = it is asking and the worker owes it a map.

  wphase        The region worker_main last ENTERED (never cleared to "run").
                `input` with a small age means the worker is spinning on its
                100 ms input timeout: healthy, and starving.

  [LOOKAHEAD] frames
                Fed by the decode/pump path, which is independent of
                presentation.  Full rate here while taps is flat is the proof
                that decode is alive and presentation is not.

Usage:  python tools_dev/freeze_triage.py <player.log>
"""

import datetime as dt
import re
import sys
from pathlib import Path

# The published map is refreshed ~12x/s in health; three seconds is a freeze
# no viewer would miss.
FREEZE_MS = 3000
# A worker parked in one region this long crossed no marked point at all.
STALL_MS = 3000


def _fields(line):
    return dict(re.findall(r"\b(\w+)=([\w.:+-]+)", line))


def _int(row, key, default=0):
    try:
        return int(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _load(path):
    """Status samples, each carrying the nearest look-ahead and pump readings."""
    samples, scout, pump = [], [], []
    for line in Path(path).read_text(errors="ignore").splitlines():
        stamp = re.match(r"(\S+ \S+?),", line)
        if not stamp:
            continue
        when = dt.datetime.strptime(stamp.group(1), "%Y-%m-%d %H:%M:%S")
        if "[LOOKAHEAD]" in line:
            scout.append((when, _int(_fields(line), "frames")))
        elif "[PUMP]" in line:
            pump.append((when, _fields(line)))
        elif "state=" in line and "wphase=" in line:
            row = _fields(line)
            row["_t"] = when
            samples.append(row)

    def _nearest(seq, when, span):
        near = [(abs((t - when).total_seconds()), v) for t, v in seq]
        near = [(d, v) for d, v in near if d <= span]
        return min(near)[1] if near else None

    for row in samples:
        row["_scout"] = _nearest(scout, row["_t"], 2)
        # The pump line is rate-limited to 30 s in health, so it is matched on a
        # wider span than the look-ahead count and may legitimately be absent.
        row["_pump"] = _nearest(pump, row["_t"], 35)
    return samples


def _window(samples):
    """The freeze, if there is one; otherwise the whole log."""
    frozen = [i for i, r in enumerate(samples)
              if _int(r, "age_ms") >= FREEZE_MS]
    if not frozen:
        return 0, len(samples) - 1, "le log"
    # Reach back one sample to catch the healthy rate just before the freeze --
    # but only if it IS healthy.  A `state=init` sample has every counter at
    # zero, so including it charges the whole warm-up ramp to the freeze window
    # and turns a renderer that stopped calling into one that calls 1.1x/s.
    lo = frozen[0]
    if lo > 0 and samples[lo - 1].get("state") == "running":
        lo -= 1
    return lo, frozen[-1], "le gel"


def _rate(first, last, key, span):
    return (_int(last, key) - _int(first, key)) / span


def main(path):
    samples = _load(path)
    if not samples:
        print("aucune ligne de statut exploitable")
        return 1
    if "taps" not in samples[-1]:
        print("pas de champ taps= : c'est l'ancien .pyd qui tourne "
              "(runtime/ à jour ?)")
        return 1

    lo, hi, label = _window(samples)
    first, last = samples[lo], samples[hi]
    span = max((last["_t"] - first["_t"]).total_seconds(), 1.0)
    print(f"{len(samples)} relevés ; mesures sur {label} : "
          f"{hi - lo + 1} relevés, {span:.0f} s")
    print(f"map publiée : âge {_int(first, 'age_ms')} ms -> "
          f"{_int(last, 'age_ms')} ms\n")

    taps = _rate(first, last, "taps", span)
    busy = _rate(first, last, "busy", span)
    submits = _rate(first, last, "submits", span)
    print(f"  renderer   taps={taps:.1f}/s  refusés busy={busy:.1f}/s  "
          f"soumis={submits:.1f}/s")
    if first["_scout"] is not None and last["_scout"] is not None:
        decode = (last["_scout"] - first["_scout"]) / span
        print(f"  décodeur   {decode:.1f} images/s vers le scout")
    else:
        decode = None
    pump = last.get("_pump")
    if pump:
        held = _int(pump, "held_ms")
        print(f"  pump       décodées={_int(pump, 'decoded')} "
              f"émises={_int(pump, 'emitted')} "
              f"acquittées={_int(pump, 'acked')}   "
              f"perdues: sync={_int(pump, 'sync_drop')} "
              f"cancel={_int(pump, 'cancel')} "
              f"backpressure={_int(pump, 'backpressure')}")
        print(f"  jeton      {pump.get('token')} depuis {held} ms "
              f"(pire {_int(pump, 'held_max_ms')} ms, "
              f"{_int(pump, 'expired')} expirations)")
        lost = _int(pump, "emitted") - _int(pump, "acked")
        if _int(pump, "expired") or lost > 1:
            print(f"      {lost} émission(s) sans acquittement — le garde-fou "
                  "a relancé la livraison,\n      mais l'acquittement perdu "
                  "reste un défaut à part entière.")
    print(f"  worker     wphase={last.get('wphase')} "
          f"depuis {_int(last, 'wphase_ms')} ms   "
          f"pipe ipipe={last.get('ipipe')}")
    mrst = _rate(first, last, "mrst", span) * 60.0
    print(f"  resets     {mrst:.0f}/min consommés   "
          f"(rst={last.get('rst')} seek:géo:attach, "
          f"mdrop={_int(last, 'mdrop')})")

    print()
    if _int(last, "age_ms") < FREEZE_MS:
        print("  pas de gel dans ce log.")
        near = max((_rate(a, b, "busy", 1) / max(_rate(a, b, "taps", 1), 1e-9)
                    for a, b in zip(samples, samples[1:])
                    if _int(b, "taps") - _int(a, "taps") >= 20), default=0.0)
        if near > 0.90:
            print(f"  >>> QUASI-ACCIDENT : jusqu'à {near * 100:.0f}% des taps "
                  "refusés sur une fenêtre.\n      Même mécanisme qu'un gel, "
                  "dénoué avant de basculer.")
        return 0

    if mrst >= 60.0:
        print(f"  >>> TEMPÊTE DE RESETS ({mrst:.0f}/min) : chaque reset fait "
              "JETER la map en vol.\n      Rien n'est bloqué, rien n'est "
              "publié. Chercher qui appelle notify_seek.")
    elif taps < 1.0:
        print("  >>> Le renderer N'APPELLE PLUS le service.")
        if pump and _int(pump, "held_ms") >= 1000:
            print(f"      CAUSE : le jeton de présentation est pris depuis "
                  f"{_int(pump, 'held_ms')} ms.\n      "
                  f"presentation_consumed() n'est jamais revenu, donc chaque "
                  f"image décodée\n      part en backpressure "
                  f"({_int(pump, 'backpressure')} au total) et aucune fenêtre "
                  f"n'est servie.")
        elif pump and _int(pump, "emitted") - _int(pump, "acked") > 1:
            print(f"      {_int(pump, 'emitted') - _int(pump, 'acked')} "
                  "émissions sans acquittement : des invocations queued se "
                  "perdent.")
        if decode and decode > 10.0:
            print(f"      Le décodeur tourne pourtant à {decode:.0f} img/s : "
                  "décodage vivant,\n      présentation arrêtée. La panne est "
                  "dans le chemin de présentation,\n      EN AMONT du service "
                  "de profondeur.")
        if last.get("wphase") == "input":
            print("      wphase=input confirme : le worker tourne à vide sur "
                  "son timeout\n      de 100 ms. Il jeûne, il n'est pas "
                  "bloqué.")
    elif busy > 0.8 * taps:
        print("  >>> Le renderer appelle mais le service le REFUSE : le worker "
              "n'a pas repris\n      la map précédente. La panne est côté "
              f"worker, région wphase={last.get('wphase')}.")
    elif _int(last, "wphase_ms") >= STALL_MS:
        region = last.get("wphase", "?")
        print(f"  >>> Le worker est parké dans `{region}` depuis "
              f"{_int(last, 'wphase_ms')} ms.")
        cause = {
            "recycle": "tâche de flot orpheline laissée vivante par un drain",
            "drain_join": "idem, côté purge de seek",
            "drain_res": "attente d'un résultat d'inférence qui ne viendra pas",
            "retry_res": "attente du résultat de la ré-inférence de coupe",
            "result": "attente normale du pipe — regarder ipipe",
            "submit": "le pipe ne reprend pas le job précédent",
        }.get(region)
        if cause:
            print(f"      {cause}")
    else:
        print("  >>> Gel sans coupable évident : taps et busy sont sains et "
              "le worker\n      progresse. Regarder la publication elle-même.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
