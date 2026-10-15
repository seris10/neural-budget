"""One-at-a-time and 2^k factorial scenarios over the ranges in reference_data/scenario_ranges.json."""

from __future__ import annotations

import itertools
import json
from importlib import resources

from .model import cortical_budget
from .params import reference_parameters, values


def scenario_ranges() -> dict:
    return json.loads((resources.files("neural_budget") / "reference_data" / "scenario_ranges.json").read_text())


def _summary(b) -> dict:
    g = b.groups()
    return {"signaling_umol_g_min": b.umol_per_g_min(b.signaling), "signaling_atp_neuron_s": b.signaling,
            **{f"{k} (ATP/neuron/s)": v for k, v in g.items()}}


def one_at_a_time() -> list[dict]:
    base = values(reference_parameters())
    ref = cortical_budget(base)
    ref_s = ref.umol_per_g_min(ref.signaling)
    rows = []
    for name, r in scenario_ranges()["ranges"].items():
        for level in ("low", "high"):
            b = cortical_budget(dict(base, **{name: r[level]}))
            s = _summary(b)
            rows.append({"analysis": "one-at-a-time", "parameter": name, "level": level, "value": r[level],
                         "reference_value": base[name], "range_basis": r["basis"], "range_source": r["source"],
                         "change_from_reference_umol_g_min": s["signaling_umol_g_min"] - ref_s,
                         "relative_change": s["signaling_umol_g_min"] / ref_s - 1, **s})
    return rows


def tornado(rows) -> list[dict]:
    """Per parameter: low and high outcomes and their swing, largest swing first."""
    by = {}
    for r in rows:
        by.setdefault(r["parameter"], {})[r["level"]] = r
    out = []
    for name, d in by.items():
        lo, hi = d["low"], d["high"]
        out.append({"parameter": name, "basis": lo["range_basis"], "low_value": lo["value"], "high_value": hi["value"],
                    "signaling_at_low": lo["signaling_umol_g_min"], "signaling_at_high": hi["signaling_umol_g_min"],
                    "swing_umol_g_min": abs(hi["signaling_umol_g_min"] - lo["signaling_umol_g_min"])})
    return sorted(out, key=lambda r: -r["swing_umol_g_min"])


def factorial() -> list[dict]:
    """Full 2^k design over the factorial parameters at their low and high values."""
    base = values(reference_parameters())
    ranges = scenario_ranges()
    names = ranges["factorial"]
    rows = []
    for levels in itertools.product(("low", "high"), repeat=len(names)):
        changes = {n: ranges["ranges"][n][lv] for n, lv in zip(names, levels)}
        b = cortical_budget(dict(base, **changes))
        rows.append({"analysis": "factorial", **{f"{n}_level": lv for n, lv in zip(names, levels)},
                     **{n: changes[n] for n in names}, **_summary(b)})
    return rows


def factorial_effects(rows, names) -> list[dict]:
    """Main effects and two-way interactions on signaling (umol/g/min), coded -1/+1."""
    def coded(r, n):
        return 1.0 if r[f"{n}_level"] == "high" else -1.0

    y = [r["signaling_umol_g_min"] for r in rows]
    n_runs = len(rows)
    out = []
    terms = [(n,) for n in names] + list(itertools.combinations(names, 2))
    scale = max(abs(v) for v in y)
    for term in terms:
        contrast = sum(yi * _prod(coded(r, t) for t in term) for yi, r in zip(y, rows))
        effect = 2 * contrast / n_runs
        out.append({"term": " x ".join(term), "effect_umol_g_min": 0.0 if abs(effect) < 1e-12 * scale else effect})
    return out


def _prod(xs):
    p = 1.0
    for x in xs:
        p *= x
    return p
