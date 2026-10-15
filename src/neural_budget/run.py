"""Reproduce the reference case, run scenarios and sensitivity, and collect every table the report shows."""

from __future__ import annotations

from importlib import resources

import pandas as pd

from .model import ACTIVITY_DEPENDENT, GROUPS, SIGNALING_COMPONENTS, cortical_budget
from .params import SCENARIO_OPTIONS, apply_scenario, load_scenario, parameter_table, reference_parameters, values
from .reference import paper_checks, spreadsheet_checks, spreadsheet_findings
from .sensitivity import factorial, factorial_effects, one_at_a_time, scenario_ranges, tornado

EXCLUDED = [
    ("cerebellum and other regions", "only the cerebral cortex reference case is reproduced"),
    ("separate inhibitory neurons", "the reference model treats every cortical neuron as glutamatergic"),
    ("white matter and myelination", "the model covers gray matter"),
    ("temperature dependence", "the spreadsheet lists Q10 values but no formula uses them"),
    ("morphology-derived membrane area", "areas come from mean axon, soma and dendrite dimensions"),
    ("explicit Na+ current integration", "action-potential Na+ entry is the capacitive minimum times an overlap factor"),
    ("oxygen and glucose accounting", "the budget is in ATP only"),
    ("activity-dependent release probability", "release probability and per-vesicle costs do not change with rate"),
    ("comparison with artificial computers", "no conversion between biological events and machine operations is made"),
]


def _budget_options(sc):
    opts = sc.get("options", {})
    unknown = set(opts) - SCENARIO_OPTIONS
    if unknown:
        raise ValueError(f"unknown scenario options {sorted(unknown)}")
    kw = {}
    if "glia_scaling" in opts:
        kw["glia_scaling"] = opts["glia_scaling"]
    if "housekeeping_basis" in opts:
        kw["housekeeping_basis"] = opts["housekeeping_basis"]
    if "housekeeping_fraction" in opts:
        kw["housekeeping_fraction"] = float(opts["housekeeping_fraction"])
    return kw


def bundled_scenarios() -> list[dict]:
    root = resources.files("neural_budget") / "scenarios"
    return [load_scenario(p) for p in sorted(root.iterdir(), key=lambda p: p.name) if p.name.endswith(".toml")]


def budget_rows(label, b, kind) -> list[dict]:
    rows = []
    groups_of = {c: g for g, cs in GROUPS.items() for c in cs}
    for comp in SIGNALING_COMPONENTS:
        x = b.components[comp]
        rows.append({"case": label, "case_kind": kind, "component": comp, "group": groups_of[comp],
                     "atp_per_neuron_s": x, "umol_atp_per_g_min": b.umol_per_g_min(x),
                     "percent_of_signaling": 100 * x / b.signaling if b.signaling else float("nan")})
    for name, x in (("signaling total", b.signaling), ("housekeeping", b.housekeeping),
                    ("total including housekeeping", b.total)):
        rows.append({"case": label, "case_kind": kind, "component": name, "group": name, "atp_per_neuron_s": x,
                     "umol_atp_per_g_min": b.umol_per_g_min(x),
                     "percent_of_signaling": 100 * x / b.signaling if b.signaling else float("nan")})
    return rows


def limiting_cases(base) -> list[dict]:
    ref = cortical_budget(base)
    out = []
    zero = cortical_budget(dict(base, firing_rate=0.0))
    act = sum(zero.components[c] for c in ACTIVITY_DEPENDENT)
    rp_same = all(zero.components[c] == ref.components[c] for c in GROUPS["resting potentials"])
    out.append({"case": "zero firing rate", "expectation": "activity-dependent terms vanish; resting potentials unchanged",
                "observed": f"activity-dependent sum {act:g} ATP/neuron/s; resting terms unchanged: {rp_same}",
                "passed": act == 0.0 and rp_same})
    dens = cortical_budget(dict(base, neuron_density=2 * base["neuron_density"]))
    ratio = dens.umol_per_g_min(dens.signaling) / ref.umol_per_g_min(ref.signaling)
    out.append({"case": "neuron density doubled", "expectation": "per-gram energy doubles; per-neuron energy unchanged",
                "observed": f"per-gram ratio {ratio:.15g}; per-neuron ratio {dens.signaling / ref.signaling:.15g}",
                "passed": abs(ratio - 2) < 1e-12 and dens.signaling == ref.signaling})
    rel = cortical_budget(dict(base, release_probability=2 * base["release_probability"]))
    syn = [c for c in ACTIVITY_DEPENDENT if c not in GROUPS["action potentials"]]
    syn_ok = all(abs(rel.components[c] / ref.components[c] - 2) < 1e-12 for c in syn)
    other_ok = all(rel.components[c] == ref.components[c] for c in GROUPS["action potentials"]
                   + GROUPS["resting potentials"])
    out.append({"case": "release probability doubled",
                "expectation": "vesicle-driven terms double; action potentials and resting potentials unchanged",
                "observed": f"vesicle terms doubled: {syn_ok}; other terms unchanged: {other_ok}",
                "passed": syn_ok and other_ok})
    a = ref.housekeeping / ref.total
    b25 = cortical_budget(base, housekeeping_basis="total", housekeeping_fraction=0.25)
    s25 = cortical_budget(base, housekeeping_basis="signaling", housekeeping_fraction=0.25)
    out.append({"case": "housekeeping denominator",
                "expectation": "1/3 of signaling equals 25% of total; 25% of signaling is a smaller amount",
                "observed": (f"1/3 of signaling is {100 * a:.6f}% of total; 25% of total = {b25.housekeeping:.6g} "
                             f"ATP/neuron/s; 25% of signaling = {s25.housekeeping:.6g}"),
                "passed": abs(a - 0.25) < 1e-12 and abs(b25.housekeeping - ref.housekeeping) < 1e-6 * ref.housekeeping
                and s25.housekeeping < b25.housekeeping})
    return out


def run() -> dict:
    ref_params = reference_parameters()
    base = values(ref_params)
    ref = cortical_budget(base)
    budgets = budget_rows("reference (Howarth et al. 2012)", ref, "reference")
    scen_rows = []
    for sc in bundled_scenarios():
        p = apply_scenario(ref_params, sc["changes"], sc["name"])
        b = cortical_budget(values(p), **_budget_options(sc))
        budgets += budget_rows(sc["name"], b, "scenario")
        g, g0 = b.groups(), ref.groups()
        described = [f"{k} = {v:g}" for k, v in sc["changes"].items()]
        described += [f"{k} = {v}" for k, v in sc.get("options", {}).items()]
        scen_rows.append({"scenario": sc["name"], "file": sc["path"], "source": sc["source"],
                          "changes": "; ".join(described),
                          "signaling_umol_g_min": b.umol_per_g_min(b.signaling),
                          "total_umol_g_min": b.umol_per_g_min(b.total),
                          "signaling_change_percent": 100 * (b.signaling / ref.signaling - 1),
                          "total_change_percent": 100 * (b.total / ref.total - 1),
                          **{f"{k} change %": 100 * (g[k] / g0[k] - 1) for k in g}})
    oat = one_at_a_time()
    fac = factorial()
    names = scenario_ranges()["factorial"]
    sens = pd.concat([pd.DataFrame(oat), pd.DataFrame(fac)], ignore_index=True)
    env = [r["signaling_umol_g_min"] for r in fac]
    return {
        "reference": ref,
        "reference_reproduction": pd.DataFrame(spreadsheet_checks(ref) + paper_checks()),
        "component_budget": pd.DataFrame(budgets),
        "parameter_sources": pd.DataFrame(parameter_table(ref_params)),
        "sensitivity": sens,
        "tornado": pd.DataFrame(tornado(oat)),
        "factorial_effects": pd.DataFrame(factorial_effects(fac, names)),
        "factorial_envelope": {"min_umol_g_min": min(env), "max_umol_g_min": max(env), "parameters": names},
        "scenario_results": pd.DataFrame(scen_rows),
        "spreadsheet_findings": pd.DataFrame(spreadsheet_findings()),
        "limiting_cases": pd.DataFrame(limiting_cases(base)),
        "excluded": pd.DataFrame(EXCLUDED, columns=["excluded", "why"]),
    }


def write(result, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("reference_reproduction", "component_budget", "parameter_sources", "sensitivity",
                 "scenario_results", "spreadsheet_findings", "limiting_cases", "tornado", "factorial_effects", "excluded"):
        result[name].to_csv(out_dir / f"{name}.csv", index=False, float_format="%.10g")
