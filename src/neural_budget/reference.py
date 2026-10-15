"""Compare the reference budget with the spreadsheet's cached cells and the paper's printed numbers."""

from __future__ import annotations

import re

from .model import cortical_budget
from .params import reference_parameters, reference_sheet, values

SPREADSHEET_TOLERANCE = 1e-12


def _cells():
    return {f["cell"]: f for f in reference_sheet()["formulas"]}


def spreadsheet_checks(budget=None) -> list[dict]:
    """Spreadsheet result cells against computed values (relative tolerance 1e-12)."""
    if budget is None:
        budget = cortical_budget(values(reference_parameters()))
    c = budget.components
    g = budget.groups()
    s = budget.signaling
    pairs = [
        ("F27", "vesicles released per neuron per second", budget.vesicles_per_s),
        ("K5", "glutamate recycling ATP per vesicle", budget.per_vesicle_costs["glutamate_recycling"]),
        ("M11", "AMPA-type receptor ATP per vesicle", budget.per_vesicle_costs["postsynaptic_ampa"]),
        ("M14", "NMDA receptor ATP per vesicle", budget.per_vesicle_costs["postsynaptic_nmda"]),
        ("S5", "glutamate recycling, ATP/neuron/s", c["glutamate_recycling"]),
        ("S11", "AMPA-type receptors, ATP/neuron/s", c["postsynaptic_ampa"]),
        ("S14", "NMDA receptors, ATP/neuron/s", c["postsynaptic_nmda"]),
        ("S18", "metabotropic receptors, ATP/neuron/s", c["postsynaptic_mglur"]),
        ("S19", "postsynaptic total, ATP/neuron/s", g["postsynaptic receptors"]),
        ("S24", "presynaptic, ATP/neuron/s", c["presynaptic"]),
        ("R27", "synaptic total, ATP/neuron/s", g["glutamate recycling"] + g["postsynaptic receptors"]
         + g["presynaptic Ca2+ and vesicles"]),
        ("Q32", "action potentials, axon, ATP/neuron/s", c["action_potential_axon"]),
        ("Q33", "action potentials, soma, ATP/neuron/s", c["action_potential_soma"]),
        ("Q34", "action potentials, dendrites, ATP/neuron/s", c["action_potential_dendrite"]),
        ("S32", "action potentials total, ATP/neuron/s", g["action potentials"]),
        ("Q41", "resting potential, neuron, ATP/neuron/s", c["resting_potential_neuron"]),
        ("Q46", "resting potential, glia, ATP/neuron/s", c["resting_potential_glia"]),
        ("S48", "resting potentials total, ATP/neuron/s", g["resting potentials"]),
        ("S52", "signaling total, ATP/neuron/s", s),
        ("S57", "signaling, ATP/m3/s", budget.per_volume(s)),
        ("S58", "signaling, mol ATP/g/min", budget.umol_per_g_min(s) * 1e-6),
        ("G60", "housekeeping, ATP/m3/s", budget.per_volume(budget.housekeeping)),
        ("G61", "housekeeping, mol ATP/g/min", budget.umol_per_g_min(budget.housekeeping) * 1e-6),
        ("S60", "total including housekeeping, ATP/m3/s", budget.per_volume(budget.total)),
        ("S61", "total including housekeeping, mol ATP/g/min", budget.umol_per_g_min(budget.total) * 1e-6),
    ]
    cells = _cells()
    rows = []
    for cell, label, mine in pairs:
        ref = float(cells[cell]["cached_value"])
        rel = abs(mine - ref) / abs(ref)
        rows.append({"reference": "spreadsheet", "quantity": label, "location": f"cell {cell}",
                     "reference_value": ref, "this_package": mine, "difference": rel, "difference_kind": "relative",
                     "tolerance": SPREADSHEET_TOLERANCE,
                     "verdict": "SUPPORTED" if rel <= SPREADSHEET_TOLERANCE else "CONTRADICTED"})
    return rows


PAPER = "Howarth, Gleeson and Attwell 2012, J Cereb Blood Flow Metab 32:1222-1232, doi:10.1038/jcbfm.2012.35"


def paper_checks() -> list[dict]:
    """Cortex numbers printed in the paper against computed values, within printed rounding."""
    base = values(reference_parameters())
    b = cortical_budget(base)
    sig = b.umol_per_g_min(b.signaling)
    groups = {k: 100 * x / b.signaling for k, x in b.groups().items()}
    hodgkin = cortical_budget(dict(base, na_overlap_factor=4.0))
    sig4 = hodgkin.umol_per_g_min(hodgkin.signaling)
    ap_cut = 100 * (1 - b.groups()["action potentials"] / hodgkin.groups()["action potentials"])
    sig_cut = 100 * (1 - b.signaling / hodgkin.signaling)
    half_total = cortical_budget(base, housekeeping_basis="total", housekeeping_fraction=0.5)
    sleep = cortical_budget(dict(base, firing_rate=3.8)).groups()
    awake = cortical_budget(dict(base, firing_rate=7.8)).groups()
    signaling_groups = [k for k in sleep if k != "resting potentials"]
    sleep_cut = 100 * (1 - sum(sleep[k] for k in signaling_groups) / sum(awake[k] for k in signaling_groups))
    spec = [
        ("signaling energy, umol ATP/g/min", 20.4, sig, 1, "Discussion: '...to 20.4 umol ATP/g/min'"),
        ("housekeeping, umol ATP/g/min", 6.79, b.umol_per_g_min(b.housekeeping), 2, "Results, Figure 1B text"),
        ("total including housekeeping, umol ATP/g/min", 27.2, b.umol_per_g_min(b.total), 1, "Results"),
        ("action potentials, % of signaling", 21, groups["action potentials"], 0, "Results"),
        ("postsynaptic receptors, % of signaling", 50, groups["postsynaptic receptors"], 0, "Results"),
        ("resting potentials, % of signaling", 20, groups["resting potentials"], 0, "Results"),
        ("glutamate recycling, % of signaling", 4, groups["glutamate recycling"], 0, "Results"),
        ("presynaptic Ca2+, % of signaling", 5, groups["presynaptic Ca2+ and vesicles"], 0, "Results"),
        ("signaling with overlap factor 4 (Attwell and Laughlin 2001), umol ATP/g/min", 30, sig4, 0, "Discussion"),
        ("reduction in action-potential cost, overlap 4 -> 1.24, %", 69, ap_cut, 0, "Discussion"),
        ("reduction in signaling energy, overlap 4 -> 1.24, %", 32, sig_cut, 0, "Discussion"),
        ("total with housekeeping 50% of total, umol ATP/g/min", 40.7, half_total.umol_per_g_min(half_total.total), 1,
         "Discussion, housekeeping range"),
        ("action-potential and synaptic energy, sleep (3.8 Hz) vs running (7.8 Hz), % less", 50, sleep_cut, None,
         "Discussion, '~50% less'; printed as approximate, tolerance 5 percentage points"),
    ]
    rows = []
    for label, printed, mine, decimals, where in spec:
        tol = 5.0 if decimals is None else 0.5 * 10 ** (-decimals)
        diff = abs(mine - printed)
        rows.append({"reference": "paper text", "quantity": label, "location": where, "reference_value": printed,
                     "this_package": mine, "difference": diff, "difference_kind": "absolute", "tolerance": tol,
                     "verdict": "SUPPORTED" if diff <= tol + 1e-12 else "CONTRADICTED"})
    return rows


def spreadsheet_findings() -> list[dict]:
    """Spreadsheet formula findings: doubled terms, unused parameters, typed constants."""
    sheet = reference_sheet()
    formulas = sheet["formulas"]
    text = " ".join(f["formula"] for f in formulas)
    referenced = set(re.findall(r"\bB(\d+)\b", text))
    unused = [p for p in sheet["parameters"] if p["cell"][1:] not in referenced]
    by = {f["cell"]: f["formula"] for f in formulas}
    findings = []
    for p in unused:
        findings.append({"finding": "parameter not used by any formula", "cell": p["cell"],
                         "detail": f"{p['label']} = {p['value']:g}; changing it changes no result"})
    if "2.0*I24" in by.get("K24", ""):
        findings.append({"finding": "vesicle costs counted twice", "cell": "K24", "detail": f"K24 = {by['K24']}"})
    if by.get("M46", "").endswith("*1.0"):
        findings.append({"finding": "glial scaling typed as a constant", "cell": "M46",
                         "detail": f"M46 = {by['M46']}; astrocytes per neuron (B35) is not referenced"})
    for cell in ("S58", "G61", "S61"):
        if "6.02e+23" in by.get(cell, "") or "6.02E+23" in by.get(cell, ""):
            findings.append({"finding": "Avogadro constant typed instead of referenced", "cell": cell,
                             "detail": f"{cell} = {by[cell]}; B30 holds 6.022e23"})
    for const in sheet["calculation_area_constants"]:
        findings.append({"finding": "input typed in the calculation area", "cell": const["cell"],
                         "detail": f"value {const['value']:g}, not listed in the parameter column"})
    for cell in ("B18", "B22", "B23"):
        if cell in by:
            findings.append({"finding": "parameter defined by a formula", "cell": cell, "detail": f"{cell} = {by[cell]}"})
    return findings
