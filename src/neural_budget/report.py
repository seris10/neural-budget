"""Static HTML report: reproduction, components, parameters, spreadsheet findings, scenarios, sensitivity."""

from __future__ import annotations

import html
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .model import GROUPS  # noqa: E402

plt.rcParams.update({
    "svg.fonttype": "none", "svg.hashsalt": "neural-budget", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "legend.frameon": False,
})
GROUP_COLORS = {"glutamate recycling": "#b7791f", "postsynaptic receptors": "#2b6cb0",
                "presynaptic Ca2+ and vesicles": "#6b46c1", "action potentials": "#c05621",
                "resting potentials": "#2f855a"}


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def _esc(x):
    return html.escape(str(x))


REPORT_LABELS = {'reference (Howarth et al. 2012)': 'Reference (Howarth et al., 2012)',
 'active running, 7.8 Hz': 'Running: 7.8 Hz',
 'overlap factor 4 (Attwell and Laughlin 2001)': 'Na⁺ overlap factor = 4',
 'housekeeping 50% of total energy': 'Housekeeping = 50% of total ATP demand',
 'sleep, 3.8 Hz': 'Sleep: 3.8 Hz',
 'Astrocyte-to-neuron ratio = 2': 'Astrocyte-to-neuron ratio = 2',
 'vesicle fusion and endocytosis ATP counted once': 'Vesicle fusion/endocytosis multiplier = 1',
 'glutamate recycling': 'Glutamate recycling',
 'postsynaptic receptors': 'Postsynaptic receptors',
 'presynaptic Ca2+ and vesicles': 'Presynaptic Ca²⁺ and vesicle cycling',
 'action potentials': 'Action potentials',
 'resting potentials': 'Resting potentials',
 'glutamate_recycling': 'Glutamate recycling',
 'postsynaptic_ampa': 'Postsynaptic AMPA receptors',
 'postsynaptic_nmda': 'Postsynaptic NMDA receptors',
 'postsynaptic_mglur': 'Postsynaptic metabotropic receptors',
 'presynaptic': 'Presynaptic Ca²⁺ and vesicle cycling',
 'action_potential_axon': 'Axonal action potentials',
 'action_potential_soma': 'Somatic action potentials',
 'action_potential_dendrite': 'Dendritic action potentials',
 'resting_potential_neuron': 'Neuronal resting potential',
 'resting_potential_glia': 'Astrocytic resting potential',
 'signaling total': 'Signaling total',
 'housekeeping': 'Housekeeping allowance',
 'total including housekeeping': 'Total ATP demand'}


def _fmt(v, digits=4):
    if v is None:
        return ""
    if isinstance(v, str):
        labels = {"glial scaling typed as a constant": "Fixed glial-scaling coefficient",
                  "Avogadro constant typed instead of referenced": "Embedded Avogadro constant",
                  "input typed in the calculation area": "Calculation-cell input",
                  "parameter defined by a formula": "Formula-derived parameter",
                  "parameter not used by any formula": "Unreferenced parameter",
                  "vesicle costs counted twice": "Repeated vesicle-cost terms"}
        v = REPORT_LABELS.get(v, labels.get(v, v))
        for source, display in (("umol ATP/g/min", "µmol ATP g⁻¹ min⁻¹"),
                                ("mol ATP/g/min", "mol ATP g⁻¹ min⁻¹"),
                                ("ATP/neuron/s", "ATP molecules neuron⁻¹ s⁻¹"),
                                ("ATP/m3/s", "ATP molecules m⁻³ s⁻¹"),
                                ("signaling energy", "signaling ATP demand"),
                                ("per-gram energy", "tissue-normalized ATP demand"),
                                ("per-neuron energy", "cellular ATP demand")):
            v = v.replace(source, display)
        return _esc(REPORT_LABELS.get(v, v).replace("Ca2+", "Ca²⁺").replace("Na+", "Na⁺"))
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v)
    if v == 0:
        return "0"
    if not np.isfinite(v):
        return "∞" if v > 0 else "−∞"
    if abs(v) >= 1e5 or abs(v) < 1e-3:
        coefficient, exponent = f"{v:.{digits - 1}e}".split("e")
        power = str(int(exponent)).replace("-", "−")
        return f"{coefficient} × 10<sup>{power}</sup>"
    return f"{v:.{digits}g}"


def _table(df, cols=None, headers=None, digits=4):
    if df is None or len(df) == 0:
        return "<p class='muted'>No rows.</p>"
    cols = cols or list(df.columns)
    headers = headers or cols
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for _, r in df[cols].iterrows():
        cells = []
        for c in cols:
            v = r[c]
            num = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
            cls = "num" if num else ""
            if isinstance(v, str) and v in ("SUPPORTED", "CONTRADICTED", "NOT TESTED"):
                cls = {"SUPPORTED": "ok", "CONTRADICTED": "bad", "NOT TESTED": "warn"}[v]
            display = {"SUPPORTED": "Within tolerance", "CONTRADICTED": "Outside tolerance", "NOT TESTED": "Not evaluated"}.get(v, v) if isinstance(v, str) else v
            cells.append(f"<td class='{cls}'>{_fmt(display, digits)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _budget_figure(cb):
    cases = list(dict.fromkeys(cb.case))
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(cases) + 1.2))
    sig = cb[cb.component.isin([c for cs in GROUPS.values() for c in cs])]
    for i, case in enumerate(cases[::-1]):
        left = 0.0
        d = sig[sig.case == case]
        for g, color in GROUP_COLORS.items():
            w = float(d[d.group == g].umol_atp_per_g_min.sum())
            ax.barh(i, w, left=left, color=color, height=0.62, label=REPORT_LABELS.get(g, g) if i == 0 else None)
            left += w
        ax.text(left + 0.3, i, f"{left:.1f}", va="center", fontsize=8)
    ax.set_yticks(range(len(cases)), [REPORT_LABELS.get(case, case) for case in cases[::-1]])
    ax.set_xlabel("Signaling ATP demand (µmol ATP g⁻¹ min⁻¹)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.45, -0.18))
    ax.set_xlim(0, max(1.0, ax.get_xlim()[1] * 1.06))
    fig.tight_layout()
    return _svg(fig)


DISPLAY_PARAMETERS = {
    "release_probability": ("Vesicle release probability", 1.0),
    "firing_rate": ("Firing rate (Hz)", 1.0),
    "ampar_na_per_vesicle": ("AMPA Na⁺ entry (10³ ions vesicle⁻¹)", 1e-3),
    "na_overlap_factor": ("Na⁺ overlap factor", 1.0),
    "boutons_per_neuron": ("Boutons (10³ neuron⁻¹)", 1e-3),
    "nmda_to_ampa_charge_ratio": ("NMDA/AMPA charge ratio", 1.0),
    "neuron_input_resistance": ("Neuronal input resistance (MΩ)", 1e-6),
    "axon_length": ("Axon length (mm)", 1e3),
    "astrocyte_input_resistance": ("Astrocytic input resistance (MΩ)", 1e-6),
    "neuron_resting_potential": ("Neuronal resting potential (mV)", 1e3),
    "glutamate_per_vesicle": ("Glutamate content (10³ molecules vesicle⁻¹)", 1e-3),
    "presynaptic_ca_per_vesicle": ("Presynaptic Ca²⁺ entry (10³ ions vesicle⁻¹)", 1e-3),
    "mglur_atp_per_vesicle": ("mGluR cost (10³ ATP molecules vesicle⁻¹)", 1e-3),
    "presynaptic_vesicle_cost_multiplier": ("Vesicle-cycling cost multiplier", 1.0),
}


def _tornado_figure(tor, ref_value):
    t = tor.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 0.36 * len(t) + 1.0))
    for i, r in enumerate(t.itertuples()):
        lo, hi = r.signaling_at_low, r.signaling_at_high
        color = "#2b6cb0" if r.basis == "sourced" else "#a0aec0"
        ax.barh(i, hi - ref_value, left=ref_value, color=color, height=0.6)
        ax.barh(i, lo - ref_value, left=ref_value, color=color, height=0.6, alpha=0.55)
        scale = DISPLAY_PARAMETERS[r.parameter][1]
        bounds = f"[{r.low_value * scale:g}, {r.high_value * scale:g}]".replace("-", "−")
        ax.text(max(lo, hi) + 0.15, i, bounds, va="center", fontsize=7)
    ax.axvline(ref_value, color="#1a202c", lw=0.8)
    ax.set_yticks(range(len(t)), [DISPLAY_PARAMETERS[name][0] for name in t.parameter])
    ax.set_xlabel("Signaling ATP demand (µmol ATP g⁻¹ min⁻¹)")
    ax.legend(handles=[Patch(facecolor="#2b6cb0", label="Literature-derived bounds"),
                       Patch(facecolor="#a0aec0", label="Analysis-defined bounds")],
              loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
    fig.tight_layout()
    return _svg(fig)


CSS = """
:root{--fg:#1a202c;--muted:#4a5568;--line:#e2e8f0;--bg:#ffffff;--panel:#f7fafc}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",
Helvetica,Arial,sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:28px 0 8px;padding-top:10px;border-top:1px solid var(--line)}
h3{font-size:14px;margin:16px 0 6px}
p{margin:6px 0 10px;max-width:92ch}
.muted{color:var(--muted)}
.lede{font-size:15px}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:12px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px 12px}
.card h3{margin:0 0 4px;font-size:13px}
.card p{margin:0;font-size:13px}
.tw{overflow-x:auto;margin:8px 0 14px}
table{border-collapse:collapse;font-size:12px;width:100%}
th,td{padding:3px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:var(--panel);font-weight:600;vertical-align:bottom}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.ok{color:#276749;font-weight:600}td.bad{color:#c53030;font-weight:600}td.warn{color:#975a16;font-weight:600}
figure{margin:10px 0 16px}
figure svg{width:100%;height:auto}
figcaption{font-size:12px;color:var(--muted);max-width:110ch}
ul{margin:4px 0 10px;padding-left:20px}
code{font-size:12px;background:var(--panel);padding:0 3px;border-radius:3px}
@media (max-width:760px){.grid3{grid-template-columns:1fr}}
"""


def _scientific(value, digits=2):
    if value == 0:
        return "0"
    coefficient, exponent = f"{value:.{digits - 1}e}".split("e")
    power = str(int(exponent)).replace("-", "−")
    return f"{coefficient} × 10<sup>{power}</sup>"


def render(r: dict, version: str) -> str:
    from .reference import SPREADSHEET_TOLERANCE

    ref = r["reference"]
    rr = r["reference_reproduction"]
    sheet = rr[rr.reference == "spreadsheet"]
    paper = rr[rr.reference == "paper text"]
    cb = r["component_budget"]
    sr = r["scenario_results"]
    tor = r["tornado"]
    env = r["factorial_envelope"]
    sig = ref.umol_per_g_min(ref.signaling)
    unit = "µmol ATP g<sup>−1</sup> min<sup>−1</sup>"
    cell_unit = "ATP molecules neuron<sup>−1</sup> s<sup>−1</sup>"
    p = ["<h1>Cortical ATP budget: numerical reproduction and sensitivity analysis</h1>",
         f"<p class='muted'>neural_budget {version} · Howarth, Gleeson and Attwell (2012), "
         "<i>Journal of Cerebral Blood Flow &amp; Metabolism</i> 32:1222–1232. "
         "<a href='https://doi.org/10.1038/jcbfm.2012.35'>doi:10.1038/jcbfm.2012.35</a></p>",
         "<p class='lede'>The cerebral-cortex model of Howarth et al. (2012) estimates ATP consumption from "
         "synaptic transmission, action potentials and maintenance of resting potentials. This analysis "
         "reimplements the supplementary workbook, compares the calculated values with the published reference, "
         "and quantifies the response to specified parameter perturbations. The estimates are conditional on "
         "the model structure and reference parameters.</p>"]
    n_sheet_ok = int((sheet.verdict == "SUPPORTED").sum())
    n_paper_ok = int((paper.verdict == "SUPPORTED").sum())
    p.append("<div class='grid3'>")
    p.append(f"<div class='card'><h3>Numerical agreement</h3><p>{n_sheet_ok}/{len(sheet)} workbook comparisons "
             f"satisfy the relative tolerance of {_scientific(SPREADSHEET_TOLERANCE)}; the maximum relative error "
             f"is {_scientific(float(sheet.difference.max()))}. {n_paper_ok}/{len(paper)} paper comparisons "
             "satisfy their specified tolerances (section 2).</p></div>")
    p.append(f"<div class='card'><h3>Reference ATP consumption</h3><p>Signaling: {sig:.2f} {unit}. "
             f"Signaling plus housekeeping: {ref.umol_per_g_min(ref.total):.2f} {unit}. The corresponding "
             f"cellular signaling rate is {_scientific(ref.signaling, 3)} {cell_unit}.</p></div>")
    p.append(f"<div class='card'><h3>Factorial parameter bounds</h3><p>The eight combinations of firing rate, "
             f"Na<sup>+</sup> overlap factor and release probability yield {env['min_umol_g_min']:.1f}–"
             f"{env['max_umol_g_min']:.1f} {unit}. These extrema describe the specified parameter combinations; "
             "no sampling distribution or coverage probability is assigned.</p></div></div>")

    p.append("<h2>1. Cellular ATP-demand model</h2>")
    p.append(f"<p>Signaling demand is N<sub>sig</sub> = Σ<sub>j</sub> λ<sub>j</sub>c<sub>j</sub> + "
             f"N<sub>rest</sub>, expressed in {cell_unit}. For each activity-dependent process j, "
             "λ<sub>j</sub> is the event rate and c<sub>j</sub> is ATP consumption per event. Synaptic demand "
             "is proportional to the vesicle-release rate bpf, where b is boutons per neuron, p is release "
             f"probability and f is firing rate; bpf = {ref.vesicles_per_s:.0f} vesicles neuron<sup>−1</sup> "
             "s<sup>−1</sup> in the reference case. Action-potential demand is proportional to f. Neuronal "
             "and astrocytic resting-potential demand follows equation 4 of Attwell and Laughlin (2001). "
             "The reference housekeeping allowance is N<sub>sig</sub>/3, equivalent to 25% of total ATP demand.</p>")
    p.append(f"<figure>{_budget_figure(cb)}<figcaption><b>Figure 1. Process-specific signaling ATP demand.</b> "
             "Stacked bars show the reference calculation and the configured parameter scenarios. Each segment "
             "represents one signaling process. Housekeeping is excluded from the bars and reported separately "
             "in the scenario table.</figcaption></figure>")
    refrows = cb[cb.case == cb.case.iloc[0]]
    p.append(_table(refrows, ["component", "group", "atp_per_neuron_s", "umol_atp_per_g_min", "percent_of_signaling"],
                    ["Component", "Process", "ATP molecules neuron⁻¹ s⁻¹", "µmol ATP g⁻¹ min⁻¹", "Relative to signaling demand (%)"]))

    p.append("<h2>2. Reference-case numerical validation</h2>")
    p.append("<h3>Supplementary-workbook comparisons</h3><p>Computed values are compared with 25 cached result "
             "cells extracted from the archived supplementary workbook. The extraction record contains the cell "
             "formulas and cached values; the workbook checksum is recorded in <code>data_manifest.json</code>. "
             "Relative error is |computed − reference|/|reference|.</p>")
    p.append(_table(sheet, ["location", "quantity", "reference_value", "this_package", "difference", "verdict"],
                    ["Cell", "Quantity", "Workbook value", "Computed value", "Relative error", "Comparison status"], digits=6))
    p.append("<h3>Comparisons with reported paper values</h3><p>Twelve comparisons use an absolute tolerance "
             "equal to half the resolution of the reported value: 0.5 × 10<sup>−d</sup> for d decimal places. "
             "The approximate 50% sleep-versus-running reduction uses a separately specified tolerance of "
             "5 percentage points. The tolerance for each comparison is listed below.</p>")
    p.append(_table(paper, ["quantity", "reference_value", "this_package", "tolerance", "verdict"],
                    ["Quantity", "Reported value", "Computed value", "Absolute tolerance", "Comparison status"]))

    p.append("<h2>3. Supplementary-workbook formula audit</h2>")
    p.append("<p>The extracted formulas identify unused parameters, repeated vesicle-cost terms and constants "
             "embedded in calculation cells. The reference implementation retains these expressions. Separate "
             "scenarios quantify the effect of each alternative expression.</p>")
    p.append(_table(r["spreadsheet_findings"], ["finding", "cell", "detail"], ["Formula property", "Cell", "Expression or dependency"]))
    impact = sr[sr.file.isin(["vesicle_costs_counted_once.toml", "two_astrocytes_per_neuron.toml", "codata_constants.toml"])]
    if len(impact):
        p.append("<h3>Effects of alternative formula conventions</h3>")
        p.append(_table(impact, ["scenario", "changes", "signaling_umol_g_min", "signaling_change_percent"],
                        ["Scenario", "Parameter or expression change", "Signaling (µmol ATP g⁻¹ min⁻¹)", "Relative change (%)"]))
        p.append("<p>The workbook glial term uses a fixed multiplier of 1.0; changing its astrocytes-per-neuron "
                 "input therefore leaves the cached calculation unchanged. The corresponding scenario replaces "
                 "this multiplier with the specified glial ratio. Counting vesicle fusion and endocytosis once, "
                 "or replacing rounded physical constants, changes reference signaling demand by less than 1%.</p>")

    p.append("<h2>4. Parameter perturbation scenarios</h2>")
    p.append("<p>Scenario definitions in <code>src/neural_budget/scenarios/</code> specify parameter changes, "
             "model options and sources. Each calculation starts from the same reference parameter set.</p>")
    p.append(_table(sr, ["scenario", "changes", "source", "signaling_umol_g_min", "total_umol_g_min", "signaling_change_percent"],
                    ["Scenario", "Changes", "Source", "Signaling (µmol ATP g⁻¹ min⁻¹)", "Total (µmol ATP g⁻¹ min⁻¹)", "Signaling change (%)"]))
    grp = [c for c in sr.columns if c.endswith(" change %")]
    p.append("<h3>Relative changes in process-specific ATP demand</h3>")
    p.append(_table(sr, ["scenario"] + grp, ["Scenario"] + [g.replace(" change %", " (%)") for g in grp], digits=3))

    p.append("<h2>5. Local and factorial sensitivity analysis</h2>")
    p.append(f"<figure>{_tornado_figure(tor, sig)}<figcaption><b>Figure 2. One-at-a-time parameter sensitivity.</b> "
             "Each parameter is evaluated at its specified lower and upper bounds while all other parameters "
             "remain at their reference values. The vertical line marks reference signaling demand. Dark and "
             "light bars indicate the upper and lower parameter bounds, respectively; annotations give those "
             "bounds. Blue intervals are literature-derived; gray intervals were specified for this sensitivity "
             "analysis. The intervals have no assigned probability distributions.</figcaption></figure>")
    display_tor = tor.copy()
    for idx, row in tor.iterrows():
        label, scale = DISPLAY_PARAMETERS[row.parameter]
        display_tor.loc[idx, "parameter"] = label
        display_tor.loc[idx, "low_value"] = row.low_value * scale
        display_tor.loc[idx, "high_value"] = row.high_value * scale
    display_tor["basis"] = display_tor.basis.replace({"sourced": "Literature-derived", "authored": "Analysis-defined"})
    p.append(_table(display_tor, ["parameter", "basis", "low_value", "high_value", "signaling_at_low", "signaling_at_high", "swing_umol_g_min"],
                    ["Parameter", "Interval source", "Lower bound", "Upper bound", "Demand at lower bound", "Demand at upper bound", "Demand range"]))
    p.append("<h3>Two-level factorial effects</h3>")
    p.append("<p>The 2³ design varies firing rate, Na<sup>+</sup> overlap factor and release probability at "
             "coded levels −1 and +1. Effects are expressed in µmol ATP g<sup>−1</sup> min<sup>−1</sup>. "
             "Interaction terms arise from the products of firing rate with overlap factor in action-potential "
             "demand and with release probability in synaptic demand. Variance-based indices are not estimated "
             "because joint parameter distributions have not been specified.</p>")
    p.append(_table(r["factorial_effects"], ["term", "effect_umol_g_min"], ["Term", "Effect (µmol ATP g⁻¹ min⁻¹)"]))

    p.append("<h2>6. Boundary-condition tests</h2>")
    p.append(_table(r["limiting_cases"], ["case", "expectation", "observed", "passed"], ["Condition", "Expected behavior", "Calculated behavior", "Passed"]))
    p.append("<h2>7. Model exclusions</h2>")
    p.append(_table(r["excluded"], headers=["Excluded process or domain", "Model restriction"]))
    p.append("<h2>8. Parameter provenance</h2>")
    p.append("<p>Parameter values and citations are transcribed from the supplementary workbook. Records identify "
             "the measurement, literature estimate, model assumption or physical constant associated with each "
             "value. The extracted records contain no parameter-specific standard errors or probability distributions.</p>")
    p.append(_table(r["parameter_sources"], ["name", "value", "unit", "status", "source", "experimental_or_model_condition"],
                    ["Parameter identifier", "Value", "Unit", "Classification", "Source", "Experimental or model condition"]))
    p.append("<p class='muted'>Documentation: <code>docs/methods.md</code>, <code>docs/limitations.md</code>, "
             "<code>docs/validation.md</code> and <code>docs/development.md</code>.</p>")
    body = "\n".join(p)
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Cortical ATP Budget</title><style>{CSS}</style></head><body><main>{body}</main></body></html>\n")
