"""Parameter records, the cortical reference set read from the extracted spreadsheet, and scenarios."""

from __future__ import annotations

import copy
import json
import math
import tomllib
from dataclasses import asdict, dataclass, replace
from importlib import resources
from pathlib import Path

import pint

UREG = pint.UnitRegistry()
SCENARIO_OPTIONS = {"glia_scaling", "housekeeping_basis", "housekeeping_fraction"}

STATUSES = {"measured", "literature estimate", "model assumption", "scenario", "physical constant"}
SPREADSHEET = "Howarth et al. 2012 supplementary spreadsheet, sheet 'Cerebral cortex - calculations'"
RODENT_CORTEX = "rodent cerebral cortex gray matter (Attwell and Laughlin 2001 model)"


@dataclass(frozen=True)
class Parameter:
    name: str
    value: float
    unit: str
    source: str
    source_location: str
    species_and_tissue: str
    experimental_or_model_condition: str
    status: str
    uncertainty_basis: str
    allowed_domain: tuple[float, float]

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"{self.name}: unknown status {self.status!r}")
        lo, hi = self.allowed_domain
        if isinstance(self.value, bool) or not math.isfinite(self.value) or not (math.isfinite(lo)
                and math.isfinite(hi) and lo <= hi):
            raise ValueError(f"{self.name}: value and allowed domain must be finite")
        UREG.Unit(self.unit)
        if not (lo <= self.value <= hi):
            raise ValueError(f"{self.name}={self.value} outside allowed domain [{lo}, {hi}]")


SHEET_PARAMETERS = {
    "glutamate_per_vesicle": ("B2", "dimensionless", "literature estimate", "molecules per vesicle", (0, 1e5)),
    "atp_per_glutamate_uptake": ("B5", "dimensionless", "literature estimate",
                                 "ATP per glutamate to reverse ion entry after uptake", (0, 10)),
    "atp_per_glutamate_metabolism": ("B6", "dimensionless", "literature estimate",
                                     "ATP per glutamate, glutamate-glutamine cycle", (0, 10)),
    "atp_per_glutamate_packaging": ("B7", "dimensionless", "literature estimate",
                                    "ATP per glutamate packaged into a vesicle", (0, 10)),
    "ampar_na_per_vesicle": ("B8", "dimensionless", "literature estimate", "", (0, 1e7)),
    "firing_rate": ("B9", "Hz", "literature estimate", "mean rate, awake rodent cortex", (0, 1000)),
    "nmda_to_ampa_charge_ratio": ("B10", "dimensionless", "literature estimate", "charge entry ratio", (0, 10)),
    "mglur_atp_per_vesicle": ("B11", "dimensionless", "literature estimate",
                              "metabotropic (G-protein) ATP per vesicle", (0, 1e6)),
    "presynaptic_ca_per_vesicle": ("B12", "dimensionless", "literature estimate",
                                   "Ca2+ ions entering per vesicle released", (0, 1e7)),
    "atp_membrane_fusion_per_vesicle": ("B13", "dimensionless", "literature estimate", "per vesicle", (0, 1e4)),
    "atp_endocytosis_control_per_vesicle": ("B14", "dimensionless", "literature estimate", "per vesicle",
                                            (0, 1e5)),
    "ap_voltage_change": ("B15", "V", "literature estimate", "axon and soma depolarization per spike", (0, 0.2)),
    "axon_length": ("B16", "m", "literature estimate", "per neuron, mouse cortex anatomy", (0, 1)),
    "axon_diameter": ("B17", "m", "literature estimate", "mean axon diameter", (0, 1e-5)),
    "membrane_capacitance": ("B18", "F/m^2", "literature estimate", "specific capacitance, 1 uF/cm2", (0, 0.1)),
    "na_overlap_factor": ("B19", "dimensionless", "measured",
                          "Na+ entry relative to the capacitive minimum", (1, 10)),
    "elementary_charge": ("B20", "C", "physical constant", "rounded in the source to 1.6e-19 C", (1e-19, 2e-19)),
    "soma_diameter": ("B21", "m", "literature estimate", "spherical soma", (0, 1e-3)),
    "dendrite_diameter": ("B22", "m", "model assumption",
                          "set in the spreadsheet as 3 x axon diameter (formula in B22)", (0, 1e-4)),
    "dendrite_length": ("B23", "m", "model assumption",
                        "set in the spreadsheet as axon length / 9 (formula in B23)", (0, 1)),
    "dendrite_voltage_change": ("B24", "V", "literature estimate", "simulated dendritic depolarization per spike",
                                (0, 0.2)),
    "neuron_input_resistance": ("B25", "ohm", "literature estimate", "whole-cell input resistance", (1e6, 1e11)),
    "neuron_resting_potential": ("B26", "V", "literature estimate", "", (-0.12, 0.0)),
    "e_k": ("B27", "V", "literature estimate", "K+ Nernst potential", (-0.2, 0.0)),
    "e_na": ("B28", "V", "literature estimate", "Na+ Nernst potential", (0.0, 0.2)),
    "faraday": ("B29", "C/mol", "physical constant", "the source uses 96495; CODATA is 96485.33", (9.6e4, 9.7e4)),
    "avogadro": ("B30", "1/mol", "physical constant", "", (6.0e23, 6.1e23)),
    "astrocyte_input_resistance": ("B31", "ohm", "literature estimate", "", (1e6, 1e11)),
    "astrocyte_resting_potential": ("B32", "V", "literature estimate", "", (-0.12, 0.0)),
    "boutons_per_neuron": ("B33", "dimensionless", "literature estimate", "presynaptic boutons per neuron",
                           (0, 1e6)),
    "release_probability": ("B34", "dimensionless", "literature estimate", "per bouton per action potential",
                            (0, 1)),
    "astrocytes_per_neuron": ("B35", "dimensionless", "literature estimate", "", (0, 100)),
    "neuron_density": ("B36", "1/m^3", "literature estimate", "neurons per cubic meter of gray matter",
                       (0, 1e16)),
}

FORMULA_CONSTANTS = {
    "na_per_atp": (3.0, "dimensionless", "model assumption", "Na+/K+ pump: 3 Na+ out per ATP; divisor in M11, M14, "
                   "N32-N34 and G24/3", (1, 10)),
    "na_per_ca_extruded": (3.0, "dimensionless", "model assumption", "Na+/Ca2+ exchange: 3 Na+ in per Ca2+ out; "
                           "J14 and G24", (1, 10)),
    "charges_per_ca": (2.0, "dimensionless", "physical constant", "Ca2+ valence; divisor in I14", (2, 2)),
    "nmda_ca_charge_fraction": (0.1, "dimensionless", "model assumption", "fraction of NMDA receptor charge "
                                "carried by Ca2+; constant typed in cell F14, outside the parameter column", (0, 1)),
    "presynaptic_vesicle_cost_multiplier": (2.0, "dimensionless", "model assumption",
                                            "formula K24 = (G24/3 + 2*I24 + 2*J24)*F27 counts fusion and endocytosis "
                                            "ATP twice per vesicle; the spreadsheet gives no reason", (0, 10)),
    "glia_per_neuron_in_spreadsheet": (1.0, "dimensionless", "model assumption",
                                       "formula M46 = K46*1.0 scales glial resting-potential cost by a typed 1.0, so "
                                       "parameter B35 (astrocytes per neuron) has no effect in the spreadsheet",
                                       (0, 100)),
    "tissue_density": (1e6, "g/m^3", "model assumption", "1 g/cm3, the divisor 1000000 in S58, G61 and S61",
                       (5e5, 2e6)),
    "avogadro_in_unit_conversion": (6.02e23, "1/mol", "physical constant",
                                    "S58, G61 and S61 divide by a typed 6.02e23 instead of B30 (6.022e23)",
                                    (6.0e23, 6.1e23)),
    "housekeeping_fraction_of_signaling": (1.0 / 3.0, "dimensionless", "model assumption",
                                           "G60 = S57/3: housekeeping equal to 1/3 of signaling energy, i.e. 25% "
                                           "of the total", (0, 10)),
}


def reference_sheet() -> dict:
    path = resources.files("neural_budget") / "reference_data" / "howarth2012_cortex.json"
    return json.loads(path.read_text())


def reference_parameters() -> dict[str, Parameter]:
    """Reference parameters: spreadsheet values plus its formula constants."""
    sheet = reference_sheet()
    by_cell = {p["cell"]: p for p in sheet["parameters"]}
    out = {}
    for name, (cell, unit, status, condition, domain) in SHEET_PARAMETERS.items():
        row = by_cell[cell]
        cited = row["reference"] or "not cited in the spreadsheet"
        out[name] = Parameter(
            name=name, value=float(row["value"]), unit=unit, source=cited,
            source_location=f"{SPREADSHEET}, cell {cell} ({row['label']})",
            species_and_tissue=RODENT_CORTEX,
            experimental_or_model_condition="; ".join(x for x in (condition, row["note"]) if x),
            status=status, uncertainty_basis="none given in the source", allowed_domain=domain)
    for name, (value, unit, status, condition, domain) in FORMULA_CONSTANTS.items():
        out[name] = Parameter(
            name=name, value=float(value), unit=unit, source="Howarth et al. 2012 spreadsheet formulas",
            source_location=SPREADSHEET, species_and_tissue=RODENT_CORTEX,
            experimental_or_model_condition=condition, status=status,
            uncertainty_basis="none given in the source", allowed_domain=domain)
    return out


def values(params: dict[str, Parameter]) -> dict[str, float]:
    """Parameter magnitudes in the model's documented units, converted rather than stripped."""
    units = {k: spec[1] for k, spec in SHEET_PARAMETERS.items()} | {
        k: spec[1] for k, spec in FORMULA_CONSTANTS.items()}
    return {k: float(UREG.Quantity(p.value, p.unit).to(units[k]).magnitude) for k, p in params.items()}


def validate_values(v: dict) -> dict[str, float]:
    """Normalize quantities to model units and enforce the same domains used by scenarios."""
    specs = {k: (s[1], s[-1]) for k, s in SHEET_PARAMETERS.items()} | {
        k: (s[1], s[-1]) for k, s in FORMULA_CONSTANTS.items()}
    missing, unknown = sorted(specs.keys() - v.keys()), sorted(v.keys() - specs.keys())
    if missing or unknown:
        raise ValueError(f"parameter names differ from the model: missing {missing}, unknown {unknown}")
    out = {}
    for name, (unit, (lo, hi)) in specs.items():
        raw = v[name]
        if isinstance(raw, bool):
            raise ValueError(f"{name}: boolean values are not parameter magnitudes")
        x = float(raw.to(unit).magnitude if isinstance(raw, pint.Quantity) else raw)
        if not math.isfinite(x) or not lo <= x <= hi:
            raise ValueError(f"{name}={x} outside finite allowed domain [{lo}, {hi}]")
        out[name] = x
    return out


def apply_scenario(params: dict[str, Parameter], changes: dict[str, float], label: str) -> dict[str, Parameter]:
    """Copy of ``params`` with ``changes`` applied and marked as scenario values."""
    out = copy.copy(params)
    for name, value in changes.items():
        if name not in out:
            raise KeyError(f"unknown parameter {name!r}")
        out[name] = replace(out[name], value=float(value), status="scenario",
                            source=f"scenario '{label}'", uncertainty_basis="scenario value, not an estimate")
    return out


def load_scenario(path) -> dict:
    """Read a scenario file: [scenario] name, description, source, options; [changes] parameter = value."""
    with open(path, "rb") as fh:
        doc = tomllib.load(fh)
    unknown = set(doc) - {"scenario", "changes", "options"}
    if unknown:
        raise ValueError(f"{path}: unknown scenario tables {sorted(unknown)}")
    sc = doc.get("scenario", {})
    if "name" not in sc or "changes" not in doc:
        raise ValueError(f"{path}: needs [scenario] name and a [changes] table")
    if not isinstance(sc["name"], str) or not sc["name"].strip():
        raise ValueError(f"{path}: scenario name must be nonempty text")
    unknown_options = set(doc.get("options", {})) - SCENARIO_OPTIONS
    if unknown_options:
        raise ValueError(f"{path}: unknown scenario options {sorted(unknown_options)}")
    unknown_parameters = set(doc["changes"]) - (SHEET_PARAMETERS.keys() | FORMULA_CONSTANTS.keys())
    if unknown_parameters:
        raise ValueError(f"{path}: unknown scenario parameters {sorted(unknown_parameters)}")
    return {"name": sc["name"], "description": sc.get("description", ""), "source": sc.get("source", ""),
            "options": dict(doc.get("options", {})),
            "changes": {k: float(v) for k, v in doc["changes"].items()}, "path": str(Path(path).name)}


def parameter_table(params: dict[str, Parameter]) -> list[dict]:
    rows = []
    for p in params.values():
        d = asdict(p)
        d["allowed_domain"] = f"[{p.allowed_domain[0]:g}, {p.allowed_domain[1]:g}]"
        rows.append(d)
    return rows
