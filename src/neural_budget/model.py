"""ATP budget per cortical neuron and its glia: event rates times ATP per event, units checked with Pint."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .params import UREG, validate_values

Q = UREG.Quantity

SIGNALING_COMPONENTS = (
    "glutamate_recycling",
    "postsynaptic_ampa",
    "postsynaptic_nmda",
    "postsynaptic_mglur",
    "presynaptic",
    "action_potential_axon",
    "action_potential_soma",
    "action_potential_dendrite",
    "resting_potential_neuron",
    "resting_potential_glia",
)
GROUPS = {
    "glutamate recycling": ("glutamate_recycling",),
    "postsynaptic receptors": ("postsynaptic_ampa", "postsynaptic_nmda", "postsynaptic_mglur"),
    "presynaptic Ca2+ and vesicles": ("presynaptic",),
    "action potentials": ("action_potential_axon", "action_potential_soma", "action_potential_dendrite"),
    "resting potentials": ("resting_potential_neuron", "resting_potential_glia"),
}
ACTIVITY_DEPENDENT = GROUPS["glutamate recycling"] + GROUPS["postsynaptic receptors"] + \
    GROUPS["presynaptic Ca2+ and vesicles"] + GROUPS["action potentials"]


def _q(v: dict, name: str, unit: str):
    return Q(v[name], unit)


def resting_atp_rate(r_in, v_rest, e_na, e_k, faraday):
    """Resting ATP use (mol/s): (V_Na - V)(V - V_K) / (F R (2 V_Na + V - 3 V_K))."""
    r_in, v_rest, e_na, e_k, faraday = (r_in.to("ohm"), v_rest.to("V"), e_na.to("V"), e_k.to("V"),
                                      faraday.to("C/mol"))
    vals = [float(q.magnitude) for q in (r_in, v_rest, e_na, e_k, faraday)]
    if not all(np.isfinite(vals)) or r_in.magnitude <= 0 or faraday.magnitude <= 0:
        raise ValueError("resting-potential inputs must be finite; resistance and Faraday constant must be positive")
    if not e_k.magnitude <= v_rest.magnitude <= e_na.magnitude or e_k.magnitude == e_na.magnitude:
        raise ValueError("resting potential must lie between distinct K+ and Na+ reversal potentials")
    num = (e_na - v_rest) * (v_rest - e_k)
    den = faraday * r_in * (2 * e_na + v_rest - 3 * e_k)
    return (num / den).to("mol/s")


def housekeeping_rate(signaling: float, basis: str, fraction: float) -> float:
    """Housekeeping rate: f S for ``basis="signaling"``, f S / (1 - f) for ``basis="total"``."""
    if not (np.isfinite(signaling) and signaling >= 0 and np.isfinite(fraction) and fraction >= 0):
        raise ValueError("signaling and housekeeping fraction must be finite and nonnegative")
    if basis == "signaling":
        return fraction * signaling
    if basis == "total":
        if not 0 <= fraction < 1:
            raise ValueError("a fraction of the total must lie in [0, 1)")
        return fraction * signaling / (1.0 - fraction)
    raise ValueError(f"unknown housekeeping basis {basis!r}")


@dataclass
class Budget:
    """ATP molecules per neuron per second by component, the vesicle rate, and the conversions used."""

    components: dict[str, float]
    vesicles_per_s: float
    per_vesicle_costs: dict[str, float]
    neuron_density_per_m3: float
    avogadro_for_conversion: float
    tissue_density_g_per_m3: float
    housekeeping_basis: str
    housekeeping_fraction: float

    @property
    def signaling(self) -> float:
        return float(sum(self.components.values()))

    @property
    def housekeeping(self) -> float:
        return housekeeping_rate(self.signaling, self.housekeeping_basis, self.housekeeping_fraction)

    @property
    def total(self) -> float:
        return self.signaling + self.housekeeping

    def groups(self) -> dict[str, float]:
        return {g: float(sum(self.components[c] for c in cs)) for g, cs in GROUPS.items()}

    def per_volume(self, atp_per_neuron_s: float) -> float:
        """ATP molecules per m3 of tissue per second."""
        return atp_per_neuron_s * self.neuron_density_per_m3

    def umol_per_g_min(self, atp_per_neuron_s: float) -> float:
        """Micromoles of ATP per gram of tissue per minute."""
        mol_per_g_min = self.per_volume(atp_per_neuron_s) / self.tissue_density_g_per_m3 * 60.0 \
            / self.avogadro_for_conversion
        return mol_per_g_min * 1e6


def cortical_budget(v: dict[str, float], *, glia_scaling="spreadsheet", housekeeping_basis="signaling",
                    housekeeping_fraction=None) -> Budget:
    """Budget from a parameter-value dict; glial cost scaled by the spreadsheet's 1.0 or astrocytes_per_neuron."""
    v = validate_values(v)
    hk = v["housekeeping_fraction_of_signaling"] if housekeeping_fraction is None else housekeeping_fraction
    housekeeping_rate(1.0, housekeeping_basis, hk)
    na_per_atp = v["na_per_atp"]
    rate = _q(v, "firing_rate", "Hz")
    ves_rate = (v["boutons_per_neuron"] * v["release_probability"] * rate).to("1/s")

    c = {}
    c["glutamate_recycling"] = v["glutamate_per_vesicle"] * (
        v["atp_per_glutamate_uptake"] + v["atp_per_glutamate_metabolism"] + v["atp_per_glutamate_packaging"])
    c["postsynaptic_ampa"] = v["ampar_na_per_vesicle"] / na_per_atp
    nmda_charge = v["ampar_na_per_vesicle"] * v["nmda_to_ampa_charge_ratio"]
    nmda_na = nmda_charge * (1.0 - v["nmda_ca_charge_fraction"])
    nmda_ca = nmda_charge * v["nmda_ca_charge_fraction"] / v["charges_per_ca"]
    c["postsynaptic_nmda"] = (nmda_na + v["na_per_ca_extruded"] * nmda_ca) / na_per_atp
    c["postsynaptic_mglur"] = v["mglur_atp_per_vesicle"]
    c["presynaptic"] = (v["presynaptic_ca_per_vesicle"] * v["na_per_ca_extruded"] / na_per_atp
                        + v["presynaptic_vesicle_cost_multiplier"]
                        * (v["atp_membrane_fusion_per_vesicle"] + v["atp_endocytosis_control_per_vesicle"]))
    comps = {k: float((Q(cost, "dimensionless") * ves_rate).to("1/s").magnitude) for k, cost in c.items()}

    cm = _q(v, "membrane_capacitance", "F/m^2")
    e = _q(v, "elementary_charge", "C")
    compartments = {
        "action_potential_axon": (np.pi * _q(v, "axon_length", "m") * _q(v, "axon_diameter", "m"),
                                  _q(v, "ap_voltage_change", "V")),
        "action_potential_soma": (np.pi * _q(v, "soma_diameter", "m") ** 2, _q(v, "ap_voltage_change", "V")),
        "action_potential_dendrite": (np.pi * _q(v, "dendrite_length", "m") * _q(v, "dendrite_diameter", "m"),
                                      _q(v, "dendrite_voltage_change", "V")),
    }
    for name, (area, dv) in compartments.items():
        na_ions = (cm * area * dv / e).to("dimensionless") * v["na_overlap_factor"]
        comps[name] = float((na_ions / na_per_atp * rate).to("1/s").magnitude)

    na = _q(v, "avogadro", "1/mol")
    f = _q(v, "faraday", "C/mol")
    neuron = resting_atp_rate(_q(v, "neuron_input_resistance", "ohm"), _q(v, "neuron_resting_potential", "V"),
                              _q(v, "e_na", "V"), _q(v, "e_k", "V"), f)
    glia_one = resting_atp_rate(_q(v, "astrocyte_input_resistance", "ohm"), _q(v, "astrocyte_resting_potential", "V"),
                                _q(v, "e_na", "V"), _q(v, "e_k", "V"), f)
    if glia_scaling == "spreadsheet":
        glia_n = v["glia_per_neuron_in_spreadsheet"]
    elif glia_scaling == "parameter":
        glia_n = v["astrocytes_per_neuron"]
    else:
        raise ValueError(f"unknown glia_scaling {glia_scaling!r}")
    comps["resting_potential_neuron"] = float((neuron * na).to("1/s").magnitude)
    comps["resting_potential_glia"] = float((glia_one * glia_n * na).to("1/s").magnitude)

    return Budget(components={k: comps[k] for k in SIGNALING_COMPONENTS}, vesicles_per_s=float(ves_rate.magnitude),
                  per_vesicle_costs=c, neuron_density_per_m3=v["neuron_density"],
                  avogadro_for_conversion=v["avogadro_in_unit_conversion"], tissue_density_g_per_m3=v["tissue_density"],
                  housekeeping_basis=housekeeping_basis, housekeeping_fraction=hk)


def atp_free_energy_power(atp_per_s, delta_g_j_per_mol, avogadro):
    """Free-energy rate of ATP hydrolysis in watts, N_dot / N_A x |dG_ATP|."""
    if not (np.isfinite(atp_per_s) and atp_per_s >= 0 and np.isfinite(delta_g_j_per_mol)
            and np.isfinite(avogadro) and avogadro > 0):
        raise ValueError("ATP rate and free energy must be finite; ATP rate must be nonnegative and Avogadro positive")
    return float((Q(atp_per_s, "1/s") / Q(avogadro, "1/mol") * Q(abs(delta_g_j_per_mol), "J/mol")).to("W").magnitude)
