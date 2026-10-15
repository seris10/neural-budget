"""Optional mean-field resource depletion coupled to the source cortical ATP accounting.

Tsodyks & Markram 1997 motivate release-dependent depletion and exponential recovery.
This one-pool approximation omits facilitation, the brief active state, and calcium-dependent
recovery; it is an illustrative extension, not a fit to a measured ATP time series.
"""
from __future__ import annotations
import numpy as np
from .model import cortical_budget
from .params import validate_values


def resource_state(firing_hz, utilization, recovery_s, *, initial=1.0, elapsed_s=None):
    """R'= (1-R)/τ - U*f*R; return steady state or exact finite-time resource fraction."""
    if not np.all(np.isfinite([firing_hz, utilization, recovery_s, initial])) or not (
            firing_hz >= 0 and 0 <= utilization <= 1 and recovery_s > 0 and 0 <= initial <= 1):
        raise ValueError("invalid synapse rate, utilization, recovery time or initial resource")
    rate = 1 / recovery_s + utilization * firing_hz
    steady = 1 / (1 + utilization * firing_hz * recovery_s)
    if elapsed_s is None:
        return float(steady)
    if not np.isfinite(elapsed_s) or elapsed_s < 0:
        raise ValueError("elapsed time must be finite and nonnegative")
    return float(steady + (initial - steady) * np.exp(-rate * elapsed_s))


def depleted_budget(v, recovery_s=0.8, *, resource=None, **budget_options):
    """Use U*R as effective release probability while retaining the source's per-event costs."""
    v = validate_values(v)
    steady = resource_state(v["firing_rate"], v["release_probability"], recovery_s)
    if resource is None:
        resource = steady
    if not np.isfinite(resource) or not 0 <= resource <= 1:
        raise ValueError("resource must be in [0,1]")
    effective = v["release_probability"] * resource
    budget = cortical_budget(dict(v, release_probability=effective), **budget_options)
    return budget, {"resource": float(resource), "steady_resource": steady,
                    "effective_release": float(effective), "recovery_s": float(recovery_s)}
