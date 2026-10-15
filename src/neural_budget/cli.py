"""Command-line interface: neural-budget {reproduce,scenario,validate}."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def cmd_reproduce(args):
    from .report import render
    from .run import run, write

    result = run()
    out = Path(args.out)
    write(result, out)
    (out / "report.html").write_text(render(result, __version__))
    for name in ("reference_reproduction.csv", "component_budget.csv", "parameter_sources.csv", "sensitivity.csv",
                 "scenario_results.csv", "spreadsheet_findings.csv", "limiting_cases.csv", "report.html"):
        print(out / name)
    bad = result["reference_reproduction"].verdict.ne("SUPPORTED").sum()
    return 0 if bad == 0 and result["limiting_cases"]["passed"].all() else 1


def cmd_scenario(args):
    from .model import cortical_budget
    from .params import apply_scenario, load_scenario, reference_parameters, values
    from .run import _budget_options

    sc = load_scenario(args.file)
    ref = cortical_budget(values(reference_parameters()))
    b = cortical_budget(values(apply_scenario(reference_parameters(), sc["changes"], sc["name"])), **_budget_options(sc))
    print(f"scenario: {sc['name']} ({sc['source']})")
    print(f"signaling {b.umol_per_g_min(b.signaling):.4f} umol ATP/g/min (reference {ref.umol_per_g_min(ref.signaling):.4f})")
    print(f"total     {b.umol_per_g_min(b.total):.4f} umol ATP/g/min (reference {ref.umol_per_g_min(ref.total):.4f})")
    for g, x in b.groups().items():
        x0 = ref.groups()[g]
        print(f"  {g:<32} {x:.4e} ATP/neuron/s ({100 * (x / x0 - 1):+.2f}%)")
    return 0


def cmd_validate(args):
    from .reference import paper_checks, spreadsheet_checks
    from .run import limiting_cases
    from .params import reference_parameters, values

    ok = True
    for r in spreadsheet_checks() + paper_checks():
        ok &= r["verdict"] == "SUPPORTED"
        print(f"{r['verdict']:<12} {r['reference']:<11} {r['quantity']}: {r['difference']:.2e} (tol {r['tolerance']:.0e})")
    for r in limiting_cases(values(reference_parameters())):
        ok &= bool(r["passed"])
        print(f"{'PASS' if r['passed'] else 'FAIL':<12} limiting case {r['case']}: {r['observed']}")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="neural-budget", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("reproduce", help="reference reproduction, scenarios, sensitivity and report")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_reproduce)
    s = sub.add_parser("scenario", help="evaluate one scenario file against the reference")
    s.add_argument("file")
    s.set_defaults(func=cmd_scenario)
    s = sub.add_parser("validate", help="reference reproduction and limiting cases")
    s.set_defaults(func=cmd_validate)
    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError, KeyError) as exc:
        print(f"{p.prog}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
