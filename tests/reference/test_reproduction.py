"""The reference case against the spreadsheet's computed cells and the paper's printed numbers."""

import json
from importlib import resources
from pathlib import Path

import pytest

from neural_budget.params import reference_parameters, values
from neural_budget.reference import paper_checks, spreadsheet_checks, spreadsheet_findings
from neural_budget.run import limiting_cases

PROJECT = Path(__file__).resolve().parents[2]


def test_spreadsheet_cells_reproduced():
    rows = spreadsheet_checks()
    assert len(rows) == 25
    assert all(r["verdict"] == "SUPPORTED" for r in rows), [r for r in rows if r["verdict"] != "SUPPORTED"]
    assert max(r["difference"] for r in rows) < 1e-15


def test_paper_numbers_reproduced():
    rows = paper_checks()
    assert len(rows) == 13
    assert all(r["verdict"] == "SUPPORTED" for r in rows), [r for r in rows if r["verdict"] != "SUPPORTED"]


def test_spreadsheet_findings_are_derived():
    f = {(r["finding"], r["cell"]) for r in spreadsheet_findings()}
    assert ("vesicle costs counted twice", "K24") in f
    assert ("glial scaling typed as a constant", "M46") in f
    assert ("parameter not used by any formula", "B35") in f
    assert ("parameter not used by any formula", "B3") in f
    assert ("input typed in the calculation area", "F14") in f
    assert ("parameter not used by any formula", "B9") not in f, "a used parameter must not be reported"


def test_limiting_cases():
    rows = limiting_cases(values(reference_parameters()))
    assert len(rows) == 4
    assert all(r["passed"] for r in rows), rows


@pytest.mark.extract
def test_packaged_extract_equals_a_fresh_extract(tmp_path):
    import subprocess
    import sys

    xls = PROJECT / "data" / "raw" / "howarth2012_supplementary_spreadsheet.xls"
    if not xls.exists():
        pytest.skip("pinned spreadsheet not present")
    pytest.importorskip("xlrd")
    out = tmp_path / "fresh.json"
    subprocess.run([sys.executable, str(PROJECT / "tools" / "extract_reference.py"), str(xls), str(out)], check=True)
    packaged = json.loads((resources.files("neural_budget") / "reference_data" / "howarth2012_cortex.json").read_text())
    assert json.loads(out.read_text()) == packaged
