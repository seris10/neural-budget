from importlib import resources

from neural_budget.cli import main


def test_reproduce_writes_every_output(tmp_path):
    out = tmp_path / "run"
    assert main(["reproduce", "--out", str(out)]) == 0
    for name in ("reference_reproduction.csv", "component_budget.csv", "parameter_sources.csv", "sensitivity.csv",
                 "scenario_results.csv", "spreadsheet_findings.csv", "limiting_cases.csv", "report.html"):
        assert (out / name).exists(), name
    html = (out / "report.html").read_text()
    assert "<svg" in html and "factorial parameter bounds" in html.lower()


def test_scenario_and_validate_commands(capsys):
    path = resources.files("neural_budget") / "scenarios" / "hodgkin_overlap_factor.toml"
    assert main(["scenario", str(path)]) == 0
    assert "30.06" in capsys.readouterr().out
    assert main(["validate"]) == 0


def test_invalid_scenario_is_a_command_error_without_a_traceback(tmp_path, capsys):
    p = tmp_path / "invalid.toml"
    p.write_text('[scenario]\nname = "invalid"\n[changes]\nfiring_rate = 4\n'
                 '[options]\nhousekeeping_basis = "signaling"\nhousekeeping_fraction = -0.5\n')
    assert main(["scenario", str(p)]) == 2
    captured = capsys.readouterr()
    assert "nonnegative" in captured.err and "Traceback" not in captured.err
