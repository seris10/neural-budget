"""Extract the parameters, formulas and cells of the pinned Howarth et al. (2012) cortex sheet to JSON.

Usage: python tools/extract_reference.py data/raw/howarth2012_supplementary_spreadsheet.xls OUT.json
"""

import hashlib
import json
import sys
from pathlib import Path

import xlrd

sys.path.insert(0, str(Path(__file__).parent))
from xls_formulas import sheet_formulas  # noqa: E402

SHEET = "Cerebral cortex - calculations"
PINNED_SHA256 = "22bba3b14ef2d807623ccd41bebb58fe76c9b884a986c023910048a53a5c4eea"


def main():
    src, out = sys.argv[1], sys.argv[2]
    digest = hashlib.sha256(Path(src).read_bytes()).hexdigest()
    if digest != PINNED_SHA256:
        raise SystemExit(f"{src}: sha256 {digest} is not the pinned file")
    bk = xlrd.open_workbook(src)
    sh = bk.sheet_by_name(SHEET)
    params = []
    for r in range(1, sh.nrows):
        label = sh.cell(r, 0).value
        if not label or sh.cell(r, 1).ctype != xlrd.XL_CELL_NUMBER:
            continue
        params.append({"cell": xlrd.cellname(r, 1), "label": str(label).strip(), "value": sh.cell(r, 1).value,
                       "reference": str(sh.cell(r, 2).value).strip() if sh.ncols > 2 else "",
                       "note": str(sh.cell(r, 3).value).strip() if sh.ncols > 3 else ""})
    formulas = [{"cell": c, "formula": f, "cached_value": v} for c, f, v in sheet_formulas(src, SHEET)]
    formula_cells = {f["cell"] for f in formulas}
    constants = []
    for r in range(sh.nrows):
        for c in range(4, sh.ncols):
            cell = sh.cell(r, c)
            name = xlrd.cellname(r, c)
            if cell.ctype == xlrd.XL_CELL_NUMBER and name not in formula_cells:
                constants.append({"cell": name, "value": cell.value})
    labels = []
    for r in range(sh.nrows):
        for c in range(4, sh.ncols):
            cell = sh.cell(r, c)
            if cell.ctype == xlrd.XL_CELL_TEXT and cell.value.strip():
                labels.append({"cell": xlrd.cellname(r, c), "text": cell.value.strip()})
    doc = {
        "source": {
            "title": "Supplementary spreadsheet to Howarth, Gleeson and Attwell (2012), Updated energy budgets for "
                     "neural computation in the neocortex and cerebellum, J Cereb Blood Flow Metab 32:1222-1232",
            "doi": "10.1038/jcbfm.2012.35",
            "file": "jcbfm201235x1.xls",
            "retrieved_from": "https://web.archive.org/web/20230426193037id_/http://www.ncbi.nlm.nih.gov/pmc/"
                              "articles/PMC3390818/bin/jcbfm201235x1.xls",
            "sha256": digest,
            "sheet": SHEET,
            "terms": "Spreadsheet text: may be used freely to investigate the effects of changing the parameters "
                     "chosen, or adapted; the authors ask that the paper be acknowledged when results are published.",
        },
        "parameters": params,
        "formulas": formulas,
        "calculation_area_constants": constants,
        "labels": labels,
    }
    Path(out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{out}: {len(params)} parameters, {len(formulas)} formulas, {len(constants)} constants")


if __name__ == "__main__":
    main()
