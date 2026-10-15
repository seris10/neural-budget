"""Print a BIFF8 .xls sheet's formulas and cached values as CSV, decompiled with xlrd.

Usage: python tools/xls_formulas.py WORKBOOK.xls SHEET_NAME > formulas.csv
"""

import csv
import struct
import sys

import xlrd
from xlrd import compdoc
from xlrd.formula import FMLA_TYPE_CELL, FMLA_TYPE_SHARED, decompile_formula

FORMULA, SHRFMLA, BOF, EOF, BOUNDSHEET = 0x0006, 0x04BC, 0x0809, 0x000A, 0x0085


def records(stream, start):
    pos = start
    while pos + 4 <= len(stream):
        rc, ln = struct.unpack("<HH", stream[pos:pos + 4])
        yield rc, stream[pos + 4:pos + 4 + ln]
        pos += 4 + ln
        if rc == EOF:
            return


def main():
    w = csv.writer(sys.stdout)
    w.writerow(["cell", "formula", "cached_value"])
    for cell, text, value in sheet_formulas(sys.argv[1], sys.argv[2]):
        w.writerow([cell, text, repr(value)])


def sheet_formulas(path, sheet_name):
    """[(cell address, decompiled formula, cached value)] for every formula cell of one sheet."""
    bk = xlrd.open_workbook(path, formatting_info=False)
    data = open(path, "rb").read()
    cd = compdoc.CompDoc(data)
    stream = cd.get_named_stream("Workbook")
    sh = bk.sheet_by_name(sheet_name)
    offset = None
    for rc, body in records(stream, 0):
        if rc == BOUNDSHEET:
            pos, = struct.unpack("<I", body[:4])
            nlen = body[6]
            flag = body[7]
            raw = body[8:8 + nlen * (2 if flag & 1 else 1)]
            name = raw.decode("utf-16-le" if flag & 1 else "latin-1")
            if name == sheet_name:
                offset = pos
    if offset is None:
        raise SystemExit(f"sheet {sheet_name!r} not found")
    shared = []
    pending = []
    for rc, body in records(stream, offset):
        if rc == SHRFMLA:
            r1, r2, c1, c2 = struct.unpack("<HHBB", body[:6])
            fmlalen, = struct.unpack("<H", body[8:10])
            shared.append((r1, r2, c1, c2, body[10:10 + fmlalen], fmlalen))
        elif rc == FORMULA:
            row, col = struct.unpack("<HH", body[:4])
            fmlalen, = struct.unpack("<H", body[20:22])
            fmla = body[22:22 + fmlalen]
            pending.append((row, col, fmla, fmlalen))
    out = []
    for row, col, fmla, fmlalen in pending:
        text = None
        if fmlalen == 5 and fmla[0] == 0x01:
            r0, c0 = struct.unpack("<HH", fmla[1:5])
            for r1, r2, c1, c2, sf, sflen in shared:
                if r1 <= row <= r2 and c1 <= col <= c2:
                    text = decompile_formula(bk, sf, sflen, FMLA_TYPE_SHARED, browx=row, bcolx=col, r1c1=0)
                    break
        else:
            text = decompile_formula(bk, fmla, fmlalen, FMLA_TYPE_CELL, browx=row, bcolx=col, r1c1=0)
        out.append((xlrd.cellname(row, col), text, sh.cell(row, col).value))
    return out


if __name__ == "__main__":
    main()
