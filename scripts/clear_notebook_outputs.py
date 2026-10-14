#!/usr/bin/env python
"""Strip outputs and execution counts from notebooks, in place.

    python scripts/clear_notebook_outputs.py                 # every *.ipynb in the repo
    python scripts/clear_notebook_outputs.py analysis.ipynb  # just this one
    python scripts/clear_notebook_outputs.py --check         # exit 1 if any output is present

Run before committing. Rendered notebook output is the largest thing in this repo by a wide
margin — `analysis.ipynb` is 1.3 MB with outputs and 56 KB without — and it embeds base64 PNGs
of figures derived from the challenge data, which is data we do not redistribute.

`--check` is the form to use in CI or a pre-commit hook: it changes nothing and just reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Cell-level metadata that records a *rendered* state rather than the source.
_RENDER_METADATA = ("scrolled", "collapsed", "execution")


def strip(path: Path, check: bool = False) -> tuple[int, int]:
    """Return (outputs found, execution counts found). Writes unless ``check``."""
    nb = json.loads(path.read_text())
    n_out = n_exec = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        n_out += len(cell.get("outputs") or [])
        n_exec += cell.get("execution_count") is not None
        if not check:
            cell["outputs"] = []
            cell["execution_count"] = None
            for k in _RENDER_METADATA:
                cell.get("metadata", {}).pop(k, None)
    if not check:
        # top-level widget state embeds rendered widget views and survives cell clearing
        nb.get("metadata", {}).pop("widgets", None)
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return n_out, n_exec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="*", type=Path)
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if anything found")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    paths = args.notebooks or sorted(
        p for p in root.rglob("*.ipynb")
        if not any(part in {".venv", ".ipynb_checkpoints", ".backups"} for part in p.parts)
    )
    dirty = False
    for p in paths:
        before = p.stat().st_size
        n_out, n_exec = strip(p, check=args.check)
        if n_out or n_exec:
            dirty = True
            if args.check:
                print(f"{p}: {n_out} outputs, {n_exec} execution counts")
            else:
                print(f"{p}: cleared {n_out} outputs, {n_exec} execution counts "
                      f"({before:,} -> {p.stat().st_size:,} bytes)")
        else:
            print(f"{p}: already clean")
    if args.check and dirty:
        print("\nrun `python scripts/clear_notebook_outputs.py` before committing", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
