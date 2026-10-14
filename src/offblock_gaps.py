"""Signed pairwise gaps between the off-block estimates.

Five timestamps estimate the same event — `SCHED_TIME_UTC_mvt`, `IOBT_flt`, `LOBT_flt`,
`EOBT_1_flt`, `AOBT_3_flt` — and the model already receives each as a proxy
(`MVT_TIME − estimate`). What it cannot form is their **differences**: a tree splits on one
axis at a time, so `LOBT − IOBT > 900` needs many axis-aligned splits to approximate. Each
signed gap is therefore a genuinely new axis, and the sign carries the information: it says
which estimate is later, which is what a hold produces.

`LOBT − IOBT` is the one with a story — LOBT is revised by flow management, so that difference
is the closest thing in the delivered data to an ATFM slot delay, the usual reason an aircraft
is held at the stand.

Measured worth: **−0.52 s full / −0.61 s trimmed** (`experiments/e23_offblock_gaps.py`), i.e.
inside the protocol's ±6 s. Shipped in v5 only because it is free, directionally consistent on
both the full and trimmed numbers, and adds no tail risk — not because it is a result.
"""

from __future__ import annotations

import pandas as pd

#: The five off-block estimates, in the order they are produced operationally.
OFFBLOCK_COLS = {
    "sched": "SCHED_TIME_UTC_mvt",
    "iobt": "IOBT_flt",
    "lobt": "LOBT_flt",
    "eobt1": "EOBT_1_flt",
    "aobt3": "AOBT_3_flt",
}


def add_offblock_gaps(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add every signed pairwise gap plus arrival-side analogues. Returns (frame, new cols)."""
    out = df.copy()
    new: list[str] = []
    keys = list(OFFBLOCK_COLS)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            name = f"gap_{b}_{a}"
            out[name] = (out[OFFBLOCK_COLS[b]] - out[OFFBLOCK_COLS[a]]).dt.total_seconds()
            new.append(name)

    stack = pd.concat([out[c] for c in OFFBLOCK_COLS.values()], axis=1)
    out["offblock_earliest"] = (out["MVT_TIME_UTC_mvt"] - stack.max(axis=1)).dt.total_seconds()
    out["offblock_latest"] = (out["MVT_TIME_UTC_mvt"] - stack.min(axis=1)).dt.total_seconds()
    out["offblock_n_present"] = stack.notna().sum(axis=1).astype("float32")
    new += ["offblock_earliest", "offblock_latest", "offblock_n_present"]

    # arrival side: planned vs flown block-to-block. A record whose ARVT_3 is off is one whose
    # AOBT_3 may be, since both come from the same flown trajectory.
    out["gap_arvt3_arvt1"] = (out["ARVT_3_flt"] - out["ARVT_1_flt"]).dt.total_seconds()
    out["blocktime_planned"] = (out["ARVT_1_flt"] - out["EOBT_1_flt"]).dt.total_seconds()
    out["blocktime_flown"] = (out["ARVT_3_flt"] - out["AOBT_3_flt"]).dt.total_seconds()
    out["blocktime_excess"] = out["blocktime_flown"] - out["blocktime_planned"]
    new += ["gap_arvt3_arvt1", "blocktime_planned", "blocktime_flown", "blocktime_excess"]
    return out, new
