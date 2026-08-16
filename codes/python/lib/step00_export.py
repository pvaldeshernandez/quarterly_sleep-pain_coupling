"""The one way to read baseline variables: step 00's export.

`data/original/participants_wideformat.xlsx` is opened by step 00 and by nothing else.
That is what step 00 is for -- it enumerates every variable the paper uses, extracts it,
and lands it on the quarter-0 row. A module that opens the raw export instead is reading
a column step 00 never declared, with whatever missing-code and sample handling it
happens to implement locally, and the two answers drift without anything noticing.

So this module exists to make the correct path shorter than the incorrect one:

    from step00_export import baseline
    wide = baseline()                       # one row per subject, ID as a column
    wide = baseline(ids=analytic_ids)       # restricted to a sample

The frame it returns is shaped like the old `pd.read_excel(WIDE_XLSX)` result -- an ID
column plus the baseline columns -- so a caller that used to read the xlsx keeps working
once the call is swapped.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
LONG_CSV = os.path.join(ROOT, "data", "step00_extracted_long.csv")


def baseline(ids=None, columns=None, path=None):
    """Step 00's quarter-0 block: one row per subject, ``ID`` as a string column.

    Parameters
    ----------
    ids : iterable, optional
        Restrict to these participant IDs. Compared as strings.
    columns : list, optional
        Baseline columns to return besides ``ID``. Raises if one is absent, because a
        missing column means the variable was never declared in step 00 -- and the fix
        for that is to declare it there, not to reach past it to the raw export.
    path : str, optional
        Override the export location (tests).
    """
    src = path or LONG_CSV
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"{src} not found; run step00_extract_data.py before this step")
    usecols = None
    if columns is not None:
        wanted = {"ID", "quarter", *columns}
        usecols = lambda c: c in wanted            # noqa: E731
    frame = pd.read_csv(src, dtype={"ID": str}, usecols=usecols,
                        float_precision="round_trip")
    if columns is not None:
        absent = [c for c in columns if c not in frame.columns]
        if absent:
            raise KeyError(
                f"{absent} not in step 00's export. Add them to BASELINE_VARS in "
                f"step00_extract_data.py and re-run it; do not read the raw export here.")
    frame = frame[frame["quarter"] == 0].drop(columns=["quarter"])
    if ids is not None:
        frame = frame[frame["ID"].isin(set(map(str, ids)))]
    return frame.reset_index(drop=True)
