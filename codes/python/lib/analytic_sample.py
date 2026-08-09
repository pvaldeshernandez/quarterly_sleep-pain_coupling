"""Who is in the analytic sample.

One definition, taking the frame that DEFINES the sample as an argument rather
than hard-coding a path. Four sandbox scripts each had their own ``analytic_ids``
pointing at a path that has since moved twice; a hard-coded path is how a helper
silently keeps answering after the thing it describes has changed.

Functions
---------
analytic_ids(path_or_frame)
    The set of participant IDs in the modelling frame, as strings.
"""
from __future__ import annotations

import pandas as pd

__all__ = ["analytic_ids"]


def analytic_ids(source, id_col="ID"):
    """The participant IDs of the analytic sample, as a set of strings.

    Parameters
    ----------
    source : str or DataFrame
        The processed long frame that defines the sample (step 04's output), or
        a path to it. Whoever calls this owns the decision of WHICH frame
        defines the sample; this function only answers "who is in it".
    id_col : str, default "ID"
        The ID column.

    Returns
    -------
    set of str
        IDs are strings because they carry re-enrolment suffixes ("2014-2");
        numeric coercion would silently drop those participants from every join.
    """
    if isinstance(source, pd.DataFrame):
        frame = source
    else:
        frame = pd.read_csv(source, usecols=[id_col])
    if id_col not in frame.columns:
        raise KeyError(f"{id_col!r} absent from the analytic frame")
    return set(frame[id_col].astype(str))
