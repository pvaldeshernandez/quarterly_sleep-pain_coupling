"""Graded Chronic Pain Scale: intensity, disability, grade and pain impact.

The lab's definition: Valdes-Hernandez et al. (2024), J Pain 25(5):104423,
section 2.4 "Selection of knee OA pain participants" (after Von Korff et al. 1992,
Pain 50:133-149). On the baseline visit it reproduces the export's
PainGrade_{no,low,high}_impact__s1 for every participant with all seven items.
Seven items:

    q1 current pain, q2 worst pain, q3 average pain          0-10
    q4 interference with daily activities,
    q5 change in social/recreational activities,
    q6 change in ability to work                              0-10
    q7 days kept from usual activities in the recall window   count

Scoring:
    characteristic pain intensity (CPI) = 10 * mean(q1, q2, q3)       0-100
    disability score                    = 10 * mean(q4, q5, q6)       0-100
    disability points = points for the score (0-29: 0, 30-49: 1, 50-69: 2, >=70: 3)
                      + points for the days (0-6: 0, 7-14: 1, 15-30: 2, >=31: 3)
    grade   0  pain free (CPI = 0)
            I  CPI < 50 and < 3 disability points      low intensity
            II CPI >= 50 and < 3 disability points     high intensity
            III 3-4 disability points                  moderately limiting
            IV  5-6 disability points                  severely limiting
    impact  grade 0: no impact; I-II: low impact; III-IV: high impact

Any missing item leaves that person's scores missing.

Functions
---------
characteristic_pain_intensity(current, worst, average)
disability_score(interference, social, work)
disability_points(score, days)
grade(cpi, points)
impact(grade_values)
score_items(frame, items)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["characteristic_pain_intensity", "disability_score", "disability_points",
           "grade", "impact", "IMPACT_LEVELS", "score_items"]

#: impact level of each grade
IMPACT_LEVELS = {0: "no impact", 1: "low impact", 2: "low impact",
                 3: "high impact", 4: "high impact"}
_SCORE_EDGES = [30, 50, 70]       # disability score -> 0..3 points
_DAYS_EDGES = [7, 15, 31]         # disability days  -> 0..3 points


def characteristic_pain_intensity(current, worst, average):
    """CPI, 0-100: ten times the mean of the three 0-10 intensity ratings."""
    return 10 * (pd.Series(current) + pd.Series(worst) + pd.Series(average)) / 3


def disability_score(interference, social, work):
    """Disability score, 0-100: ten times the mean of the three 0-10 ratings."""
    return 10 * (pd.Series(interference) + pd.Series(social) + pd.Series(work)) / 3


def _points(values, edges):
    """0..len(edges) points by the bin a value falls in; missing stays missing."""
    v = pd.Series(values, dtype=float)
    return pd.Series(np.digitize(v, edges), index=v.index).where(v.notna())


def disability_points(score, days):
    """Disability points, 0-6: score points plus days points."""
    return _points(score, _SCORE_EDGES) + _points(days, _DAYS_EDGES)


def grade(cpi, points):
    """GCPS grade 0-4 (IV = 4) from CPI and disability points."""
    cpi, points = pd.Series(cpi, dtype=float), pd.Series(points, dtype=float)
    g = np.select([cpi == 0, points >= 5, points >= 3, cpi >= 50],
                  [0, 4, 3, 2], default=1).astype(float)
    return pd.Series(g, index=cpi.index).where(cpi.notna() & points.notna())


def impact(grade_values):
    """'no impact', 'low impact' or 'high impact' for each grade (missing stays)."""
    return pd.Series(grade_values).map(IMPACT_LEVELS)


def score_items(frame, items):
    """Every GCPS score for the rows of ``frame``.

    ``items`` maps q1..q7 to column names of ``frame``. Returns a frame with
    cpi, disability, points, grade and impact, indexed like ``frame``.
    """
    q = {k: frame[items[k]].astype(float) for k in (f"q{i}" for i in range(1, 8))}
    cpi = characteristic_pain_intensity(q["q1"], q["q2"], q["q3"])
    dis = disability_score(q["q4"], q["q5"], q["q6"])
    pts = disability_points(dis, q["q7"])
    g = grade(cpi, pts)
    return pd.DataFrame({"cpi": cpi, "disability": dis, "points": pts,
                         "grade": g, "impact": impact(g)}, index=frame.index)
