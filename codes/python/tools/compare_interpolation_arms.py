"""Compare the two interpolation arms on everything the paper concludes.

The question is not whether numbers move -- they will -- but whether any CONCLUSION moves:
a credibility verdict, a threshold crossing, a sign, a ranking, or a formula that stops
being computable. Small shifts in an estimate are noise for the reader; a verdict flip is
a different paper.

Arms:
  OLD  SINGLETON_GAPS_ONLY = False  -- the submitted behaviour: fills the first quarter of
       an interior gap of ANY length, up to endpoints 18 months apart. N = 229 / 1,818.
  NEW  SINGLETON_GAPS_ONLY = True   -- gaps of exactly one quarter, which is what the
       Methods sentence describes. N = 227 / 1,793.

Both arms are run through the identical pipeline: same non-centered model, same
4000/4000/0.99 sampler, same seed. The only difference is the interpolation rule.

    python tools/compare_interpolation_arms.py --new-root ../UPLOAD2_results_newinterp_20260808
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))     # codes/python/tools
CUR = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # UPLOAD2
ROOT = os.path.dirname(CUR)                           # the parent holding both arms


def credible(lo, hi):
    return (lo > 0) or (hi < 0)


def load(root_results, root_deriv, rel):
    p = os.path.join(root_results, rel)
    if os.path.exists(p):
        return pd.read_csv(p)
    p = os.path.join(root_deriv, rel)
    return pd.read_csv(p) if os.path.exists(p) else None


def table4(res, der):
    d = load(res, der, "step07_coupling_model/step07_table4_coupling.csv")
    return d.set_index("Parameter") if d is not None else None


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-results", default=os.path.join(ROOT,
                                                          "UPLOAD2_results_newinterp_20260808"))
    ap.add_argument("--new-deriv", default=os.path.join(ROOT,
                                                        "UPLOAD2_deriv_newinterp_20260808"))
    a = ap.parse_args()

    OLD_R, OLD_D = os.path.join(CUR, "results"), os.path.join(CUR, "derivatives")
    NEW_R, NEW_D = a.new_results, a.new_deriv

    flips = []

    # ---- sample --------------------------------------------------------------
    section("SAMPLE")
    for lab, R, D in (("OLD (submitted rule)", OLD_R, OLD_D), ("NEW (Methods rule)", NEW_R, NEW_D)):
        p = os.path.join(D, "step04_varx_data/step04_processed_long.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            b = d.dropna(subset=["pain_within_lag1", "sleep_within_lag1"])
            print(f"  {lab:22s} {d['ID'].nunique():4d} participants, {len(d):5d} person-quarters, "
                  f"{len(b):5d} transitions")

    # ---- Table 4 -------------------------------------------------------------
    section("TABLE 4 — population coupling parameters")
    o, n = table4(OLD_R, OLD_D), table4(NEW_R, NEW_D)
    if o is not None and n is not None:
        print(f"  {'parameter':34s} {'OLD':>22s} {'NEW':>22s}  verdict")
        for k in o.index:
            if k not in n.index:
                continue
            ro, rn = o.loc[k], n.loc[k]
            co, cn = credible(ro["CrI_lo"], ro["CrI_hi"]), credible(rn["CrI_lo"], rn["CrI_hi"])
            f = lambda r: f"{r['Estimate']:+.4f}[{r['CrI_lo']:+.3f},{r['CrI_hi']:+.3f}]"
            mark = "   <<< FLIP" if co != cn else ""
            if co != cn:
                flips.append(f"Table 4 {k} ({ro['Description']}): "
                             f"{'credible' if co else 'ns'} -> {'credible' if cn else 'ns'}")
            print(f"  {str(ro['Description'])[:34]:34s} {f(ro):>22s} {f(rn):>22s}  "
                  f"{'cred' if co else 'ns':>4}->{'cred' if cn else 'ns':<4}{mark}")

    # ---- LOO -----------------------------------------------------------------
    section("LOO — the |delta/SE| > 2 claim")
    for lab, R, D in (("OLD", OLD_R, OLD_D), ("NEW", NEW_R, NEW_D)):
        d = load(R, D, "step07_coupling_model/step07_loo_comparison.csv")
        if d is None:
            continue
        row = d[(d.model_a == "full") & (d.model_b == "no_PS")]
        if len(row):
            r = row.iloc[0]
            print(f"  {lab}: full vs no-PS  delta_elpd={r['delta_elpd']:+7.2f}  SE={r['se']:5.2f}  "
                  f"delta/SE={r['delta_over_se']:.2f}  {'PASSES' if abs(r['delta_over_se'])>2 else 'FAILS'} the >2 threshold")

    # ---- the optimal-lag formula --------------------------------------------
    section("SECTION S14 — is the optimal lag computable?")
    for lab, R, D in (("OLD", OLD_R, OLD_D), ("NEW", NEW_R, NEW_D)):
        t = table4(R, D)
        if t is None:
            continue
        pp, ps = t.loc["a1", "Estimate"], t.loc["b2", "Estimate"]
        lsp, lps = t.loc["a2", "Estimate"], t.loc["b1", "Estimate"]
        disc = (pp - ps) ** 2 + 4 * lsp * lps
        l1 = ((ps + pp) + np.sqrt(disc)) / 2
        l2 = ((ps + pp) - np.sqrt(disc)) / 2
        ok3 = ps > 0
        w = (-np.log(np.log(pp) / np.log(ps)) / (np.log(pp) - np.log(ps))) if ok3 else None
        print(f"  {lab}: phi_s={ps:+.4f}  lambda2={l2:+.4f}")
        print(f"      Eq.S2 (needs both eigenvalues > 0): "
              f"{'applicable' if (l1 > 0 and l2 > 0) else 'INAPPLICABLE'}")
        print(f"      Eq.S3 (needs ln phi_s):             "
              f"{'computable, omega_opt=%.3f quarters = %.0f days' % (w, w*91) if ok3 else 'UNDEFINED (phi_s < 0)'}")
        if not ok3:
            flips.append("Section S14: the optimal-lag surrogate is undefined (phi_s < 0)")

    # ---- moderation tables ---------------------------------------------------
    for title, rel, est, lo, hi, key in (
        ("TABLE 5 — sleep-to-pain fMRI moderation",
         "step14_sp_moderation/step14_table5_sp_moderation.csv",
         "gamma_sp", "gamma_sp_ci_lo", "gamma_sp_ci_hi", "ROI"),
    ):
        section(title)
        o = load(OLD_R, OLD_D, rel)
        n = load(NEW_R, NEW_D, rel)
        if o is None or n is None:
            print("  (not found in one arm)")
            continue
        m = o[[key, est, lo, hi]].merge(n[[key, est, lo, hi]], on=key, suffixes=("_o", "_n"))
        for _, r in m.iterrows():
            co = credible(r[lo + "_o"], r[hi + "_o"]); cn = credible(r[lo + "_n"], r[hi + "_n"])
            mark = "   <<< FLIP" if co != cn else ""
            if co != cn:
                flips.append(f"{title.split('—')[0].strip()} {r[key]}: "
                             f"{'credible' if co else 'ns'} -> {'credible' if cn else 'ns'}")
            print(f"  {str(r[key])[:26]:26s} {r[est+'_o']:+.4f} -> {r[est+'_n']:+.4f}   "
                  f"{'cred' if co else 'ns':>4}->{'cred' if cn else 'ns':<4}{mark}")

    # ---- arousal relay -------------------------------------------------------
    section("TABLE S9 — arousal relay (rank order matters for the PBN sentence)")
    for lab, R, D in (("OLD", OLD_R, OLD_D), ("NEW", NEW_R, NEW_D)):
        d = load(R, D, "step19_ps_moderation/step19_text_numbers.csv")
        if d is None:
            continue
        s = d.set_index(d.columns[0])[d.columns[1]]
        rows = [(k.replace("gamma_ps_fmri_", ""), float(v)) for k, v in s.items()
                if k.startswith("gamma_ps_fmri_") and not k.endswith("_p")]
        rows.sort(key=lambda r: -abs(r[1]))
        print(f"  {lab}: " + ", ".join(f"{r}={v:+.4f}" for r, v in rows))
        conc = s.get("vbm_sign_concordance", "?")
        print(f"       VBM sign concordance {conc}, p={s.get('vbm_sign_concordance_p','?')}")

    # ---- verdict -------------------------------------------------------------
    section("CONCLUSION CHANGES (old rule -> new rule)")
    if flips:
        for f in flips:
            print(f"  * {f}")
    else:
        print("  none — no credibility verdict, threshold or formula changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
