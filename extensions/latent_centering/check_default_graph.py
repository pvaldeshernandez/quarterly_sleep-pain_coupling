"""Default graph of fit_bayesian_varx1 (centering="observed") vs the committed module."""
import importlib.util, sys, numpy as np
MAIN = "/orange/cruzalmeida/pvaldeshernandez/Sleep-Pain_Coupling/UPLOAD2/codes/python/lib/coupling_model.py"
SAND = "/orange/cruzalmeida/pvaldeshernandez/Sleep-Pain_Coupling/UPLOAD2-sandbox-latent/codes/python/lib/coupling_model.py"
CSV = "/orange/cruzalmeida/pvaldeshernandez/Sleep-Pain_Coupling/UPLOAD2/derivatives/step07_varx_data/step07_processed_long.csv"
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
def capture(mod, **kw):
    got = {}
    mod.run_fit = lambda model, **k: got.setdefault("m", model)
    _, model_df, ids, idmap = mod.load_varx_frame(CSV)
    mod.fit_bayesian_varx1(model_df, ids, idmap, include_agesex=True, **kw)
    return got["m"]
m0 = capture(load(MAIN, "cm_main")); m1 = capture(load(SAND, "cm_sand"))
names0 = [v.name for v in m0.free_RVs]; names1 = [v.name for v in m1.free_RVs]
print("free RVs identical:", names0 == names1, len(names0))
print("deterministics identical:", [v.name for v in m0.deterministics] == [v.name for v in m1.deterministics])
rng = np.random.default_rng(0)
for k in range(3):
    pt0 = m0.initial_point()
    pt = {n: v + (rng.normal(size=np.shape(v)) * 0.3 if k else 0) for n, v in pt0.items()}
    l0 = m0.compile_logp()(pt); l1 = m1.compile_logp()(pt)
    print(f"point {k}: logp main={l0:.10f} sandbox={l1:.10f} diff={l1 - l0:.3e}")
