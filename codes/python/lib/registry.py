"""The numbers registry — every quantity a step computes, under a stable key.

Steps no longer generate prose. They compute numbers and write them here; the numbers
are typed into the documents by hand, and `tools/check_numbers.py` then verifies that
nothing in the documents fails to trace back to a run.

Named `registry`, not `numbers`: `lib/` has no `__init__.py` and every step reaches it
with `sys.path.insert(0, LIB_DIR)` followed by a bare import, so a module called
`numbers` would shadow the standard library's and break unrelated imports.

Keys are namespaced by step. That is not tidiness: `step01_factor_analysis` already
publishes `n_rows_interpolated`, and the interpolation-sensitivity step computes a
different quantity that would naturally take the same name. A flat namespace would let
one silently overwrite the other in the merged registry.

Usage:
    from registry import write_numbers
    write_numbers(STEP_RESULTS_DIR, {"lambda_ps": -0.136, "n_persons": 229})
"""
import json
import os

FILENAME = "numbers.json"


def _plain(v):
    """Convert numpy scalars and arrays to JSON-safe Python values."""
    if hasattr(v, "item") and getattr(v, "ndim", 0) == 0:
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


def write_numbers(step_dir, mapping, prefix=None):
    """Write `mapping` to `<step_dir>/numbers.json`, replacing any previous content.

    `prefix` namespaces every key (`prefix_key`). It defaults to the step directory's
    own name, so a step cannot collide with another by accident.

    Returns the path written.
    """
    os.makedirs(step_dir, exist_ok=True)
    if prefix is None:
        prefix = os.path.basename(os.path.normpath(step_dir))
    out = {f"{prefix}.{k}": _plain(v) for k, v in mapping.items()}
    path = os.path.join(step_dir, FILENAME)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    return path


def read_all(results_root):
    """Merge every step's `numbers.json` under `results_root` into one dict.

    Raises on a duplicate key rather than letting one step's value win silently — a
    collision means two steps claim the same quantity, which the checker could not
    then adjudicate.
    """
    merged = {}
    for entry in sorted(os.listdir(results_root)):
        path = os.path.join(results_root, entry, FILENAME)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            block = json.load(fh)
        clash = set(block) & set(merged)
        if clash:
            raise KeyError(f"{entry}: keys already registered elsewhere: {sorted(clash)}")
        merged.update(block)
    return merged
