"""Which version of the coupling model the pipeline runs, and where its outputs go.

The one selector. Every step reads its output folders from here, and
``fit_bayesian_varx1`` reads its default centering from here, so one setting
switches the whole paper:

    python run_pipeline.py --refit                    # latent (the default)
    python run_pipeline.py --refit --model observed   # the published model

``run_pipeline.py --model`` sets the environment variable ``COUPLING_MODEL``
before any step is imported; a step run on its own reads the same variable
(``COUPLING_MODEL=observed python step08_fit_coupling_model.py --refit``).

The default is ``latent``, which is not the model of the published paper; select
``observed`` to reproduce the paper. Output folders follow the model, not the
default: the published model always writes to the unsuffixed folders.

Models
------
latent (default)
    Each person's usual pain and sleep level is a parameter of the model, and the
    first quarter of every segment is modeled from the person's stationary
    distribution (see ``fit_bayesian_varx1``). Outputs go to
    ``derivatives_latent/`` and ``results_latent/``.
observed
    The published model: each person's observed mean is subtracted in step 07.
    Outputs go to ``derivatives/`` and ``results/``, unchanged.

The input folder ``data/`` is shared: it holds the raw export and step 00's
extraction, which no model choice touches.
"""
from __future__ import annotations

import os

#: model name -> centering passed to ``fit_bayesian_varx1``.
MODELS = {"observed": "observed", "latent": "latent"}
DEFAULT = "latent"
#: the model of the published paper; its outputs keep the unsuffixed folders
PUBLISHED = "observed"
ENV_VAR = "COUPLING_MODEL"


def current():
    """The selected model name, validated."""
    name = os.environ.get(ENV_VAR, DEFAULT).strip().lower() or DEFAULT
    if name not in MODELS:
        raise ValueError(f"{ENV_VAR}={name!r}; expected one of {sorted(MODELS)}")
    return name


def centering():
    """The centering ``fit_bayesian_varx1`` uses when the caller names none."""
    return MODELS[current()]


def suffix():
    """Folder suffix: '' for the published model, '_<name>' for any other."""
    name = current()
    return "" if name == PUBLISHED else f"_{name}"


def versioned(root, kind):
    """Output folder ``<root>/<kind><suffix>`` (kind: 'derivatives' or 'results')."""
    if kind not in ("derivatives", "results"):
        raise ValueError(f"only output folders are versioned; got {kind!r}")
    return os.path.join(root, kind + suffix())
