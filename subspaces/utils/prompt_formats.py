"""ICL prompt-format registry (paper template-robustness variants).

The submitted paper renders every in-context demonstration with a single fixed
template — ``"{x}->{y}#"`` per demo, then ``"{x_q}->"`` for the query. Todd et
al. (2023), *Function Vectors in Large Language Models* (arXiv:2310.15213),
instead probe robustness to the *form* of the ICL prompt by varying the
input/output **prefixes** and the input/output **separators** (their
``portability_eval`` builds templates from ``all_prefixes`` × ``all_separators``).

Their template model renders one demonstration as::

    in_prefix + str(x) + in_sep + out_prefix + str(y) + out_sep

and the trailing (answer-less) query as::

    in_prefix + str(x_q) + in_sep + out_prefix

The paper's ``"{x}->{y}#"`` arrow format is itself an instance of this model
(``in_prefix=""``, ``out_prefix="->"``, ``in_sep=""``, ``out_sep="#"``), so the
``"arrow"`` entry below reproduces the legacy ``format_input`` **byte-for-byte**
and is the default when ``FV_PROMPT_FORMAT`` is unset.

The five non-arrow formats are drawn directly from Todd et al.'s
``all_prefixes`` / ``all_separators`` lists, chosen to span the variation
(label style: Question/Answer, Input/Output, A/B, text/label, functional x/f(x);
delimiters: newline, double-newline, space, pipe).

Selecting a format
-------------------
The whole pipeline (matrix training, z-extraction, evaluation) routes prompt
construction through ``subspaces.utils.data.format_input``, which consults
``active_format_name()``. Set the environment variable once and every stage —
including ``run_pipeline``'s subprocess ``train_matrix`` call — picks it up::

    FV_PROMPT_FORMAT=qa python -m subspaces.runners.run_pipeline ...

``format_tag()`` is appended to the cached ``z_results_dict`` filename so
per-format activations never collide (and the arrow cache keeps its legacy,
untagged name for backwards compatibility).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

ENV_VAR = "FV_PROMPT_FORMAT"
DEFAULT_FORMAT = "arrow"

# Each entry: in_pre, out_pre, in_sep, out_sep.
#   demo  = in_pre + x + in_sep + out_pre + y + out_sep
#   query = in_pre + x_q + in_sep + out_pre
# All non-arrow prefixes/separators are taken verbatim from Todd et al.'s
# portability_eval all_prefixes / all_separators.
FORMATS: Dict[str, Dict[str, str]] = {
    # Legacy paper-of-record format; byte-identical to the original format_input.
    "arrow": {"in_pre": "", "out_pre": "->", "in_sep": "", "out_sep": "#"},
    # prefix {question/answer-style}; sep {input '\n', output '\n\n'}
    "qa": {
        "in_pre": "Question:",
        "out_pre": "Answer:",
        "in_sep": "\n",
        "out_sep": "\n\n",
    },
    # prefix {Input/Output}; sep {input '\n', output '\n'}
    "io": {"in_pre": "Input:", "out_pre": "Output:", "in_sep": "\n", "out_sep": "\n"},
    # prefix {A/B}; sep {input ' ', output '\n'}
    "ab": {"in_pre": "A:", "out_pre": "B:", "in_sep": " ", "out_sep": "\n"},
    # prefix {text/label}; sep {input ' ', output '\n\n'}
    "textlabel": {
        "in_pre": "text:",
        "out_pre": "label:",
        "in_sep": " ",
        "out_sep": "\n\n",
    },
    # prefix {x/f(x)}; sep {input ' ', output '|'}
    "fx": {"in_pre": "x:", "out_pre": "f(x):", "in_sep": " ", "out_sep": "|"},
}


def active_format_name() -> str:
    """Name of the format selected via ``$FV_PROMPT_FORMAT`` (default ``arrow``)."""
    return os.environ.get(ENV_VAR) or DEFAULT_FORMAT


def get_format(name: Optional[str] = None) -> Dict[str, str]:
    name = name or active_format_name()
    if name not in FORMATS:
        raise KeyError(
            f"Unknown prompt format {name!r}. Known: {sorted(FORMATS)}. "
            f"Set ${ENV_VAR} to one of these."
        )
    return FORMATS[name]


def render_demo(x, y, fmt: Optional[Dict[str, str]] = None) -> str:
    f = fmt or get_format()
    return f"{f['in_pre']}{x}{f['in_sep']}{f['out_pre']}{y}{f['out_sep']}"


def render_query(x_q, fmt: Optional[Dict[str, str]] = None) -> str:
    f = fmt or get_format()
    return f"{f['in_pre']}{x_q}{f['in_sep']}{f['out_pre']}"


def format_tag(name: Optional[str] = None) -> str:
    """Filename suffix for per-format artifacts. Empty for the arrow default so
    legacy untagged caches/paths keep working unchanged."""
    name = name or active_format_name()
    return "" if name == DEFAULT_FORMAT else f"_fmt-{name}"
