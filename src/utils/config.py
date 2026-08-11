"""Reading `config/`. EMPTY ON PURPOSE — this project fills it in.

How a project reads its configuration is project-specific: what the knobs are, whether a file
describes one run or a sweep, whether anything is validated. The template does not guess.

What the template does ask: whatever you build here, a run should write the **fully resolved**
configuration it used into `output/manifests/`. The YAML on disk may have been edited since, so
that copy is the only reliable answer to "what produced this result?".
"""

from __future__ import annotations
