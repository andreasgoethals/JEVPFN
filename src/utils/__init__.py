"""Cross-cutting helpers: paths, config, logging, cleanup, the notebook runner.

Deliberately no re-exports. `from src.utils.paths import outputs_dir` says which module
owns the name; `from src.utils import outputs_dir` hides it, and the whole point of
`paths.py` is that there is one obvious place a path comes from.
"""
