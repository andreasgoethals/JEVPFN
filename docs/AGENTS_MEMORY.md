# Operational notes

## Current state

Exploration and feature design only. No live API client, paid features, model fitting or VSC
jobs. The public GitHub repository is connected; generated outputs and raw data stay ignored.
The literature submodule remains read-only at 81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba.
Current research choices are in RESEARCH_PLAN.md, not duplicated here.

## Avoid repeating these failures

- **Raw-data loss (30-09-2026):** the owner reported data deletion after an earlier empty-folder
  cleanup command. All 20 pinned releases were restored and hash-checked. The command is now
  an error-only stub. Never revive it or recursively clean input/cache folders.
- **Parallel audit race on Windows:** atomic replacement alone cannot replace an open NPZ.
  Retain the OS-backed audit lock for writers and readers, and the report aggregation lock.
- **Wrong Python:** the default runtime was Python 3.14/pandas 3. Use this project's
  `.venv/Scripts/python.exe` (Python 3.12), including actual Jupyter-kernel verification.
- **Windows ReadOnly directories:** relocation/case-only renaming may fail. Resolve exact
  workspace paths and verify contents before a native PowerShell move/removal. Do not combine
  filesystem operations across shells. The dataset folders and pinned source snapshots are protected.
- **VSC environment shadowing:** explicitly use the activated Conda interpreter; an inherited
  local venv can otherwise take precedence. Keep JEVPFN and CreditPFN environments separate.
- **Cluster account:** reuse the configured CreditPFN account after checking association.
  The former Mindwell pilot account is not a fallback. No JEVPFN cluster runs exist yet.

Historical details remain in Git history. Keep this file short and record only reusable
operational evidence; put current task results in generated reports and edits in CHANGELOG.md.
