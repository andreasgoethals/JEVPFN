# Agent instructions — JEVPFN

Read README.md, docs/LITERATURE_REVIEW.md and docs/AGENTS_MEMORY.md before changing this
repository. Existing source/config/notebook conventions remain authoritative.

## Boundaries

- Never stage, commit, push or change Git branches/remotes/history; the owner publishes.
- Proposed plans and requested explanations belong in chat, not temporary documents.
- `tfm-library/` is a pinned read-only submodule. Never edit, move, delete, lint or test it.
  The only allowed library write is its gitignored PROJECT_SPECIFIC.md. Report library errors
  to the owner. Cite code snapshots by symbol, not line number; record the pin used.
- Raw datasets, prepared inputs, permanent response/features and checkpoints are protected.
  Never recursively clean them or a system temporary root. Never force-add data/weights.
- Credentials must never enter code, tracked configuration, notebooks, reports or tool output.
- Main inputs are deterministic and label-free: no generative questions, non-text row values,
  true row targets, class frequencies, target statistics or model feedback. Keep original
  class vocabularies; source meanings live in dataset_semantics.yaml. Skip empty text and
  retain missing features. Do not silently truncate, drop rows/classes or cap training rows.
- The selected roster and method are in README/config. Microsoft is excluded. No new models
  or changes to open design choices without instruction. Use open weights where available.
- Full extraction/benchmark switches remain disabled pending acceptance. Bounded GPU contracts
  require explicit execution, staged pins, complete exact token evidence and a VSC allocation.
  Do not submit allocations or perform paid calls as a side effect of cleanup.
- Preserve tokenization.py and decision_tokenization.py byte-for-byte while audits use their
  fingerprints. Historical availability constants there do not select active providers.
- Existing authorization for environment setup persists; verify the actual interpreter and
  do not alter another project's/shared environment without authorization.

## Layout and operations

Reusable Python is in src/; upstream data-source snapshots stay under src/data/upstream/ and
are never executed. Config/notebook/output phase folders use the same numbers. Do not create
empty phase folders or copied question files. All notebooks run locally; permanent questions
and responses/features each use one SQLite file per selected dataset under data/.

On VSC, PROJECT holds large inputs/weights/environments, permanent caches, token counts,
predictions and large results. DATA holds the checkout, compact manifests and operational
logs. Both use JEVPFN/output_JEVPFN/<number>. phase/. Use the validated transfer export/import,
never wholesale replacement of populated caches. See docs/VSC.md for commands/site details.

PowerShell commands must remain compatible with Windows PowerShell 5.1: no &&, ternary or ??.
Slurm files are Linux Bash and must use LF endings. New/editable text files are UTF-8.

## Notebooks and validation

- Put logic in source modules; notebooks contain no function/class definitions.
- Each notebook ends with the complete printed report in section order.
- src/visualize/style.py owns colors/sizes. Use FigureSaver, PDF output and inline display;
  use its named A4/inspection canvases. Captions describe the figure, ready for a paper.
- Save figures under output_JEVPFN/figures/<phase>/<notebook>/, aggregate captions under
  figures/Captions.md and the complete report under output_JEVPFN/All Results.md.
- Verify actual files/runtime/evidence before claiming success. Synthetic tests establish
  contracts, not CUDA/checkpoint/provider compatibility or full Experiment 0 acceptance.
- Run `python -m pytest -q` before finishing. If a notebook or src/visualize/ changed, also run
  `python -m src.utils.run_notebooks`. Check lint and relevant shell syntax.
- Record substantive changes concisely in docs/CHANGELOG.md, newest first, DD-MM-YYYY.
  Record each cluster run and reusable failures in docs/AGENTS_MEMORY.md. Submission is not
  completion; retain the exact observed status. Keep docs uppercase, durable and concise.
