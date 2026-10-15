# Operational evidence

Current stage: Experiment 0A. Full extraction and predictive runs remain disabled; GPU
contract execution is implemented but no VSC model result has been supplied. The prior CPU
preflight passed; the new clean run still needs setup and fresh CPU/token/storage checks.
Storage quota and independent backup remain unverified. The read-only library pin is
`de324efcca55be392feb0c01830b3b6ce364587f`.
Method constraints are in README; operations are in VSC; source meanings are in DATA_SOURCES.

## VSC runs

| Date | Cluster / job | Configuration | Verified outcome |
|---|---|---|---|
| 14-10-2026 | wICE / 62316611 | experiment_0_cpu.slurm | COMPLETED, exit 0, 1m25s; MaxRSS 1,065,032K. Owner supplied sacct after the dependency/test-scope repair. |
| 14-10-2026 | wICE / 62316685 | experiment_0_tokens.slurm | COMPLETED, exit 0, 1h14m26s; MaxRSS 4,868,592K. All five providers counted 1,698,127 unique inputs each; zero overflows. Historical version-1 questions; repeat after version-2 preparation. |
| 14-10-2026 | wICE / 62312865 | experiment_0_cpu.slurm | FAILED, exit 1, 1m30s; 309 tests passed, 3 notebook tests lacked IPython/nbformat, 1 platform skip; dataset checks passed. |
| 14-10-2026 | wICE / 62312903 | experiment_0_tokens.slurm | FAILED, exit 1, 1h06m02s; Perplexity lacked Jinja2; Liquid/Strands completed counting; committed databases survive. |

Submission/RUNNING are not successful completion. Do not update a checkout/environment while
jobs use it. Record final status from sacct and logs. Account balance is not storage quota.

The completed version-1 audit counted Liquid 821,257,572; Strands 1,141,530,844; Clef
1,143,617,155; Perplexity 960,172,532; Intern 1,316,907,633 input tokens. These are not Jev
billing counts. Clef and Intern weights were staged and fingerprinted on PROJECT; no real
GPU forward result has been supplied. Version 2 removes source URLs from payloads, retains
historical input sets and explicitly activates new sets. Existing audit databases remain
evidence, but cannot authorize execution against changed inputs.

Local follow-up on 14-10-2026 regenerated all 18 input files (1,698,127 unique inputs per
extractor), passed 410 tests and reran all five notebooks (21 figures). The checked upload
bundle contains all 18 byte-identical databases, about 713 MB compressed; no upload has been
performed by the agent. VSC output archiving and transfer recovery passed synthetic tests.
An unauthenticated HTTPS HEAD to
TypeSafe returned 405: connectivity only, not authentication, billing or inference evidence.
The two-node storage probe and real GPU/provider tests remain unexecuted; quota/backup are
still unverified. Cleanup removed only rebuildable reports/figures and legacy question copies.

## Reusable failure lessons

- **15-10-2026 storage setup:** the owner's VSC transcript at commit `08ff22bd` confirms the
  environment check and DATA/PROJECT output deletion succeeded. Setup then failed because it
  tried creating a temporary file in `data/raw`. Existing raw inputs need directory-read access,
  not writes; actual file/hash checks belong in CPU preflight. Setup now separates that check
  from required writes to mutable destinations. Resume setup after publishing this fix, without
  repeating deletion or changing raw permissions. No remote success of this fix is claimed.
- **15-10-2026 local review:** 429 tests and five notebooks/17 inspected figures passed;
  all comparison plot types were additionally rendered with synthetic six-provider data,
  including 30-class output. Packaging, dependencies, dataset preflight, lint and shell checks
  passed. No new cluster/inference evidence exists; rerun 0A after publishing these fixes.
- **KU Leuven job environment:** submitting-shell exports are not inherited. Slurm scripts
  take provider/environment/root arguments; the prepared job sets `SLURM_EXPORT_ENV=ALL`
  for workers. Host-memory requests can increase billed cores; use the profiles in VSC.
- **Acceptance reports:** GPU contracts and self-hosted summaries must retain the full runtime
  settings hash. A checkpoint/version string alone cannot identify compatible evidence.
- **Cleanup aliases:** reject linked output roots before resolving them, and linked figure
  paths before clearing them. Routine report cleanup retains logs and personal notes;
  an explicit full-output reset deletes them. Permanent feature stores remain protected.
- **Interpreter shadowing:** a JEVPFN Conda prompt did not prevent pip from using another
  project's inherited venv. Use the shared activation helper and its explicit verified
  interpreter; repair any affected external environment only from its own requirements.
- **Optional imports:** pip check missed undeclared Jinja2 and absent notebook extras.
  Token-audit now requires Jinja2 and checks every task format before full counting.
  VSC excludes only local notebook tests; the complete local suite retains them.
- **Audit fingerprints:** tokenization.py and decision_tokenization.py are frozen implementations
  used by ongoing audits. Retain their byte identities; filter the active roster outside them.
  Their historical availability constants are not active model selection. Never substitute a
  base tokenizer or character estimate for a complete publisher prompt.
- **Protected data:** earlier recursive empty-folder cleanup deleted raw data; all 20 pinned
  datasets were restored. Never recursively clean raw/prepared inputs, permanent caches or
  weights. Cleanup tools must not target a temporary directory's parent/system TEMP root.
- **Concurrent notebook reports:** Windows sharing violations require atomic replacement,
  bounded retries and recovery copies. Readers/writers share OS locks; keep the report lock.
  Execute real kernels and persist rich/partial .ipynb outputs; preserve edits made during a run.
- **Checkpoint imports:** publisher Python imports must not create bytecode inside the
  fingerprinted checkpoint tree. Disable bytecode before imports; retain strict file hashes.
- **Fresh VSC runs:** the owner requests deletion, not archival, of prior generated output.
  Use fresh_start with --delete --include-archives; never clean data or checkpoints.
  Maintenance refuses active JEVPFN jobs and requires stopped manual writers.
  Question uploads preserve file bytes and validate every member before replacing old files.
  The 15-10-2026 delete-mode follow-up passed all 440 tests, including 19 reset tests.
  The owner's first real deletion then exposed Windows read-only directories (WinError 5),
  absent from those tests. Reproduced the failure, added Windows-specific attribute recovery
  and bounded lock retries, and completed the actual local deletion. All 80 raw/prepared/cache
  files retained identical hashes. After isolating a synthetic retry test's logs, all 448
  tests passed and the real output folder remained empty. No VSC cleanup was performed.
- **Figure verification:** an inline overview overlapped despite a correct PDF. Separate
  table/header/footer axes and inspect notebook-kernel output, not PDF alone.
- **Artifact completeness:** checking only trace groups present in a file missed an entirely
  deleted group. Require every expected text group and every original row; failure is not
  missing text. Input hashes, class order and provider/checkpoint identities must match.
- **Transfer recovery:** merge validated dataset stores transactionally with backups;
  never overwrite populated caches. Preserve conflicts, bundles and historical spending rows.
- **Test isolation:** clear ambient phase/job variables in synthetic fixtures; otherwise local
  tests can inherit VSC routing. Real host checks happen outside synthetic fixtures.
- **Scope of evidence:** approximate character screens flagged inputs that fitted the first
  three exact tokenizers. Current local counts and VSC results have separate provenance.
  GPU contracts/predictor fits/full-fold resources require real allocation evidence; synthetic
  tests or CUDA tensors alone do not establish compatibility.

Historical change details belong in CHANGELOG/Git, not duplicated status narratives here.

Independent review verified recovery/cleanup/resource gaps. TabICL's many-class default was
already true; it is now explicit. Rounded documentation examples do not prove rounded live
Jev responses. Keep strict decoding until a real precision probe supplies evidence. Known-answer
probe failure requires inspecting the saved raw response; it is not automatically a decoder bug.
