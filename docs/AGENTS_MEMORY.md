# Operational evidence

Current stage: Experiment 0A. Full extraction and predictive runs remain disabled; GPU
contract execution is implemented but no VSC model result has been supplied. Project/data
write access and the CPU preflight are verified. Storage quota and independent backup remain
unverified. The read-only library pin is `de324efcca55be392feb0c01830b3b6ce364587f`.
Method constraints are in README; operations are in VSC; source meanings are in DATA_SOURCES.

## VSC runs

| Date | Cluster / job | Configuration | Verified outcome |
|---|---|---|---|
| 14-10-2026 | wICE / 62316611 | experiment_0_cpu.slurm | COMPLETED, exit 0, 1m25s; MaxRSS 1,065,032K. Owner supplied sacct after the dependency/test-scope repair. |
| 14-10-2026 | wICE / 62316685 | experiment_0_tokens.slurm | RUNNING at 15:24 CEST, 3m24s. Owner verified dependency install and all five tokenizer format checks first. Completion not supplied. |
| 14-10-2026 | wICE / 62312865 | experiment_0_cpu.slurm | FAILED, exit 1, 1m30s; 309 tests passed, 3 notebook tests lacked IPython/nbformat, 1 platform skip; dataset checks passed. |
| 14-10-2026 | wICE / 62312903 | experiment_0_tokens.slurm | FAILED, exit 1, 1h06m02s; Perplexity lacked Jinja2; Liquid/Strands completed counting; committed databases survive. |

Submission/RUNNING are not successful completion. Do not update a checkout/environment while
jobs use it. Record final status from sacct and logs. Account balance is not storage quota.

## Reusable failure lessons

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
