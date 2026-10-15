# Operational evidence

Current stage: Experiment 0A. Current-input token audits and Clef/Intern real GPU contracts
passed. The CPU suite requires a rerun after a Linux test-fixture repair; the two-node
storage job has not run. Full extraction and predictive runs remain disabled. Larger pilots,
three remaining GPU runtimes, Jev live checks, storage quota and independent backup remain
unverified. The read-only library pin is
`de324efcca55be392feb0c01830b3b6ce364587f`.
Method constraints are in README; operations are in VSC; source meanings are in DATA_SOURCES.
Latest local verification (15-10): 470 tests, lint and dependency consistency passed, including
the installed TypeSafe SDK with mocked successful and rate-limited HTTP responses. This is
not live API evidence. No notebooks, input files or frozen tokenizer code changed in this review.

## VSC runs

| Date | Cluster / job | Configuration | Verified outcome |
|---|---|---|---|
| 15-10-2026 | wICE / 62328904 | experiment_0_cpu.slurm | Dataset/hash/CV checks passed; contract suite failed: 442 passed, 2 cleanup-fixture failures, 9 skipped, 3 local notebook tests deselected. Fixed locally; remote rerun pending. |
| 15-10-2026 | wICE / 62329058 | experiment_0_tokens.slurm | Version-2 report: counted_all_inputs for all five providers, 18 datasets and 1,698,127 unique inputs each; zero overflows. |
| 15-10-2026 | wICE / 62332201 | Clef GPU contracts | gpu_contracts_passed_requires_review: 74 forwards, 7 known-answer probes and 36 real inputs across 18 datasets; runtime token parity; A100 80 GB; peak allocated 18.52 GiB. |
| 15-10-2026 | wICE / 62332327 | Intern GPU contracts | gpu_contracts_passed_requires_review: same 74-request coverage and parity; A100 80 GB; peak allocated 9.32 GiB. |
| 14-10-2026 | wICE / 62316611 | experiment_0_cpu.slurm | COMPLETED, exit 0, 1m25s; MaxRSS 1,065,032K. Owner supplied sacct after the dependency/test-scope repair. |
| 14-10-2026 | wICE / 62316685 | experiment_0_tokens.slurm | COMPLETED, exit 0, 1h14m26s; MaxRSS 4,868,592K. All five providers counted 1,698,127 unique inputs each; zero overflows. Historical version-1 questions; repeat after version-2 preparation. |
| 14-10-2026 | wICE / 62312865 | experiment_0_cpu.slurm | FAILED, exit 1, 1m30s; 309 tests passed, 3 notebook tests lacked IPython/nbformat, 1 platform skip; dataset checks passed. |
| 14-10-2026 | wICE / 62312903 | experiment_0_tokens.slurm | FAILED, exit 1, 1h06m02s; Perplexity lacked Jinja2; Liquid/Strands completed counting; committed databases survive. |

Submission/RUNNING are not successful completion. Do not update a checkout/environment while
jobs use it. Record final status from sacct and logs. Account balance is not storage quota.

The owner downloaded the DATA-only `output_JEVPFN.zip`, SHA256
`8d3fb261ea0af0fd5c869efe319ea8bfdb4a607309f18514b470682f2882e1f9`.
It contains reports/logs, not PROJECT token databases, sampled real responses or synthetic
GPU feature artifacts. Do not import it as a validated transfer bundle. All 18 local question
file hashes match its version-2 token report. Current token totals are Liquid 703,871,352;
Strands 1,001,295,131; Clef 1,010,144,410; Perplexity 826,699,787; Intern 1,176,671,920.
These include complete publisher prompts and are not Jev billing counts. Historical version-1
counts do not authorize current inputs. Frozen tokenizer modules remain byte-identical.

The 15-10 setup receipt confirms access checks passed; upload receipt
`20261015T114600Z_88790052` confirms all 18 inputs installed. Clef/Intern GPU reports retain
runtime settings hashes; reference-kernel fallback is correct but needs sustained-throughput
profiling before full extraction. Successful short probes do not establish campaign duration.
At 16:51 on 15-10, the owner's `sacct --starttime 2026-10-15 --name=jevpfn-exp0-storage`
returned no jobs, matching the missing storage report. Submit that check after the CPU rerun.
An earlier unauthenticated TypeSafe HEAD returned 405: reachability only, not live API evidence.

## Reusable failure lessons

- **15-10-2026 storage setup:** the owner's VSC transcript at commit `08ff22bd` confirms the
  environment check and DATA/PROJECT output deletion succeeded. Setup then failed because it
  tried creating a temporary file in `data/raw`. Existing raw inputs need directory-read access,
  not writes; actual file/hash checks belong in CPU preflight. Setup now separates that check
  from required writes to mutable destinations. At `cdc1d56`, the raw check passed but writing
  to `data/decision_inputs` failed. The owner's `getfacl` output confirms owner `r-x` and ACL
  mask `---`; the parent, cache and checkpoint directories are owner-writable. Add only owner
  write permission to that exact input directory (`chmod u+w`), without recursion or altering
  the group/ACL mask, then repeat setup. The upload installer needs that write access. The
  repair and subsequent installation are now confirmed by the downloaded receipts. Do not
  repeat deletion or installs. Setup deliberately verifies installation-time write access;
  compute jobs read the prepared inputs.
- **Linux cleanup fixtures:** fd-based shutil.rmtree attaches a filename to raised OS errors.
  The two failed tests constructed exceptions without errno/strerror, so this rewrote their
  display text. Use realistic OS exceptions; keep assertions that POSIX errors propagate
  without chmod/retry. The downloaded failure did not demonstrate a production cleanup bug.
- **Jev policy revisions:** an immutable first-use allocation blocked reviewed pilot-to-full
  budget/price/bound changes. Versioned approvals now retain cumulative spend and history;
  in-flight/ambiguous reservations block revisions. Never hand-edit a paid cache to reset it.
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
  The first real deletion exposed Windows read-only directories (WinError 5). Recovery now
  clears only validated output attributes and retries brief locks. The subsequent local
  deletion preserved all 80 raw/prepared/cache file hashes. Historical test counts remain
  in CHANGELOG; they describe different revisions, not the current VSC suite result.
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
