# Operational evidence

Current stage: Experiment 0A setup, before the owner-requested final clean validation.
Latest direct OnDemand check (19-10): Clef container build 62366728 FAILED 1:0 after
6m49s, MaxRSS 13,924,368 K. The log shows repeated `cp: preserving permissions: Invalid
argument` errors during the wrapper's environment copy, before image sealing or GPU tests.
This is not evidence of OOM. Prepared a private scratch copy of site wrapper 0.4.2 whose
known wrap template copies content/links/timestamps without native ACLs or ownership.
The site installation and native environments are untouched. Unknown templates stop the
patch; source package identities and real GPU checks remain required. The new build must
still pass on VSC. Keep its failure log and do not reset outputs yet.
Earlier build 62346143 FAILED 1:0 after 4m48s; its log was deleted by an owner reset and its
cause remains unknown. Local preparation/audit and the uploaded v3 bank are unchanged;
VSC installation previously reported already_current. No repeated upload is needed now.
Direct VSC comparison found identical HEADs but 52 differing file hashes, all explained by
Windows CRLF versus Git/Linux LF. Freeze schema v2 hashes normalized text source without
rewriting files; shell scripts still require actual LF. Content changes still stop execution.
The old local freeze was explicitly finished as a debugging restart, not acceptance of 0A;
the prior receipt is retained. Publish fixes and finish runtime setup before another freeze.
Question content/identities and its uploaded bundle are unchanged; do not repeat that upload.

Question v3 is frozen; all 18 files contain 1,698,127 unique inputs and 3,294,340 original
row/text-slot links. CPU 62343023 and exact-token job 62340824 completed successfully.
All five open tokenizers found zero overflows; counts are below. The two-node storage test
passed. All five open checkpoints are staged, but only Clef/Intern native GPU execution
with older questions is verified. Remaining GPU/throughput tests and backup review are pending.
The library pin is `de324efcca55be392feb0c01830b3b6ce364587f`.

On 16-10 the owner confirmed USD 5 account funding. Three live local Jev 1.13 probes passed
again on 19-10 after fixing credential setup:
binary p=0.97, three-class correct option p=1, nine-level score expected direction +4.
Reported input usage: 288 + 355 + 408 = 1,051 tokens; computed input cost USD 0.000044142.
Current responses and the passed receipt are saved locally under Experiment 0 `jev_smoke`.
An initial owner-reported HTTP 401 was explicitly retried after the exact supplied key
authenticated via GET models. Its original diagnostic and USD 0.002688 conservative
reservation remain in authentication_rejections/history; no refund or unbilled outcome is
asserted. A repeat reused all three verified responses without creating a network client.
No dataset requests or full extraction ran. Unknown outcomes stop rather than silently
dispatch again. At the owner's explicit request the key is now saved only in ignored
secrets/typesafe.local.json. Local commands load it; tests isolate it; transfers and VSC
never use it. An explicit environment value takes precedence. Published model
IDs are accepted even when GET models lists aliases only; live returned version was 1.13.0.
The visible console confirmed USD 5 credit and auto-recharge off. Organization ID is under
user menu → Settings → Organization; it was not saved in public files. Home, Organization,
Billing and Usage did not display RPS/TPS. Those remain organization-specific and unverified. Full dataset dispatch retains its
account/context/budget gates; the separate three-call contract has a documented USD 0.008064
maximum at 64k request tokens and USD 0.042/M, no long-context header and no retries.

Latest reported PROJECT quota: 252.5 G / 1 T; 147,339 / 150,000 files. Breakdown: JEVPFN
113,370, CreditPFN 29,019, TabPFNCredit 4,789, CreditICL 160. JEVPFN's unpacked Clef/Intern
runtimes contribute about 111k files. Liquid installation rolled back at ncurses; the loop
never reached Strands/Perplexity. `hpc-container-wrapper/0.4.2` was directly checked in the owner's OnDemand shell: it loads
without prerequisites; wrap-install/conda-containerize and /usr/bin/apptainer are present.
The module does not export CW_GLOBAL_YAML; use its default_config/config.yaml. An earlier
lookup failed only because `0.4.2.` had an extra final dot.
The same read-only OnDemand check directly confirmed nonempty checkpoint manifests for
Clef, Intern, Liquid, Strands and Perplexity on PROJECT. Build/load checks still verify every
checkpoint file hash; a present manifest alone is not a successful inference test.
Compact runtime build/verification/guarded retirement and source-freeze checks are implemented,
but real VSC container builds/GPU execution remain unverified. Preserve native sources until
matching container GPU evidence passes; build unpacked temporary files in scratch.
Local credential/recovery validation passed 609 tests before the wrapper-copy workaround;
the final combined-suite result is recorded in CHANGELOG. Neither protected tokenizer module
nor the library submodule changed. No VSC job submission, cleanup or Git publication was
performed; the current VSC failure and installed wrapper source were inspected read-only.

The owner wants setup/debugging completed FIRST, then permanent deletion of local and both
VSC output trees, repeated local question preparation/audit/upload, and one fresh frozen 0A.
Preserve raw/prepared data, paid responses/features, weights, runtimes and images. Do not
clean yet while runtime preparation is unfinished. A source/config/runtime change after
freezing stops jobs; record operational stage-enabling changes separately from source fixes.
Finish a reviewed stage with `freeze --finish-stage --reviewed-by Andreas --writers-stopped`
before changing its configuration/pins; the tested receipt is retained. Freeze the next stage
again rather than deleting prior 0A evidence. Completion does not sign test acceptance.
Research method is in README, operations in VSC, and meanings in DATA_SOURCES.

## VSC runs

| Date | Cluster / job | Configuration | Verified outcome |
|---|---|---|---|
| 19-10-2026 | wICE / 62366728 | Clef wrap build, 09f7982 | FAILED 1:0 after 6m49s, MaxRSS 13,924,368 K. Site wrapper cp -a failed preserving permissions while copying the native environment. No seal/GPU pass or retirement. Scratch-local copy-template workaround prepared, not yet executed on VSC. |
| 16-10-2026 | wICE / 62346143 | Clef container build, 6f375ab | FAILED 1:0 after 4m48s on l33c25n2. Original stdout was deleted by the subsequent output reset; cause unverified. Empty destination remains; no sealed container or source retirement verified. |
| 16-10-2026 | wICE / 62343023 | CPU at 606296c | COMPLETED 0:0, 5m19s; preflight_passed and synthetic contracts passed; MaxRSS 5,068,440 K. |
| 16-10-2026 | wICE / 62340719 | Fresh version-3 CPU job, 0113f19 | Owner sacct: COMPLETED 0:0, 5m19s, MaxRSS 5,010,580 K. Full report not yet downloaded. |
| 16-10-2026 | wICE / 62340794 | Fresh storage job | Owner sacct: COMPLETED 0:0, 59s; both worker steps completed. |
| 16-10-2026 | wICE / 62340824 | Fresh version-3 token audit | COMPLETED 0:0, 1h17m14s, MaxRSS 7,499,032 K. All five providers counted 1,698,127 v3 unique inputs; zero overflows. Totals: Clef 1,014,192,144; Liquid 707,094,152; Strands 1,005,342,865; Perplexity 830,747,521; Intern 1,180,719,654 tokens. These are not Jev billing counts. |
| 16-10-2026 | wICE / 62338536 | experiment_0_storage.slurm | storage_contracts_passed_requires_quota_and_backup_review; nodes r33c25n1/r33c30n4; held-lock exclusion, release/reacquisition and SQLite snapshot/restore passed. Released receipt became visible after 29.509 s. Quota/independent backup remain unverified. |
| 15-10-2026 | wICE / 62333801 | experiment_0_storage.slurm at 39a39ff | Coordinator FAILED on immediate released-receipt lookup. Both workers returned valid matching receipts: held-lock exclusion and acquisition after release succeeded on r33c22n3. Receipt later readable; snapshot/restore not reached. |
| 15-10-2026 | wICE / 62333525 | experiment_0_cpu.slurm at aaf2153 | preflight_passed; all 18 dataset/hash/CV checks passed; 457 tests passed, 10 expected skips, 3 local notebook tests deselected. |
| 15-10-2026 | wICE / 62333541 | experiment_0_storage.slurm | FAILED with JSONDecodeError while parsing the first worker's stdout. No cross-node storage pass established. Worker stdout was not retained, so its exact contents are unknown. |
| 15-10-2026 | wICE / 62328904 | experiment_0_cpu.slurm | Dataset/hash/CV checks passed; contract suite failed: 442 passed, 2 cleanup-fixture failures, 9 skipped, 3 local notebook tests deselected. Superseded by 62333525. |
| 15-10-2026 | wICE / 62329058 | experiment_0_tokens.slurm | Version-2 report: counted_all_inputs for all five providers, 18 datasets and 1,698,127 unique inputs each; zero overflows. |
| 15-10-2026 | wICE / 62332201 | Clef GPU contracts | gpu_contracts_passed_requires_review: 74 forwards, 7 known-answer probes and 36 real inputs across 18 datasets; runtime token parity; A100 80 GB; peak allocated 18.52 GiB. |
| 15-10-2026 | wICE / 62332327 | Intern GPU contracts | gpu_contracts_passed_requires_review: same 74-request coverage and parity; A100 80 GB; peak allocated 9.32 GiB. |
| 14-10-2026 | wICE / 62316611 | experiment_0_cpu.slurm | COMPLETED, exit 0, 1m25s; MaxRSS 1,065,032K. Owner supplied sacct after the dependency/test-scope repair. |
| 14-10-2026 | wICE / 62316685 | experiment_0_tokens.slurm | COMPLETED, exit 0, 1h14m26s; MaxRSS 4,868,592K. All five providers counted 1,698,127 unique inputs each; zero overflows. Historical version-1 questions; repeat after version-2 preparation. |
| 14-10-2026 | wICE / 62312865 | experiment_0_cpu.slurm | FAILED, exit 1, 1m30s; 309 tests passed, 3 notebook tests lacked IPython/nbformat, 1 platform skip; dataset checks passed. |
| 14-10-2026 | wICE / 62312903 | experiment_0_tokens.slurm | FAILED, exit 1, 1h06m02s; Perplexity lacked Jinja2; Liquid/Strands completed counting; committed databases survive. |

Submission/RUNNING are not successful completion. Do not update a checkout/environment while
jobs use it. Record final status from sacct and logs. Account balance is not storage quota.

The validated combined transfer `20261015T152724Z_9e160b0c` was imported locally; bundle SHA256
`23fe2e574693423a6dbb660d5dcbd5109df840f03fd2e8b26c35fca7f2bda1ae`.
It merged 72 verified real responses (36 Clef, 36 Intern) into 18 permanent dataset stores
and imported 70 outputs. All five exact-token databases are present locally. The real datasets
have no complete feature artifacts yet; the comparison notebook's readiness-only output is
expected. Synthetic GPU artifacts remain under Experiment 0 results. An earlier DATA-only
download lacked these PROJECT files and was not a valid transfer bundle. The former 18 question
file hashes matched the version-2 token report. Historical version-2 totals are Liquid 703,871,352;
Strands 1,001,295,131; Clef 1,010,144,410; Perplexity 826,699,787; Intern 1,176,671,920.
These include complete publisher prompts and are not Jev billing counts. Neither version-1
nor version-2 counts authorize current version-3 inputs. Frozen tokenizer modules remain byte-identical.

The 15-10 setup receipt confirms access checks passed; upload receipt
`20261015T114600Z_88790052` confirms all 18 inputs installed. Clef/Intern GPU reports retain
runtime settings hashes; reference-kernel fallback is correct but needs sustained-throughput
profiling before full extraction. Successful short probes do not establish campaign duration.
The storage test uses per-job/per-probe JSON receipts on DATA and saves launcher stdout
and stderr separately. In job 62333801 the first stdout contains the VSC banner followed by
JSON; the second contains a successful response. Both receipts match the job, node and probe.
The coordinator's immediate lookup falsely reported the second receipt missing. Job 62338536
then passed with the bounded 60-second visibility wait, recording an actual 29.509-second delay
and successful snapshot/restore. Missing receipts still time out; malformed/foreign receipts,
failed steps and permission errors fail immediately. Workers are never relaunched to recover
a delayed receipt. Tokenizer and GPU runtime identities are unchanged; prepared inputs now
use version 3; its token audit passed 62340824 and fresh GPU runtime checks remain pending. The 16-10 08:42 inventory listed only
cloudflare/intern checkpoints/environments; the later staging finished at 09:04 and added the
remaining three checkpoint trees, including the Strands adapter/base. The unauthenticated
Hugging Face warning did not prevent these downloads. Downloads do not establish inference compatibility.
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
- **Slurm worker results:** do not assume captured srun stdout is a pure JSON reply. Structured
  worker receipts need job/phase/invocation identity; preserve stdout/stderr even on failure.
  A successful remote exit does not guarantee immediate local receipt visibility. Retry only
  missing-file reads for a bounded interval; never rerun a worker's completed database write.
  A successful export or notebook run can contain a failed test report; inspect its status.
- **Jev policy revisions:** an immutable first-use allocation blocked reviewed pilot-to-full
  budget/price/bound changes. Versioned approvals now retain cumulative spend and history;
  in-flight/ambiguous reservations block revisions. Never hand-edit a paid cache to reset it.
- **Jev request recovery:** exhausted verified-unbilled rejections and uncertain billing now
  have a reviewed, job-state-bound recovery command. Saved invalid responses can be decoded
  again without paying again. Recovery preserves charges and before/after history; a timeout
  is not evidence that an attempt was unbilled. Synthetic recovery tests are not live billing evidence.
- **Duplicate-text review:** exact repeats may cross row folds; this is an interpretation risk,
  not row-target leakage in extraction. Added conditional metrics on unchanged predictions.
  The independent review's Women's Clothing repeat share included empty inputs; only five
  nonempty rows repeat, while 666 joint inputs are empty. Its regression identity R-squared
  figures were not reproduced by explicit leave-one-out group means on nonempty repeated rows.
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
