# Operational evidence

## 21-10-2026

Owner's newer Downloads output_JEVPFN.zip: 497,502 bytes, modified 09:16:28. Token job
62378350 completed all five providers over 18 datasets / 1,698,127 unique inputs each,
with zero overflows. Clef 62378354 and Intern 62378432 passed actual container GPU
contracts (106 and 102 dispatches respectively); the owner's guarded retire commands
removed 55,640 and 55,442 native entries. Quota after retirement has not yet been supplied.
Do not repeat those installations. Intern's former pending state is superseded.

Liquid build 62378491 FAILED: packages went to /data/leuven/383/vsc38338/conda_envs/env1;
the image expected /LOCAL_TYKKY_SsbThB9/miniforge/envs/env1. Pip/tokenizer checks succeeded
outside the image, but the wrapper warned "Created env is empty" and sealing found no
containers/liquid/bin/python. The build now overrides CONDA_ENVS_PATH inside the pre-install
hook and asserts the Python prefix before pip. Remove only this unsealed image with the
new preview/apply command before retrying. The accidental DATA env1 needs separate inspection;
do not remove the CPU JEVPFN environment. Strands/Perplexity/E5 were not reached by that loop.

Real-text sensitivity is NOT an invariance pass: Clef's binary example moved from 0.304
to 0.867 under injected instructions; Intern's multiclass option-order probe changed its
top answer (aligned total variation 0.647). Keep frozen questions/order and review these
limited diagnostics before extraction; they are observations, not automatically decoder bugs.

The local exhaustive readonly question audit passed 18 datasets, 1,698,127 unique inputs
and 3,294,340 row/slot links; file hashes before/after match. Reviewed all 18 documented
task sentences; no wording change. V3 and exact tokenizer fingerprints remain untouched.
The local setup failure was the literal "$0.042" passed to float(); numeric prompts now
accept currency notation and reprompt. Owner confirmed organization console limits of
80 RPS / 100,000 TPS at console.typesafe.ai/rate-limits. Public models/errors docs specify
32k state-plus-question / 64k request and 400 max_tokens_exceeded. Setup now uses this
server-enforced policy with a separate 32 MiB HTTP-body check; it does not claim exact
local tokenization or turn on long context. No paid calls made during repair.

PCA16 is an optional E5-only ablation across all six predictors; PCA32 remains main.
0B now checks every complete E5 artifact and main predictor fixture paths with both PCA
dimensions/combined decision features. Actual predictor runtimes/full-fold profiling remain
outstanding. Current checkout at start: 375c3e5; all three committed library links matched:
TFM cbaff2f5fa0a6e910d7e7d1cd148437a3e0c217a,
DM a0e40a60e7f3be4a791b0063046767633641ecc4,
VSC 9e376b5ee9106e82d37eca86ca6acb861f580c93.
Finish runtime setup and local pilot before the owner's final clean/frozen 0A campaign.
Local validation completed: 698 passed, two skipped (local PyTorch and POSIX-only link
behavior); lint, pip check, 18 shell syntax checks and library-pin checks passed. No
notebook/visualization source changed. Liquid's corrected build and the expanded 0B
predictor contracts still need execution in their VSC environments.

## 20-10-2026

Latest supplied VSC terminal: checkout 6a12e8f, setup write checks passed. Token job
62372655 was submitted at 15:29 and still PENDING at 16:03, with zero elapsed time.
This is scheduler waiting, not a tokenizer failure. Do not submit an overlapping audit.
Clef: sealed container, 55,640 native entries still present. Intern: 55,442 native
entries, no container. Liquid: 119 partial entries, no Python/container. Strands and
Perplexity: no native environments/containers. Shared PROJECT quota: 147,540/150,000
files and 253.1 GB/1 TB. No retirement is established by this output.

Jev synthetic robustness made and saved all 15 responses. The final score response
contained score=7.99 and a returned probability vector implying 8.00; strict 1e-4
consistency rejected it. The decoder now permits only a feasible two-decimal rounding
envelope for Jev (an explicit compatibility policy, not a provider guarantee), retaining
the raw response and deriving the feature from the returned probabilities. Offline-only
revalidation succeeded: 5,336 input tokens, USD 0.000224112; no new requests. Largest
synthetic TV distance was 0.11 on the injected binary input; top options were unchanged.
This is not evidence of robustness over real datasets.

Review repairs: resolved-path comparison in fold completion; serial isolation of failed
open-weight batches only in explicit continue mode; presence-only controls for all six
predictors; 13 label-blind real-text sensitivity probes with canonical option alignment;
PCA training explained-variance records. Source questions/tokenizer files are unchanged.
The GPU contract ceiling is now 110 (at most 106 planned). Local real-text diagnostics
remain behind the reviewed context envelope; no live execution occurred during repair.
The complete local test suite passed: 667 passed, two skipped (local PyTorch absent and
POSIX-only symlink behavior). All five notebooks completed and saved 24 figures; lint,
dependency consistency and all 17 shell syntax checks passed. Actual new GPU/real-text
provider checks remain outstanding.

HEAD 6a12e8f contains neither the old nor new DM gitlink. `.gitmodules` points to
`dm-library`, whose local untracked checkout is c4040baad01db55976a2eda22257cdcfd599f5e5.
The owner must stage and publish that gitlink; the agent must not stage/commit it.
`src.utils.check_libraries` now detects missing committed links and checkout drift
before local 0A / VSC CPU tests. Prior notes about the old index entry are historical.

Implementation after the read-only review: evaluation now stores Parquet prediction payloads,
metrics and fold-completion records inside one SQLite file per dataset/phase on PROJECT.
DATA gets one locator per dataset; fixed fold-lock stripes bound file growth. Transfer merges
these stores transactionally; legacy file readers remain. Source/input/runtime fingerprints
and optional exact 0A campaign/job selection now bind CPU/storage/GPU/pilot acceptance.
New synthetic prompt-robustness tests require explicit review. Jev's separate bounded
15-attempt diagnostic has permanent accounting outside output (maximum USD 0.04032 under
the configured bounds, before any explicit auth-rejection retry reservation).
No live calls were made in this implementation turn.

E5 batches across texts/chunks; all decision/E5 arms share text-missing indicators.
Uncertainty views derive entropy/peak/directional variance rather than comparing unlike
provider confidence fields. Clef/Perplexity batch two inputs, with serial parity required
in GPU contracts; other providers remain serial. Their new adapter/settings identities
require new GPU/pilot evidence. Both fingerprinted tokenizer modules are unchanged.
Container inventory/guarded replacement retirement covers obsolete partial installations;
all eight predictor runtimes can also be wrapped once separately configured. Their retirement
requires matching predictor contracts. New installations test package-index connectivity first.

Local validation: 660 tests passed, two skipped (PyTorch absent locally; POSIX-only link test).
All five notebooks executed successfully and saved 24 figures. Dependency consistency and
lint passed. Live VSC runs, native E5 batching/decision batch parity, actual quota reduction
and Jev prompt diagnostics remain unverified. No output, environment, question store, model
weight or paid feature was deleted; no submodule, Git index, branch or remote was changed.
Before the owner's 6a12e8f commit, the DM rename was unstaged and the old gitlink was still
in the index. That commit removed it but omitted its replacement, as recorded above.
No automatic cleanup or paid inference should follow a
local test pass alone. Finish runtime setup/retirement, then perform the requested clean 0A.

- Added DM Library as an independent read-only submodule for this project only; canonical maintenance and pin propagation run from `0. Literature`. No model settings changed.

- Added the independent read-only VSC Library submodule and daily pin-update proposals; shared HPC guidance moved out of TFM Library, with project instructions retained in docs/VSC.md. No cluster workload was submitted.

Current stage: Experiment 0A setup, before the owner-requested final clean validation.
Newest Downloads `output_JEVPFN.zip` (383,820 bytes; local modified 19-10 15:05)
contains token jobs 62370453 and 62370457. Both failed because they ran concurrently
against the same PROJECT count databases: `UNIQUE constraint failed: counts.dataset_id,
counts.input_no`. Job 62370453 completed Clef, Liquid, Strands and Perplexity; job 62370457
completed Intern. The combined report and all five individual reports cover 18 datasets
and 1,698,127 unique inputs each, with zero overflows. Maximum tokens / context limits:
Clef 3,140 / 16,384; Liquid 3,042 / 32,768; Strands 3,169 / 4,096;
Perplexity 3,083 / 8,192; Intern 3,276 / 8,192. This downloaded DATA archive contains
summary evidence, not the large PROJECT databases; verify those on VSC by one resumed audit.
The local fix adds an exclusive coordinator lock, per-database writer locks and serialized
report merging. Existing tokenizer fingerprints/input identities are unchanged, so matching
counts are reused. Do not clean output, rebuild containers or regenerate questions now.
Local validation: token/GPU tests passed (25). Full suite: 648 passed, one POSIX-only
skip and one 30-second Windows Git Bash timeout in the packed-runtime fixture. The same
test passed unchanged in 1.28 seconds; the complete shell suite then passed 36 tests with
one POSIX-only skip. No shell code was changed for the non-reproduced timeout. Lint,
formatting, dependency consistency and diff checks passed. VSC validation of the new
concurrency guards remains pending; no remote jobs, inference or deletion were performed.

Build 62369713 succeeded: wrapper installation completed, source-only sentinel was
invisible, package equivalence passed, and container
a701323896f9554825df71b5d9eec617abcda30704142b21fd846aafd2a0a40d was sealed.
GPU job 62370298 selected that packed runtime and passed the CUDA tensor check on an
A100 80 GB, then stopped because DATA `before_extraction/token_audit.json` was absent.
No Clef model inference was reached. That missing-report diagnosis is superseded by
the new token reports above; retry the read-only `gpu_contracts --check-inputs` gate after
the resumed audit passes, then only the GPU contract job. Do not retire its native source yet.
Downloaded `jevpfn-container-build_62369139.log` (19-10, 1,042 bytes) shows copying/setup
progressed to SquashFS creation, then failed with `mksquashfs: command not found`.
No sealed image, GPU pass or source retirement is established. The terminal's missing
62369106 log refers to a different job. The owner's corrected `module spider squashfs`
found no matching module; `apptainer buildcfg` reports 1.4.4-1 and LIBEXECDIR=/usr/libexec.
The owner then executed `/usr/libexec/apptainer/bin/mksquashfs -version` successfully
on the login node: version 4.6.1. No additional package installation is needed there.
The build now checks its tools before copying and discovers Apptainer's documented bundled
packer when absent from PATH. Build 62369713 subsequently confirmed this repair on VSC;
the newer failure described above concerns missing token evidence, not packaging.
Local validation of this prerequisite fix: 642 tests passed, one POSIX link test skipped;
lint, dependency consistency and all 17 shell syntax/LF checks passed. No paid inference,
remote submission, output deletion, question regeneration or Git publication occurred.
Local validation after the template/cleanup refactor (19-10): 634 tests passed, one POSIX
link test deferred to VSC; five notebooks/24 PDFs passed execution and visual inspection;
all 18 input stores passed the exhaustive audit without modification. Local data/CV
preflight, dependencies, lint and 16 shell syntax/LF checks passed. No remote job or paid
inference was performed. The current code is not yet verified by a fresh VSC 0A campaign.
The new 0A coordinator submits one job at a time, preserving IDs across login disconnects;
only matching source/runtime sequences resume. Cleanup preserves smoke spending/response
records outside output. Do not reset outputs until container setup works.
Downloaded `output_JEVPFN.zip` inspected on 19-10 (313,493 bytes; modified 11:44).
Earlier log 62368266 stops with FileNotFoundError at the private wrapper's
`frontends/wrap-install` alias replacement. Some site installations lack that alias.
The local fix now tolerates an absent alias and creates the checked relative link;
tests cover absent and existing aliases. This failure is superseded by build 62369713.
The ZIP also retains 62366728's earlier ACL-preserving copy failure (FAILED 1:0,
6m49s, MaxRSS 13,924,368 K). The private copy-template fix avoids source ACLs and
ownership; no shared installation or source environment is modified. Neither log
shows a sealed image or matching GPU tests. Do not infer cancellation from the
OnDemand disconnection, and do not reset output until runtime setup succeeds.
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
Current responses and the spending ledger live permanently in `data/jev_cache/_control/jev_smoke/`;
Experiment 0 holds rebuildable copies. Cleanup migrates legacy output records before deletion.
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
Compact runtime build/verification/guarded retirement and source-freeze checks are implemented.
Clef's build/seal and CUDA check passed; model contracts remain pending. Preserve native sources until
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
| 19-10-2026 | wICE / 62370298 | Clef packed GPU contracts | Packed runtime and A100/CUDA tensor preflight passed. Contract report failed on missing DATA token_audit.json before model loading; no inference results. |
| 19-10-2026 | wICE / 62369713 | Clef compact wrapper build | Downloaded log confirms installed/sealed image, package equivalence and hidden source sentinel. Wrapper installation reports 431 seconds; scheduler elapsed/exit not supplied. |
| 19-10-2026 | wICE / 62369139 | Clef compact wrapper build | Attached log reaches SquashFS creation, then fails because mksquashfs is absent from PATH. No image seal/GPU pass. Scheduler exit code not supplied. |
| 19-10-2026 | wICE / 62368266 | Clef compact wrapper build, 20bf5e0 | Downloaded log: FileNotFoundError at missing private frontends/wrap-install alias; build/GPU validation not reached. |
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
  Use output_cleanup with --include-archives --apply --writers-stopped; never clean data or checkpoints.
  The older fresh_start --delete interface remains supported.
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
