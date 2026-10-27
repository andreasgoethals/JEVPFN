# Current operational evidence

Updated 26 October 2026. README owns the method, VSC owns commands,
DATA_SOURCES owns dataset meanings and LITERATURE_REVIEW owns research claims.
Older details/test counts remain in CHANGELOG/Git. Submission is not completion.

## Execution decision and storage

The owner selected Mindwell B200 without requesting a new GPFS allocation.
Their shell confirmed /gpfs1/scratch/383/vsc38338 and no /gpfs1/project/stg_00211.
The durable master remains /lustre1/project/stg_00211/JEVPFN. CPU preparation
stages per-runtime native working sets under
$VSC_SCRATCH_GPFS1/jevpfn-work/<runtime>/JEVPFN. DATA keeps logs/manifests.
Committed decision/embedding/evaluation SQLite results are snapshotted/merged to
durable Lustre periodically and at command exit; failures retain scratch for recovery.
Scratch is not a backup. Never stream bulk model/data inputs from Lustre on B200.

Separate blackwell-cu130 images preserve existing cu126 images. Exact pinned
wheel availability is checked before building; absent wheels stop the build.
Startup checks actual B200, native sm_100, BF16 operations and float export.
These checks are not real model acceptance. Both native GPFS and durable Lustre
need two-node lock/snapshot tests; every provider/encoder needs B200 execution.
The final index check on 24 October found the exact Torch 2.12.1/vision 0.27.1
and Perplexity Torch 2.14.0/vision 0.29.0 wheels under cu130, not cu128.
The B200 profile therefore uses CUDA 13.0, preserving package pins, and requires
a live driver-580-or-newer check plus actual GPU kernels before model loading.

No A100 route is allowed. Default jobs no longer use H100. Cancellation checks
live ownership, H100 partition and checkout paths; CPU and other projects stay.
Do not update source used by active jobs. No remote cancellation/submission was
performed during this implementation.

Owner quota report, 26 October 08:44: shared Lustre PROJECT 473.3 GB/1 TB and
40,313/150,000 files; DATA 55,045/76,800 MB and 528k/10M files; GPFS scratch
9.537/500 GB. These are dated observations, not current capacity. Compact
containers addressed the previous ~111k-file native-environment problem.
Never delete working images or other projects to make space.

## Latest downloaded VSC evidence

Inspected Downloads/output_JEVPFN.zip, modified 26 October 2026 21:39:06,
772,588 bytes, 395 members. It contains DATA logs/manifests, not the PROJECT count
or feature databases. Correlate individual receipts with job IDs; the coordinator
stopped at 15:33 and its submitted/pending totals are stale.

| Check | Latest observed result |
|---|---|
| CPU/data/questions, 62431318 | Passed; setup checks, not complete 0A acceptance. |
| Liquid/Strands/Perplexity/Clef/Intern tokens | All ten parts and strict merge passed. Merge jobs 62431739/40/69, 62431826/27; each covers 1,698,127 unique inputs with zero overflow. |
| H2O/SII/Nimble tokens | All ten parts complete for each provider; final merges not submitted. Do not infer complete-audit acceptance until merging validates coverage. |
| Kev/Decision3 tokens | Checkpoint staging and token preparation completed; partition jobs not submitted. |
| Ten Blackwell containers | Setup reused ten sealed images. Presence/sealing is not real GPU validation. |
| Lustre, 62431319 | Failed: srun could not contact wICE's controller; main log also reports controller DNS failure. No locking assertion was reached. |
| Liquid GPFS, 11739854 | Passed on p11c05n2/p11c08n2 with posix_lockf: competing writer blocked, acquired after release, snapshot restored. Quota/independent backup remain unverified. |
| B200 contracts, pilots and encoders | Not yet executed in this setup; no current B200 inference pass. |

Owner sacct at 21:40 confirmed GPFS job 11739854 COMPLETED 0:0, submitted at
15:33:52 despite the preceding sbatch timeout. Its report is
storage_contracts/fd0c26ce4cea421eb0620598c7a2a035.json, source fingerprint
a1575b2fdd12840d6f282c2d6aa053086b75ed61c01f4dbb2af526ad71778547.
The launcher receipt still has liquid_gpfs_storage starting without a job ID.
The new setup_recovery command verifies live accounting against the saved command,
user/account/checkout, records that accepted ID and preserves the stopped status.
After all JEVPFN jobs stop, publish/pull source and use the explicit setup restart.
No output reset is needed now; completed token batches and sealed images are reusable.

Older errors in the ZIP are historical: GPFS BSD flock permitted two owners;
POSIX record locking now has the real two-node Liquid pass above. Other runtime
working sets and a new Lustre pass remain gates. Perplexity staging omitted source/uv.lock;
the filter was corrected, but its new scratch stage still needs to run. Tykky link
normalization uses the same rules for sealing and relocation. No guard was bypassed.

Token audits use ten disjoint ranges per provider: up to 40 CPU token jobs, six other
CPU checks/merges, 12 GPU tests, and two jobs in each build/download/transfer pool,
64 active total. Setup has 176-186 steps depending on reused images; frozen 0A has
165. Tokenization modules/registry remain byte-identical, allowing matching committed
counts to seed partitions. Strict merges check complete coverage and identities;
restart databases stay off GPFS and out of result exports. H2O/SII/Nimble's longest
completed counting parts took about 354/285/78 minutes; queueing and remaining work
cannot be predicted from those durations alone.

Historical H100 contracts/pilots passed for Liquid, Perplexity, Clef and Intern,
with source 8c348bdaf743ac676b1b6cd6a6ff8a550d23a64bf7e5ef8b55eab3d511097b95.
Strands then failed its four-decimal rounding validation, subsequently corrected.
Historical E5-small/E5-large/Qwen contracts passed on H100. None certifies B200,
current code or final acceptance. Reports now include source/job/runtime identities.
Earlier detailed run histories remain in CHANGELOG/Git and retained receipts.

## Jev and permanent accounting

Pinned jev-1.13.0 synthetic contracts passed: 1,051 reported tokens and
USD 0.000044142 recorded cost. A prior HTTP401/reservation remains in history.
The later local pilot completed 288 requests across 18 datasets, 238,111 input
tokens, USD 0.010000662, no pending reservations; receipt
hosted/9f844b4ee73c4ece9c3dacff37fb611e.json. These are diagnostic samples;
full extraction remains disabled.

The owner confirmed 80 RPS/100,000 input TPS on 21 October from the console.
Recheck before extraction. Conservative 64k per-request billing reservations
are separate from account per-second limits; do not replace them with an
unverified tokenizer estimate. Timeout does not imply an unbilled request.
Jev two-decimal and Strands four-decimal rounding have separate bounded
projections; preserve raw outputs/scores/usage, never claim recovered precision.
Other providers retain their validated policies.

Credentials stay in the explicitly authorized ignored local secret file.
Never print/upload them. Environment overrides take precedence. Paid responses,
queues, recovery/spending history and smoke state are permanent data; cleanup
preserves them. Unknown outcomes require reviewed recovery, not blind retries.

## Inputs and literature

Question v3 supersedes v2 evidence. Mechanical audit: 665,406 selected rows,
80 text columns across datasets, 1,698,127 unique inputs, 3,294,340 slots,
79,461 empty slots, 1,516,752 reused nonempty slots. All 20 raw datasets remain;
Spotify/Zomato are excluded from extraction/evaluation. Exact counts differ by tokenizer.

All eleven extractors receive the same canonical task/question/text/class meanings.
Native framing, tokens, attention, readouts and confidence fields differ. Shared
numeric schema does not imply equal probabilities. Exclude provenance URLs,
non-text row values and row targets from model text; retain sources separately.
Mechanical checks cannot prove language interpretation or calibrated confidence.

Read-only pins: DM 5f1b3f8f1611681579aba052a86c2626a6a17b96;
TFM 702d9ab0b66f6639ba8d443ff9c4a91231964465;
VSC 3bcdf55e576d10161d6189b40ac39380bbba9eca.
Reviewed sibling manuscript, CreditICL figure recommendations, MulTaBench v2 and
relevant library guides. LoRA TAR is supervised; frozen task conditioning here
does not train on dataset targets. TILE/STRABLE are adjacent work; do not claim
firstness. Libraries are dated selections, not exhaustive current literature.
H2O's 0.801 winning-probability floor is publisher behavior, not measured confidence.

## Validation and next gates

After the recovery helper, the 26 October local suite passed 1,049 tests in
331.55 seconds, with 22 platform/optional-runtime skips. Ruff, dependency and
diff checks passed. The prior 25 October check verified all 28 unchanged Bash/Slurm files. Synthetic/local tests do not establish Blackwell compatibility. Both
historical tokenizer modules and registry remain byte-identical to HEAD.
Earlier validation counts belong in CHANGELOG.

On 26 October, cached local Jev responses were revalidated without network calls:
three synthetic contracts passed; 15 synthetic and 13 real-text robustness cases
decoded successfully and still require semantic review. This does not replace a
fresh source-bound local 0A or re-run the 288-request dataset pilot.

Six notebooks run locally, prioritizing text/reuse, exact questions, feature
widths, saved pilot behavior and paired effects. Missing benchmark results stay
pending. PCA fits training rows only. Across-dataset regression figures use
relative RMSE changes, not averages mixing salary/price/score units. Synthetic
layout data must not be stored as research results.

The 24 October methodological review found combined train/query preprocessing in
the pinned LimiX-2 implementation. The adapter now fits filtering, categorical
inference/encoding and polynomial scaling on training rows only, uses torch.device
and the allocated CPU thread budget, and rejects unreviewed API/multi-GPU paths.
The publisher's RebalanceFeatureDistribution already uses training rows and was
not changed. Real 0B testing remains required. Presence/statistics controls now use
the same empty/nonfinite-text policy as extraction; nonfinite regression targets
are rejected before splitting. Joint-PCA summary curves use matched dataset/split
cohorts across widths, preventing partial completion from changing the comparison.
The 26 October check also aligned embedding input construction with the same
missing-text policy: numeric infinity is missing; literal "inf"/"nan" remain text.
CPU checkpoint staging now accepts CPU Slurm jobs while rejecting GPU allocations;
encoder staging is refused after freeze. Per-column PCA32 and joint-PCA ablations
retain separate training-only projections, including each binary inner partition.

The owner authorized an interim clean debugging rerun once active jobs finish;
use output_cleanup and test_all, without freezing yet. Finish setup before the final
clean campaign. Then stop writers, delete local/DATA/Lustre/GPFS output (including
known runtimes with incomplete staging), regenerate/audit/upload questions, freeze matching source
and run fresh local/VSC 0A. Preserve data/features/weights/images. Acceptance
requires current matching reports and quota/backup/throughput/robustness review.
Create features, run 0B including real encoder/predictor/complete-feature checks,
then benchmark. Full-fold resources and native predictor runtimes remain unproven
until their real gates pass. A source/runtime change requires a reviewed stage
transition, not deletion of mismatch evidence. Publication requires the owner's
explicit authorization; it was given for the current parallel-testing update.

## Reusable failure lessons

- srun stdout is not pure JSON. Use job/probe-bound receipts; briefly retry only
  absent receipts after success, never repeat completed writes.
- Immutable prepared SQLite files must copy byte-for-byte: backup changes headers
  and invalidates token evidence. Use snapshots for mutable caches.
- Merge stores transactionally; preserve conflicts/accounting. Do not overwrite
  populated caches with partial remote copies or reset ambiguous paid jobs.
- Disable checkpoint bytecode. Linux hash reuse includes ctime; Windows rehashes.
  Preserve pinned publisher code and frozen tokenizer implementations.
- Slurm scripts must establish roots/runtime explicitly; submitting exports are
  not inherited automatically. Verify the interpreter, not the Conda prompt.
- Raw inputs need reads; installation needs writes. Fix only a confirmed owner's
  exact blocked directory, never recursively alter ACLs.
- Cleanup rejects links/mount escapes and touches output only. Windows attribute
  recovery must not change POSIX ACLs. Never recursively remove empty data folders.
- Notebooks need atomic writes/shared locks, real kernels, partial-output recovery
  and protection against concurrent user edits.
- Empty text is a separate overlap cohort. Repetition across row folds is an
  interpretation concern; report paired cohort metrics without changing splits.
- Require every expected row/text group and class order. Missing extraction is
  failure, not missing text. A finite passing suite never proves all bugs absent.
