# Current operational evidence

Updated 30 October 2026. README owns the method, VSC owns commands, DATA_SOURCES
owns dataset meanings and LITERATURE_REVIEW owns dated research evidence. This file
owns current run state and reusable failures; historical details remain in Git/CHANGELOG.
Submission is not completion. Preserve active or queued checkouts and recovery inputs.

## Current campaign and authority

The owner authorizes autonomous JEVPFN setup, Slurm debugging, reviewed output reset,
fresh 0A, evidence review/acceptance, then full open-weight feature creation. Full Jev
extraction, 0B and benchmarks remain separate later stages. Reviewed commits/pushes have
been authorized since 28 October. Never change another project's jobs, files or runtimes.
On 30 October the owner explicitly selected Jev 1.14 revalidation before a bounded local
first-100-source-rows pilot across all eighteen datasets. Preserve all 1.13 charges/responses.

The VSC DATA checkout remains **73807ee6c193377b99134196e69e6ee9538410be**, frozen source
`eb20d0e666fd1c9a0bed5e12c73b043221f293eaf4ae5ab855e3e1a0178cd2b3`.
The current receipt is
`3. experiment_0/manifests/0A_vsc/20261029T215821Z_8d152b91.json`.
Latest direct SSH: **140 completed, zero failures, two submitted, 50 waiting on dependencies**.
Mindwell contracts 11753859 (Liquid, 2m17s) and 11753882 (Strands, 2m34s) passed:
known-answer semantics, batch parity, runtime/token parity and all eighteen dataset cases.
Their separate pilots are 11755221 and 11755222. The other provider/encoder branches
still need to finish; two passes do not accept the complete campaign. The existing
PROJECT-controller watcher advances ready jobs and resumes their recorded IDs.

Local HEAD advanced to e11a351 through a TEMPLATE-only commit. Byte-identical executable
source/config/runtime pins were verified against the earlier local 0A receipt before its
local-only stage closure. That closure explicitly records the documentation-only ancestry;
it does not claim a test at the newer HEAD. Jev 1.14 source c2c72bfd0a49d26e2446f19df0529897fa1ea4efd3e8914add26027afc0a6b8e
was validated locally at d24b79c: all eight steps in
0A_local/20261030T081306Z_c9a72019.json passed, including 1,163 tests (25 optional/platform
skips), 31 live diagnostics and 288 new dataset pilot calls. The bounded review is
before_extraction/accepted_jev14_pilot.json; full Jev remains unauthorized. It does not modify or retroactively validate the queued VSC source.

Temporary progress monitor: **jevpfn-0a-progress**, every 30 minutes, local host/app required.
Its source/job/owner/completion receipt is `3. experiment_0/manifests/progress_monitor.json`.
It follows descendants in the existing campaign, stays quiet without meaningful change,
never authenticates/submits/cleans, pauses on certificate expiry and is removed after all
tracked work is terminal and evidence assessed. Do not create a duplicate monitor.

## Verified gates

| Evidence | Result and scope |
|---|---|
| CPU image build 62468642 | Passed, 27m02s; sealed PROJECT controller. |
| Controller integration 62469013 | Passed, 3m44s; DATA-to-PROJECT cutover recorded. |
| Input install 62470782 | Passed; eighteen independently validated question stores. |
| Native GPFS 11753505 | Passed; cross-node exclusion/reacquisition and snapshot restore. |
| Basic B200 migration 11753506 | Passed CUDA 13.0, native sm_100 and BF16; not full decision inference. |
| Corrected controller integration 62471351 | Passed, 4m04s; four host/container launch checks, two-node storage, eighteen datasets, 1,139 CPU contracts. |
| Frozen VSC 0A CPU/data/question checks | Passed on source 73807ee. |
| Frozen VSC exact token audits | All eleven providers, all eighteen datasets, zero overflowing unique inputs. |
| Historical bounded B200 setup | Eleven decision runtimes, three encoders and TabSTAR passed on their recorded earlier sources. Not current frozen acceptance. |
| Jev 1.14 local 0A | All eight steps passed on d24b79c; 1,163 tests passed, 25 skipped. Bounded review accepted; first-100-row validation follows. |
| Earlier local 0A | All eight steps passed on source 73807ee with Jev 1.13; superseded for new Jev calls by the 1.14 validation requirement. |
| Earlier notebook execution | Eleven notebooks, thirteen PDFs, saved captions/reports; all visually inspected without clipped/overlapping labels. |

Question v3 audit covers **1,698,127 unique inputs and 3,294,340 row/slot links** across
18 datasets. No true labels/non-text row values enter the decision payloads. Empty text
has no request. Eight unintentionally missing upstream source snapshots were restored from
Git and verified against their catalogued SHA-256 values; the owner did not delete them.

## Storage and access

The actual checkout and Git operations are on DATA. The CPU controller is sealed under
PROJECT/JEVPFN/containers/control-v1, identity
`f04a2aef749aea1e56a397289aab237465a6f9dbc4cd6a8494d500bf305a2b7d`.
The former DATA/conda_envs/JEVPFN prefix is absent; retirement evidence is persistent.
Shared Miniconda and other projects were preserved. GPU dependencies are compact sealed
PROJECT images. Native GPFS holds temporary working copies and mutable recovery stores;
publication merges committed data back into authoritative Lustre PROJECT.

Latest quota observation: PROJECT 636.9 GiB/1 TiB, 47,690/150,000 files; DATA about
44.8/75 GiB; personal GPFS scratch 157/500 GiB across projects. JEVPFN working sets use
about 41 GiB (Liquid 16, Strands 24, other leaves compact recovery data). Preserve these
queued consumers' inputs. The two-resident/200 GiB application budget with 20 GiB personal
headroom avoids the former 416 GiB working-set duplication. Never wipe shared scratch.

The final authorized reset already removed local output and 84.2 GB of VSC output before
the current campaign. It preserved raw/prepared data, caches/accounting, weights and images.
An independent compact local 0A/cache backup is on PROJECT/uploads/local_0A_eb20d0e6.zip,
SHA-256 `ab0e4140ebee5d3d7485f7bf5cd3155f530e4726078d2a2c73abcfae61aa651a`.
The uploaded v3 questions were independently checked and installed before the freeze.

Manual Windows-agent SSH remains usable. The verified transport uses native ssh.exe -F
none -T, strict known-host checks, LF Bash on stdin, exec 2>&1 and explicit start/end/status
markers. The gateway drops unredirected remote stderr; empty successful output is not proof.
Only ask for `step ssh login --context VSC` when actual access needs renewal; no automated MFA.

## Jev accounting and new bounded validation

New work pins 1.14.0 at the documented unchanged USD 0.042/M input tokens, 64k/request,
32k/state-plus-question and 32 MiB body; official references checked 30 October. The owner
confirmed organization 80 RPS/100,000 input TPS. These are account limits, not a reason to
remove conservative per-attempt reservations. There is no exact local Jev tokenizer claim.

The earlier 1.13 dataset pilot used 288 requests, 238,111 reported input tokens and
USD 0.010000662. Fresh-round synthetic/real-text diagnostics added USD 0.000667506.
Keep those ledgers, responses, reservations and checksummed historical acceptances intact.
Real-text diagnostics showed instruction/answer-order sensitivity; neither semantic
consistency nor calibration is guaranteed. New 1.14 responses need their own review.

The extra prefix validation plans 7,643 distinct requests for 1,800 original rows, one
worker at 1 RPS, with a USD 1 cumulative dataset-charge/reservation cap in addition to the
existing private campaign/dataset limits. It skips empty text, reuses exact saved 1.14
requests and stores joint/per-column numeric blocks as validation-only subset artifacts.
Those artifacts are excluded from benchmark inventory and rejected by full-data assembly.
Full Jev extraction remains disabled and owner-run. The Jev 1.14 bounded review passed; prefix progress is in its exact current-source
receipt. Do not infer all 1,800 rows completed from pilot approval. Real-text instruction
sensitivity persisted: probability change 0.48 with a class flip; multiclass option reversal
had total variation 0.07 without a class flip. These limitations were explicitly reviewed.

## Documentation and research scope

The local TEMPLATE equals the canonical 0. Template copy, SHA-256
`9e113ac591a7d702bc61060a3bce703419cb7a296fcf9cf77283c1e5ffd27a20`.
The current template requires single-main-branch operation, temporary bounded progress
monitoring, sealed PROJECT dependencies and protected routine cleanup. Existing numbered
phases/aggregate reports are declared project deviations. Library pins are unchanged;
LITERATURE_REVIEW preserves its dated review pins and scope rather than claiming a refresh.
Active science remains Experiments 1–3 only: decision features, frozen embeddings and
native-text references. No winner-selected follow-ups or lexical baselines were added.

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
- Output cleanup rejects links/mount escapes and touches output only. The separate
  scratch-release helper removes verified working copies as described above. Windows
  attribute recovery must not change POSIX ACLs. Never recursively remove empty data folders.
- Notebooks need atomic writes/shared locks, real kernels, partial-output recovery
  and protection against concurrent user edits.
- Empty text is a separate overlap cohort. Repetition across row folds is an
  interpretation concern; report paired cohort metrics without changing splits.
- Require every expected row/text group and class order. Missing extraction is
  failure, not missing text. A finite passing suite never proves all bugs absent.
