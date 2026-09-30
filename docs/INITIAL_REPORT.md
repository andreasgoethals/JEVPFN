# Initial implementation and audit report

Updated 30-09-2026 after environment verification and further raw-data cleanup. Initial stage only: no Jev
inference calls, model-weight downloads, model fitting, benchmark or Slurm allocation.

## 1. Original repository

The checkout contained 45 tracked paths, including the literature gitlink: a research template
with flat `src/`, YAML config, an example notebook/config, A4 style/FigureSaver, path/storage and
cleanup utilities, a parallel notebook runner, template tests, cluster stubs and empty data/output
markers. README/package/project names were placeholders. No project datasets or models existed.
Useful template conventions were retained; the owner subsequently requested deleting the
inherited initializer files and reorganising config and raw-data folders.

The actual Git root is the inner `JEVPFN/` directory. Git remains on `main`, with public origin `https://github.com/andreasgoethals/JEVPFN.git`.
The template remote and original history are preserved. The TFM Library URL
was repaired and its read-only checkout synced to `81c749bdf17e88b5152f4dc7f2e49bd48e9cc8ba`;
the project records that pin. No library content was edited.

## 2. Created files

The following paths are additions relative to the original Git template. This includes the
previously completed audit code as well as the requested follow-up:

- `config/experiment_0/debug.yaml`
- `config/experiment_1/main.yaml`
- `config/experiment_2/representation_ablation.yaml`
- `config/experiment_3/output_ablation.yaml`
- `config/exploration/audit.yaml`
- `config/exploration/datasets.yaml`
- `config/exploration/template_example.yaml`
- `config/feature_creation/default.yaml`
- `data/jev_cache/.gitkeep`
- `docs/DATA_SOURCES.md`
- `docs/DESIGN_NOTES.md`
- `docs/FEATURE_CREATION.md`
- `docs/GITHUB.md`
- `docs/INITIAL_REPORT.md`
- `docs/RESEARCH_PLAN.md`
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_jev_input_design.ipynb`
- `notebooks/03_feature_creation.ipynb`
- `output_JEVPFN/allresults.md (ignored)`
- `output_JEVPFN/captions.md (ignored)`
- `src/utils/notebook_report.py`
- `tests/test_notebook_report.py`
- `scripts/slurm/_activate_env.sh`
- `scripts/slurm/experiment_0_cpu.slurm`
- `scripts/slurm/experiment_0_gpu.slurm`
- `src/cluster/__init__.py`
- `src/cluster/preflight.py`
- `src/data/audit.py`
- `src/data/metadata.py`
- `src/data/official_types.py`
- `src/data/prepare.py`
- `src/data/upstream/3bb95079a6146ed35d45187e98bcad5b063273a1/core.py`
- `src/data/upstream/3bb95079a6146ed35d45187e98bcad5b063273a1/curation.py`
- `src/data/upstream/3bb95079a6146ed35d45187e98bcad5b063273a1/feat_types.py`
- `src/data/upstream/3bb95079a6146ed35d45187e98bcad5b063273a1/ids.py`
- `src/data/upstream/3bb95079a6146ed35d45187e98bcad5b063273a1/load.py`
- `src/data/upstream/3bb95079a6146ed35d45187e98bcad5b063273a1/nulls.py`
- `src/data/upstream/3bb95079a6146ed35d45187e98bcad5b063273a1/summary.csv`
- `src/data/upstream/3bb95079a6146ed35d45187e98bcad5b063273a1/summary_script.py`
- `src/data/upstream/README.md`
- `src/jev/__init__.py`
- `src/jev/cache.py`
- `src/jev/cost.py`
- `src/jev/features.py`
- `src/jev/inspection.py`
- `src/jev/provider.py`
- `src/jev/requests.py`
- `src/jev/templates.py`
- `src/utils/notebook_display.py`
- `src/utils/provenance.py`
- `src/utils/serialization.py`
- `src/visualize/audit.py`
- `tests/test_cluster_preflight.py`
- `tests/test_feature_creation.py`
- `tests/test_jev_design.py`
- `tests/test_multabench.py`
- `tests/test_phase_config.py`

Ignored local artifacts include the Python 3.12 environment/kernel, 20 versioned raw datasets,
audit CSV/JSON/length arrays, local full request/interactive notebook previews,
resolved environment/config manifests and PDF figures. Real semantic feature tables do not yet exist.

## 3. Existing files changed or removed

Changed relative to the original Git template:

- `.gitignore`
- `.gitmodules`
- `AGENTS.md`
- `LICENSE`
- `README.md`
- `docs/AGENTS_MEMORY.md`
- `docs/CHANGELOG.md`
- `docs/VSC.md`
- `pyproject.toml`
- `scripts/slurm/job.slurm`
- `src/data/loaders.py`
- `src/utils/config.py`
- `src/utils/paths.py`
- `src/visualize/style.py`
- `tfm-library`

Removed/moved tracked paths:

- `_template/README.md`
- `_template/init_project.py`
- `config/example.yaml`

`config/example.yaml` moved to `config/exploration/template_example.yaml`; the two project
configs moved into the same exploration folder. Data preparation moved from `src/utils` to
`src/data`. Eight byte-identical upstream reference files moved out of ignored raw data into
`src/data/upstream/<commit>/`. The 20 dataset folders moved directly under `data/raw`, gained
stable numbers and lost the redundant local `multabench-` prefix. Upstream slugs stay unchanged.

The `_template` files were removed at the owner's request. The later owner-run cleanup command
was retired after reported dataset loss on 30-09-2026. All 20 pinned datasets were restored;
do not follow earlier directory-cleanup instructions. Empty directories do not affect Git.

## 4. Resulting tree

```text
JEVPFN/                         actual Git root; inside the outer project folder
├── config/
│   ├── exploration/            audit.yaml, datasets.yaml, template_example.yaml
│   ├── feature_creation/       default.yaml; shared artifacts, independent of experiments
│   ├── experiment_0/           debug.yaml; VSC only
│   ├── experiment_1/           main.yaml; proposed main comparison
│   ├── experiment_2/           representation_ablation.yaml; proposed
│   └── experiment_3/           output_ablation.yaml; proposed
├── notebooks/                  three project notebooks + preserved synthetic example
├── src/
│   ├── data/                   loading, official definitions, audits, preparation
│   │   └── upstream/<commit>/  immutable reference source snapshots, never executed
│   ├── jev/                    templates, requests, cache, budget, provider preview, features
│   ├── cluster/                credential-free compute-node preflight
│   ├── utils/                  paths, config, provenance, notebook runner
│   └── visualize/              shared style, PDF figures and audit plots
├── data/                       ignored except directory markers
│   ├── raw/                    01_fake_job_posting/ ... 20_zomato_restaurants/
│   │   └── <number_name>/      data.csv, metadata.json, download.json
│   ├── processed/              disposable caches
│   └── jev_cache/              reserved for future responses and feature artifacts
├── output_JEVPFN/              reports and captions; phase-specific figures/logs/manifests/results
├── checkpoints/                ignored future weights
├── scripts/slurm/              environment activation + experiment-0 CPU/GPU checks
├── tests/                      deterministic unit and contract tests
├── docs/                       design, data sources, plans, VSC/GitHub guides and logs
└── tfm-library/                read-only Git submodule
```

Full setup and storage explanations are in [README](../README.md). Config folders now separate
exploration, feature creation and experiments 0–3. Examples, base tests and shared plotting/path
utilities remain; these are deliberate, documented adaptations of the template.

## 5. Notebook 1

Loads the pinned collection and builds one summary row per dataset; every feature has dtype,
type, missingness and distinct counts/ratios. Text fields have character, whitespace-word and
approximate-token mean/median/p95/max plus deterministic local examples. Numeric/categorical
summaries and target distributions/quantiles are expandable per dataset. Two A4 PDF figures
show row text burden and dataset text volume. Fingerprinted machine-readable artifacts are
cached; every displayed detail is printed into ignored full reports. Source notebooks keep empty outputs.

## 6. Notebooks 2 and 3

Notebook 2 reuses the audit, shows deterministic templates and representative requests for all
three task types, and calculates three text-only mode budgets. Empty inputs are skipped; joint
requests use available text. The temporary mock cache demonstration makes no API call.

Notebook 3 plans 40 dataset/representation table views, shows the documented API body and feature
schemas, and reports workload, deduplication and context screens. It does not generate features.
Each notebook's final cell prints and saves every displayed table/JSON/figure detail. Reports
and captions are grouped by phase under `output_JEVPFN/`; source notebooks stay output-free.

## 7. Data availability

**20/20 datasets successfully downloaded, hash-verified and loaded:** 821,071 rows, 273 features across dataset schemas, including 90 text and 183 non-text columns. There are 3 binary, 7 multiclass and 10 regression tasks. Every row/text/non-text count matches the pinned official summary. No missing targets were observed. All data are Kaggle version 1; upstream membership/definitions are pinned to `3bb95079a6146ed35d45187e98bcad5b063273a1` (`paper_version`).

| Dataset | Rows | Text fields | Median combined tokens | p95 combined tokens | Total text tokens |
|---|---:|---:|---:|---:|---:|
| Fake Job Postings | 12,725 | 3 | 272 | 734.00 | 4,082,241 |
| Jigsaw Toxicity | 100,000 | 1 | 51 | 239.00 | 7,471,809 |
| Kickstarter | 86,502 | 3 | 49 | 120.00 | 4,862,825 |
| Data Scientist Salary | 15,841 | 5 | 55 | 66.00 | 814,672 |
| Michelin Guide | 18,843 | 6 | 168 | 271.00 | 3,316,455 |
| Product Sentiment | 5,091 | 1 | 27 | 36.00 | 134,899 |
| Spotify Genres | 114,000 | 3 | 12 | 27.00 | 1,594,155 |
| US Accidents | 100,001 | 6 | 26 | 42.00 | 2,720,373 |
| Wine Review | 84,123 | 2 | 62 | 93.00 | 5,352,308 |
| Women's Clothing | 18,788 | 2 | 78 | 133.00 | 1,483,039 |
| Baby Products | 5,085 | 4 | 16 | 26.00 | 86,044 |
| Book Price | 4,989 | 5 | 252 | 529.00 | 1,382,934 |
| Book Readability | 4,724 | 6 | 275 | 325.85 | 1,299,261 |
| Mercari Marketplace | 100,000 | 6 | 43 | 154.00 | 5,757,219 |
| Montgomery Salaries | 9,228 | 3 | 16 | 24.00 | 154,945 |
| Rotten Tomatoes | 7,158 | 13 | 140 | 164.00 | 984,389 |
| SciMagojr Impact | 31,136 | 10 | 52 | 87.00 | 1,702,022 |
| Vancouver Salaries | 44,574 | 2 | 7 | 12.00 | 347,170 |
| Video Games Sales | 16,598 | 2 | 9 | 17.00 | 162,009 |
| Zomato Restaurants | 41,665 | 7 | 698 | 16,129.80 | 130,724,890 |

## 8. Metadata and target semantics

- Released metadata has target/task/shape/class count, but no explicit text-column list. The project follows the pinned official detector and freezes its results; it does not invent a language detector. All 90 column names reproduce that rule in the supported environment.
- The rule includes high-cardinality short strings and some numeric-looking/date-like fields. Rotten Tomatoes has 13 text fields, including RatingCount, ReviewCount and Release Date; SciMagojr has 10, including SJR and two ratio fields. These are upstream definitions, not evidence that each is natural language.
- Data Scientist Salary is six-class classification with interval-like string labels; Wine Review predicts variety (30 classes), not a review score; Spotify has 114 genre classes. Mercari predicts log_price. These original tasks were retained.
- Binary orientation and multiclass ordering are canonical JSON order. No positive-class domain meaning or ordinal relationship is inferred. Target units, scalar reduction and relative regression interpretation remain review items.

## 9. Methodologically important observations

- Raw combined text totals approximately **174,433,659 tokens** under `ceil(characters / 4)`. These are uncalibrated character estimates, not Jev tokenizer measurements.
- **Zomato contributes 74.94%** (130,724,890 tokens). Median combined row burden is 698 tokens, p95 is 16,129.8, and maximum is 321,073. The reviews_list field dominates. No context-limit or truncation policy has been selected.
- Vancouver Salaries and Video Games Sales have median combined text burdens of 7 and 9 tokens; Spotify has 12. A text-column count alone therefore does not identify substantial prose. Book Readability, Book Price, job descriptions and restaurant reviews carry much longer text.
- Fake Job Postings salary_range is 83.27% missing; Book Readability British Words is 90.60% missing; Baby Products company_free is 78.03% missing. Empty inputs now skip Jev and produce missing numeric features; joint requests use available fields.
- Book Readability includes other readability measurements among non-text features. Jigsaw includes identity annotations/reactions, and US Accidents includes end-time/event fields. These observed fields motivate reviewing predictor availability and related-task proxies; no leakage or predictive effect was measured or asserted.
- Task/question/class-choice overhead can exceed raw text volume, especially in per-column mode. Jev now receives text only; non-text features enter TabPFN separately.

## 10. Jev workload and costs

The agreed text-only per-column-plus-joint build has 3,985,167 nonempty request slots and
2,032,351 distinct requests after reuse. The central API-JSON approximation is 853,391,404
input tokens, about USD 35.84 at the configurable reference rate. Sensitivity: USD 28.68-61.94.
Long-text handling and pilot calibration remain necessary; no paid calls have been made.
See [FEATURE_CREATION.md](FEATURE_CREATION.md) for modes, sources and caveats.

## 11. Unresolved choices and phase plan

[DESIGN_NOTES.md](DESIGN_NOTES.md) preserves the full open-methodology list.
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) gives the exploration, feature pilot/full build, VSC
experiment-0 debugging, main experiment and proposed ablation plans. [VSC.md](VSC.md) records
CreditPFN-derived account/environment choices and the exact prepared connectivity/CUDA checks.

Still open: final question wording/class orientation and target interpretation; long-text overflow handling; live Jev transport/authentication/version behaviour; token calibration,
account rate limits, pilot budget and approval; concurrency/retry/crash reconciliation and backups;
selected downstream feature view; TabPFN interface/weights/categorical handling; experiment-0
acceptance; evaluation splits/metrics/sample sizes; extra baselines; generated questions;
extended datasets; statistical tests and paper framing. No final protocol was chosen.

## Verification and practical limits

Current checks are recorded in CHANGELOG.md and AGENTS_MEMORY.md. All 20 datasets remain
loadable offline; CSV/metadata hashes and pinned upstream sources are checked by preparation.
The 60 source files remain directly in numbered folders. No data relocation was part of the
output refactor. The older dataset-loss recovery and retired cleanup command are documented
in AGENTS_MEMORY.md; do not reuse old cleanup commands.

The current path is `output_JEVPFN/exploration/manifests/latest_data_audit.json`. Full reports,
raw examples, features, weights and credentials are ignored by Git. No model training,
inference call, weight download or VSC allocation has run. Declared SDK/TabPFN/Torch extras
remain optional; cluster Python/CUDA/storage checks require a real allocation later.

The public GitHub setup preserves the template history and read-only literature pin. See
[GITHUB.md](GITHUB.md) for push/pull and separate raw-data/feature transfer instructions.
