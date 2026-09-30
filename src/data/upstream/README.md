# Frozen upstream reference files

These files came from the pinned MulTaBench source commit recorded in
`config/exploration/datasets.yaml`. They are retained byte-for-byte for provenance and
AST-only metadata extraction, not imported or executed. `summary.csv` contains public
benchmark-level metadata, not raw dataset rows. Source URLs and SHA-256 hashes are in
the catalog. This directory is excluded from formatting, linting and package discovery.

Reusable project code belongs directly in `src/data/`; raw datasets belong in
`data/raw/<number>_<name>/`. No Python files belong in `data/`.
