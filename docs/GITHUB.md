# GitHub and VSC sync

The local Git repository is the inner `JEVPFN/` directory. VS Code Source Control shows that
local history and local changes; it does not mean a GitHub repository exists or receives edits
automatically. The original `origin` pointed to the template repository.

The public repository is now [andreasgoethals/JEVPFN](https://github.com/andreasgoethals/JEVPFN).
`origin` points there; the old URL is preserved as `template`. The branch remains `main` and
original history is retained. See [GitHub's documentation](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).

## Publish later changes from the local computer

Run inside the inner repository after reviewing your changes and clearing notebook outputs:

```powershell
git status --short
git diff --check
git add -A
git diff --cached --stat
git commit -m "Describe the research change"
git push origin main
```

GitHub updates only after **commit and push**. Do not push to `template` or use `git add -f`
for ignored files. Raw data, full notebook reports, request previews, credentials, feature
caches, environments and weights are deliberately excluded. The pinned upstream source scripts
under `src/data/upstream/` are included for reproducibility. The literature is a submodule
pointer, not a copy of all its contents.

## First VSC checkout

```bash
cd "$VSC_DATA"
git clone --recurse-submodules https://github.com/andreasgoethals/JEVPFN.git
cd JEVPFN
```

## Update an existing VSC checkout

```bash
cd "$VSC_DATA/JEVPFN"
git pull --ff-only origin main
git submodule update --init --recursive
```

Do this between jobs, so a running experiment's code does not change. A public repository
can be cloned without a GitHub token. Writing to it still requires your authenticated account.

**Git transfers code, not datasets or Jev features.** Transfer raw datasets and the completed,
closed Jev cache separately to project storage, preserving hashes. Use [VSC.md](VSC.md) to
set the storage parent and check the environment. Verify with `python -m src.data.prepare --offline`.
The retired cleanup command must not be reused; data restoration uses `python -m src.data.prepare`.
