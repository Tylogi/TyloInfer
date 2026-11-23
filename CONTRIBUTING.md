# Contributing

This file holds rules about the repository itself. It currently covers ignore files.

## Ignore files

The root `.gitignore` records two kinds of paths, and only those two:

1. Files that this repository's own tooling produces — the Python toolchain in
   `app/server/pyproject.toml`, the Node toolchain in `app/`, the CMake
   build in `csrc/`, and the packaging scripts in `packaging/`.
2. Editor and OS noise that nobody working in this repository can avoid.

Everything else is the developer's own environment policy, not the repository's.

### What stays ignored here

| Block | Entries |
| --- | --- |
| Python | `__pycache__/`, `*.py[cod]`, `*$py.class`, `*.egg`, `*.egg-info/`, `.eggs/`, `build/`, `dist/`, `develop-eggs/`, `pip-wheel-metadata/`, `wheels/`, `MANIFEST`, `.venv/`, `venv/`, `env/`, `ENV/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.coverage`, `.coverage.*`, `htmlcov/`, `.tox/` |
| Node | `node_modules/`, `.pnpm-store/` |
| Native build | `CMakeFiles/`, `CMakeCache.txt`, `cmake_install.cmake`, `compile_commands.json`, `*.o`, `*.obj`, `*.a`, `*.lib`, `*.dll`, `*.dylib`, `*.so`, `*.metallib` |
| Runtime output of this project | `.tyloinfer/` |
| Editors and OS | `.idea/`, `.vscode/`, `*.swp`, `*.swo`, `*~`, `.DS_Store`, `Thumbs.db`, `desktop.ini` |

### What is not ignored here

Downloaded content and personal scratch: model files (`*.gguf`,
`*.safetensors`, `*.pt`, `*.pth`), containers this project can also generate
(`*.tq`, `*.mfq`), `*.logits`, `*.jsonl`, `*.sqlite`, `*.sqlite3`, `data/`,
`models/`, `*.log`, `tmp/`, `runs/`, local credentials such as `deploy.env`, and
any personal tooling directory.

Put those in your own global ignore (`git config --global core.excludesfile`), or
delete them yourself.

The reason is asymmetric cost. A large file committed by mistake shows up in
`git status` and is cheap to undo. A fixture that a test or a release needs, and
that `.gitignore` swallowed, is expensive to notice. `data/` and `models/` are
also ordinary source directory names in this repository, so an unanchored ignore
rule would hide real files.

### Before you add an entry

- Name the tool in this repository that produces it. If nothing here produces
  it, the entry does not belong here.
- Anchor the pattern (`/models/`) when the same name is used for source
  directories. `csrc/backends/cuda/models/` and
  `app/web/src/features/models/` are source.
- Keep nested `.gitignore` files for paths that only that subtree's tooling
  creates and the root file does not already cover. See `packaging/.gitignore`
  and `app/web/.gitignore`.
- Do not add un-ignore (`!`) rules for source directories. The two that covered
  `app/web/src/features/models/` were dead and were removed.

### Checking an ignore change

```sh
git ls-files -i -c --exclude-standard          # must print nothing
git ls-files --others --exclude-standard       # must list only expected files
git check-ignore -v <path>                     # shows which rule matched, if any
```
