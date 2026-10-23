# Contributing

Issues and pull requests are welcome. CI runs the checks and tests below on
every pull request, but running them locally first saves a round trip.

## Submitting changes

Every change reaches `main` through a pull request; nobody pushes to it
directly.

1. Fork the repository and clone your fork.
2. Create a branch in your fork and commit your changes there.
3. Push the branch to your fork and open a pull request against `main`.

Collaborators with write access can skip the fork and push a branch to this
repository instead. Its name must start with `feature/`, `fix/` or `docs/`.

A pull request can be merged when:

- **CI passes.** The workflow in `.github/workflows/ci.yml` runs `lint` (ruff
  and mypy), then `test` (pytest). For a first-time contributor, a maintainer
  approves the run before it starts.
- **A maintainer approves it.** Anyone can review, but only approvals from
  people with write access count. New commits dismiss earlier approvals.
- **All review conversations are resolved.**
- **The branch is up to date with `main`.**

## Development setup

```bash
uv sync
```

This installs the project with the `dev` dependency group (pytest, ruff, mypy);
see [Setup](README.md#setup) for the requirements.

## Tests and checks

Tests (170, about 14 s; they train a tiny tokenizer and use small fixture files,
so they need no downloaded data or network, and never touch MLflow).
`tests/test_pipeline.py` runs preprocessing, feature engineering, training and
evaluation end to end on a synthetic corpus (baselines come from a fake Hub).
Trained models in `data/models/` are also checked (special-token ids, round-trip and the tiktoken equality gate
on a holdout sample) when present, and skipped otherwise:

```bash
uv run pytest
```

Lint and static checks (ruff with a 120-character line length; mypy over
`src/` with untyped definitions disallowed):

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
```

## Conventions

- **No MLflow in tests.** `tests/conftest.py` blocks `import mlflow`. Stages
  take a `tracker` argument, and tests pass `utils.tracking.NullTracker`.
  Only `utils/tracking.py` imports MLflow.
- **Tests need no network or downloaded data.** Hub downloads are faked
  (`fake_hub_download` in `tests/conftest.py`), and checks of real models in
  `data/models/` are skipped when the models are missing.
- **Every dependency has an upper bound**: the next major version, or the next
  minor version for 0.x packages (see the comment in `pyproject.toml`).
- **Bump `VERSION` when output changes.** Preprocessing and feature engineering
  skip work whose fingerprint matches, and the fingerprint includes each
  module's `VERSION` constant. `tests/test_versions.py` pins every module's
  source hash, so any code change fails until `tests/versions.json` is updated;
  bump the module's `VERSION` too if the change alters the output (see
  [architecture.md](docs/architecture.md#preprocessing)).
- **Settings live in code.** Each stage has a pydantic settings class, and
  `src/main.py` builds them and turns stages on or off with `run_step`. There is
  no config-file or command-line layer.
- **Data problems raise a `PipelineError`** subclass (`src/utils/errors.py`), so
  `main.py` reports them in one line instead of a traceback.
- **Outputs are typed and written atomically.** JSON files that a later stage
  reads are `utils.artifacts.Artifact` models, and files are written to a
  temporary name and renamed when complete.
- **Code style**: ruff (120 characters, isort with the stage packages as
  first-party), mypy with untyped definitions disallowed.

## Adding a stage

Follow the existing stages, for example `src/evaluation/`:

1. A package under `src/` whose `__init__.py` exports the settings class and
   `run as run_<stage>`. The module docstring documents what the stage reads and
   writes and how to run it; the `__main__` block runs inside
   `exclusive_run(default_data_root())`.
2. Its paths as properties of `DataPaths` and a config class in
   `WorkflowConfigs` (`src/utils/project.py`), and a `run_step` entry in
   `src/main.py`.
3. The package in `[tool.hatch.build.targets.wheel].packages` and in ruff's
   `known-first-party` in `pyproject.toml`.
4. Its run in `tests/test_pipeline.py`, and its documentation in
   [docs/architecture.md](docs/architecture.md).
