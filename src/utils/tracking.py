"""Experiment tracking for the training and evaluation stages.

Stages talk to a :class:`Tracker`. :class:`MlflowTracker` is the only code that
imports ``mlflow`` (lazily, on first use); :class:`NullTracker` records nothing
and is what the tests use.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from logging import WARNING, getLogger
from pathlib import Path
from typing import Any, Protocol

from utils.project import DataPaths

getLogger("alembic").setLevel(WARNING)


class Tracker(Protocol):
    def run(self, name: str, run_id: str | None = None) -> AbstractContextManager[str | None]:
        """Start (or, with *run_id*, resume) a run; yields its id, or None if nothing is recorded."""
        ...

    def find_run(self, name: str) -> str | None:
        """Id of the latest run called *name*."""
        ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float]) -> None: ...

    def log_artifact(self, path: Path, artifact_path: str) -> None: ...

    def log_artifacts(self, directory: Path, artifact_path: str) -> None: ...


class NullTracker:
    """Records nothing."""

    @contextmanager
    def run(self, name: str, run_id: str | None = None) -> Iterator[str | None]:
        yield run_id

    def find_run(self, name: str) -> str | None:
        return None

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        pass

    def log_artifact(self, path: Path, artifact_path: str) -> None:
        pass

    def log_artifacts(self, directory: Path, artifact_path: str) -> None:
        pass


class MlflowTracker:
    """MLflow with a local SQLite store in ``data/models/mlflow.db``.

    Browse with ``uv run mlflow ui --backend-store-uri sqlite:///data/models/mlflow.db``.
    """

    def __init__(self, experiment_name: str, paths: DataPaths) -> None:
        self.experiment_name = experiment_name
        self.paths = paths
        self._configured = False

    def _mlflow(self) -> Any:
        import mlflow

        if not self._configured:
            self.paths.models.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(f"sqlite:///{self.paths.mlflow_db}")
            if mlflow.get_experiment_by_name(self.experiment_name) is None:
                mlflow.create_experiment(self.experiment_name, artifact_location=self.paths.mlflow_artifacts.as_uri())
            mlflow.set_experiment(self.experiment_name)
            self._configured = True
        return mlflow

    @contextmanager
    def run(self, name: str, run_id: str | None = None) -> Iterator[str | None]:
        with self._mlflow().start_run(run_id=run_id, run_name=None if run_id else name) as active:
            yield active.info.run_id

    def find_run(self, name: str) -> str | None:
        runs = self._mlflow().search_runs(
            filter_string=f"tags.mlflow.runName = '{name}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
            output_format="list",
        )
        return runs[0].info.run_id if runs else None

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow().log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._mlflow().log_metrics(metrics)

    def log_artifact(self, path: Path, artifact_path: str) -> None:
        self._mlflow().log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self, directory: Path, artifact_path: str) -> None:
        self._mlflow().log_artifacts(str(directory), artifact_path=artifact_path)
