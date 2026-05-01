from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvalArtifactNames:
    report_json: str = "report.json"
    predictions_csv: str = "predictions.csv"


ARTIFACTS = EvalArtifactNames()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_path(output_dir: Path) -> Path:
    return Path(output_dir) / ARTIFACTS.report_json


def predictions_path(output_dir: Path) -> Path:
    return Path(output_dir) / ARTIFACTS.predictions_csv
