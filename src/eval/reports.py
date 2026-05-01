# pyright: basic
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import PerformanceMetrics


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class EvalReport:
    dataset_name: str
    split_name: str
    tier_size: int
    predictor_name: str
    metrics: PerformanceMetrics
    extras: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "split_name": self.split_name,
            "tier_size": int(self.tier_size),
            "predictor_name": self.predictor_name,
            "metrics": asdict(self.metrics),
            "extras": self.extras,
            "created_at": self.created_at,
        }


def write_report_json(path: Path, report: EvalReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")


def new_report(
    *,
    dataset_name: str,
    split_name: str,
    tier_size: int,
    predictor_name: str,
    metrics: PerformanceMetrics,
    extras: dict[str, Any],
) -> EvalReport:
    return EvalReport(
        dataset_name=dataset_name,
        split_name=split_name,
        tier_size=int(tier_size),
        predictor_name=predictor_name,
        metrics=metrics,
        extras=extras,
        created_at=_utc_iso(),
    )

