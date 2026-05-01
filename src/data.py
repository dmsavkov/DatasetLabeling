# pyright: basic

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

EXPECTED_LABEL_ORDER = [
	"casual",
	"possibly_needs_caution",
	"probably_needs_caution",
	"needs_caution",
	"needs_intervention",
]

LABEL_DISTANCE_MAP = {
	"casual": 0,
	"possibly_needs_caution": 1,
	"probably_needs_caution": 2,
	"needs_caution": 3,
	"needs_intervention": 4,
}


def now_stamp() -> str:
	return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def normalize_label(value: Any) -> str:
	label = str(value).strip().lower()
	if label.startswith("__") and label.endswith("__"):
		label = label[2:-2]
	label = label.replace(" ", "_")
	return label


def normalize_annotation(value: Any) -> str:
	return str(value).strip().lower().replace(" ", "_")


def collapse_label_to_three(value: Any) -> str:
	label = normalize_label(value)
	if label in {"possibly_needs_caution", "probably_needs_caution", "needs_caution"}:
		return "needs_caution"
	return label


def collapse_series_to_three_labels(series: pd.Series) -> pd.Series:
	return pd.Series([collapse_label_to_three(v) for v in series], dtype="string")


def load_split_jsonl(path: Path, *, include_all_features: bool = False) -> pd.DataFrame:
	df = pd.read_json(path, lines=True)
	df = df.copy()
	df["source_index"] = df.index
	df["context"] = df["context"].fillna("").astype(str).str.strip()
	df["safety_label"] = df["safety_label"].map(normalize_label)
	if "safety_annotations" in df.columns:
		df["safety_annotations"] = [
			[normalize_annotation(item) for item in values] if isinstance(values, list) else []
			for values in df["safety_annotations"]
		]
	if "safety_annotation_reasons" in df.columns:
		df["safety_annotation_reasons"] = [
			[str(item).strip() for item in values] if isinstance(values, list) else []
			for values in df["safety_annotation_reasons"]
		]
	df = df[df["context"] != ""].reset_index(drop=True)

	if include_all_features:
		cols = ["source_index"] + [c for c in df.columns if c != "source_index"]
		return df.loc[:, cols].copy()

	keep_cols = ["source_index", "context", "response", "safety_label", "dialogue_id", "response_id", "source"]
	cols = [c for c in keep_cols if c in df.columns]
	return df.loc[:, cols].copy()


def load_prosocial_dialog(root: Path | None = None, *, include_all_features: bool = False) -> dict[str, Any]:
	root_dir = Path.cwd() if root is None else Path(root)
	data_dir = root_dir / "data" / "prosocial-dialog"
	results_dir = root_dir / "data" / "results"
	results_dir.mkdir(parents=True, exist_ok=True)

	train_df = load_split_jsonl(data_dir / "train.json", include_all_features=include_all_features)
	valid_df = load_split_jsonl(data_dir / "valid.json", include_all_features=include_all_features)
	test_df = load_split_jsonl(data_dir / "test.json", include_all_features=include_all_features)

	all_labels = sorted(set(train_df["safety_label"]) | set(valid_df["safety_label"]) | set(test_df["safety_label"]))
	label_order = [label for label in EXPECTED_LABEL_ORDER if label in all_labels]

	return {
		"root": root_dir,
		"data_dir": data_dir,
		"results_dir": results_dir,
		"train_df": train_df,
		"valid_df": valid_df,
		"test_df": test_df,
		"label_order": label_order,
	}


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> dict[str, Any]:
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division="warn")),
		"weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division="warn")),
		"report": classification_report(
			y_true,
			y_pred,
			labels=labels,
			zero_division="warn",
			output_dict=True,
		),
	}


def evaluate_adjusted_distance(
	y_true: pd.Series,
	y_pred: pd.Series,
	*,
	unknown_penalty: int = 4,
	label_distance_map: dict[str, int] | None = None,
) -> float:
	label_levels = LABEL_DISTANCE_MAP if label_distance_map is None else label_distance_map
	distances: list[int] = []
	for true_value, pred_value in zip(y_true, y_pred):
		true_label = normalize_label(true_value)
		pred_label = normalize_label(pred_value)
		true_level = label_levels.get(true_label)
		pred_level = label_levels.get(pred_label)
		if true_level is None or pred_level is None:
			distances.append(unknown_penalty)
			continue
		distances.append(abs(true_level - pred_level))

	if not distances:
		return float(unknown_penalty)
	return float(sum(distances) / len(distances))


def save_json(path: Path, payload: dict[str, Any]) -> None:
	_ = path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
