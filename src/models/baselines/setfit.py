# pyright: basic
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass

from src.models.interfaces import Prediction, Usage


@dataclass(frozen=True, slots=True)
class TrainStats:
    train_time_s: float


class SetFitPredictor:
    """SetFit predictor using the `setfit` package (HF)."""

    def __init__(
        self,
        *,
        embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_steps: int | None = 2000,
        epochs: int = 1,
        name: str = "setfit",
    ) -> None:
        self._name = name
        self._embedding_model_id = embedding_model_id
        self._max_steps = int(max_steps) if max_steps is not None else None
        self._epochs = int(epochs)

        self._model: object | None = None
        self._label_order: list[str] | None = None
        self.train_stats: TrainStats | None = None

    @property
    def name(self) -> str:
        return self._name

    def fit(self, texts: list[str], labels: list[str]) -> TrainStats:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have same length")
        if not texts:
            raise ValueError("empty training set")

        start = time.perf_counter()
        label_order = sorted(set(str(x) for x in labels))
        self._label_order = label_order

        setfit_mod = importlib.import_module("setfit")
        datasets_mod = importlib.import_module("datasets")
        SetFitModel = getattr(setfit_mod, "SetFitModel")
        Trainer = getattr(setfit_mod, "Trainer")
        TrainingArguments = getattr(setfit_mod, "TrainingArguments")
        Dataset = getattr(datasets_mod, "Dataset")

        # SetFit expects a `label` column; keep labels as strings and set the
        # model's label names explicitly.
        train_ds = Dataset.from_dict({"text": [str(t) for t in texts], "label": [str(y) for y in labels]})

        model = SetFitModel.from_pretrained(self._embedding_model_id, labels=label_order)
        args_kwargs = {
            "num_epochs": int(self._epochs),
            "batch_size": 16,
            "show_progress_bar": False,
        }
        if self._max_steps is not None:
            args_kwargs["max_steps"] = int(self._max_steps)
        args = TrainingArguments(**args_kwargs)
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, column_mapping={"text": "text", "label": "label"})
        trainer.train()
        self._model = model

        elapsed = time.perf_counter() - start
        self.train_stats = TrainStats(train_time_s=float(elapsed))
        return self.train_stats

    def predict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
        if self._model is None:
            raise RuntimeError("Model is not fit yet. Call fit(...) first.")
        if not texts:
            return []
        if not allowed_labels:
            raise ValueError("allowed_labels must be non-empty")

        allowed = set(allowed_labels)
        model = self._model
        pred = getattr(model, "predict")([str(t) for t in texts])

        # SetFit may return list[str], numpy arrays, or other array-like structures.
        # Normalize everything into a flat python list of labels.
        if isinstance(pred, list):
            pred_list = pred
        elif isinstance(pred, (str, bytes)):
            pred_list = [pred]
        elif hasattr(pred, "tolist"):
            pred_list = pred.tolist()  # type: ignore[no-untyped-call]
        else:
            try:
                pred_list = list(pred)  # type: ignore[arg-type]
            except TypeError:
                pred_list = [pred]

        if not isinstance(pred_list, list):
            pred_list = [pred_list]

        # Be defensive: if a backend returns a scalar label, broadcast it.
        if len(pred_list) == 1 and len(texts) > 1:
            pred_list = [pred_list[0] for _ in texts]

        out: list[Prediction] = []
        for lab in pred_list:
            lab_s = str(lab)
            if lab_s not in allowed:
                lab_s = None
            out.append(
                Prediction(
                    pred_label=lab_s,
                    confidence=None,
                    probs=None,
                    usage=Usage(None, None),
                    raw=None,
                )
            )
        return out

