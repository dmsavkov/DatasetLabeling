from __future__ import annotations

from time import perf_counter

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from src.data import evaluate_predictions, now_stamp, save_json
from src.dataset_scripts import load_prosocial_dialog_bundle, make_dspy_sample_splits

SEED = 42


def main() -> None:
    bundle = load_prosocial_dialog_bundle()
    train_df = bundle["train_df"]
    valid_df = bundle["valid_df"]
    test_df = bundle["test_df"]
    label_order = bundle["label_order"]

    splits = make_dspy_sample_splits(test_df, seed=SEED, sample_size=50, train_size=25)
    dspy_test_df = splits["dspy_test_df"]

    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    X_train = train_valid_df["context"].astype(str)
    y_train_str = train_valid_df["safety_label"].astype(str)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_str)

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=80000,
                ),
            ),
            (
                "xgb",
                XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    n_estimators=350,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    tree_method="hist",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    t0 = perf_counter()
    pipeline.fit(X_train, y_train)
    fit_seconds = perf_counter() - t0

    y_full_true = test_df["safety_label"].astype(str)
    y_full_pred = label_encoder.inverse_transform(pipeline.predict(test_df["context"].astype(str)))
    full_metrics = evaluate_predictions(y_full_true, pd.Series(y_full_pred), label_order)

    y_25_true = dspy_test_df["safety_label"].astype(str)
    y_25_pred = label_encoder.inverse_transform(pipeline.predict(dspy_test_df["context"].astype(str)))
    metrics_25 = evaluate_predictions(y_25_true, pd.Series(y_25_pred), label_order)

    stamp = now_stamp()
    results_dir = bundle["results_dir"]
    pred_25_path = results_dir / f"mvp_xgb_preds_25test_{stamp}.csv"
    summary_path = results_dir / f"mvp_xgb_summary_{stamp}.json"

    pd.DataFrame(
        {
            "context": dspy_test_df["context"],
            "true_label": y_25_true,
            "pred_label": y_25_pred,
        }
    ).to_csv(pred_25_path, index=False)

    save_json(
        summary_path,
        {
            "workflow": "tfidf_xgboost",
            "train_size": int(len(train_valid_df)),
            "timing_seconds": {"fit": float(fit_seconds)},
            "metrics": {
                "full_test": {
                    "accuracy": full_metrics["accuracy"],
                    "macro_f1": full_metrics["macro_f1"],
                    "weighted_f1": full_metrics["weighted_f1"],
                },
                "dspy_test_25": {
                    "accuracy": metrics_25["accuracy"],
                    "macro_f1": metrics_25["macro_f1"],
                    "weighted_f1": metrics_25["weighted_f1"],
                },
            },
            "artifacts": {
                "predictions_25_test_csv": str(pred_25_path),
            },
        },
    )

    print({"accuracy_25": metrics_25["accuracy"], "macro_f1_25": metrics_25["macro_f1"], "summary": str(summary_path)})


if __name__ == "__main__":
    main()