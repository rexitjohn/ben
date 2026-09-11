from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import yaml
from sklearn.model_selection import train_test_split

from .data import FEATURE_COLUMNS, TARGET_COLUMN, make_synthetic_canine_dataset
from .evaluate import classification_metrics
from .model import build_model
from .plotting import save_confusion_matrix, save_roc_curve


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_demo_data(args):
    df = make_synthetic_canine_dataset(
        args.rows,
        args.positive_rate,
        args.seed,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote synthetic demo data to {out}")


def train(args):
    cfg = _load_config(args.config)

    seed = int(cfg["experiment"]["seed"])
    test_size = float(cfg["experiment"]["test_size"])

    df = make_synthetic_canine_dataset(
        int(cfg["data"]["rows"]),
        float(cfg["data"]["positive_rate"]),
        seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS],
        df[TARGET_COLUMN],
        test_size=test_size,
        stratify=df[TARGET_COLUMN],
        random_state=seed,
    )

    model = build_model(
        float(cfg["model"]["C"]),
        int(cfg["model"]["max_iter"]),
        str(cfg["model"]["class_weight"]),
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    threshold = float(cfg["evaluation"]["threshold"])

    metrics = classification_metrics(
        y_test,
        probabilities,
        threshold,
    )

    predictions = (probabilities >= threshold).astype(int)

    out = Path(cfg["output"]["directory"])
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out / "model.joblib")

    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    save_confusion_matrix(
        y_test,
        predictions,
        out / "confusion_matrix.png",
    )

    save_roc_curve(
        y_test,
        probabilities,
        out / "roc_curve.png",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to {out}")
    print("NOTE: synthetic data only; not clinical evidence.")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ben",
        description="Biomarker Evaluation Node research CLI",
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    demo = sub.add_parser("make-demo-data")

    demo.add_argument(
        "--rows",
        type=int,
        default=1000,
    )

    demo.add_argument(
        "--positive-rate",
        type=float,
        default=0.5,
    )

    demo.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    demo.add_argument(
        "--output",
        default="data/demo_synthetic.csv",
    )

    demo.set_defaults(func=make_demo_data)

    train_parser = sub.add_parser("train")

    train_parser.add_argument(
        "--config",
        default="configs/alpha_1_0.yaml",
    )

    train_parser.set_defaults(func=train)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
