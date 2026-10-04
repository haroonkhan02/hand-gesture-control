"""Train the landmark gesture classifier and write an evaluation report.

Compares three models on the same held-out data:
  * rule-based baseline (no training)
  * logistic regression
  * MLP (2 hidden layers)
The best learned model (by macro-F1) is saved to models/gesture_mlp.joblib.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .config import CLASSIFIER_PATH, DATASET_PATH, REPORTS_DIR  # noqa: E402
from .dataset import LandmarkDataset, load_dataset  # noqa: E402
from .features import normalize_landmarks  # noqa: E402
from .gestures import LandmarkClassifier, RuleBasedClassifier  # noqa: E402


def featurize(ds: LandmarkDataset) -> np.ndarray:
    pairs = zip(ds.points, ds.handedness, strict=True)
    return np.stack([normalize_landmarks(p, h) for p, h in pairs])


def split_by_session(ds: LandmarkDataset, seed: int = 0) -> tuple[np.ndarray, np.ndarray, str]:
    """Hold out each gesture's most recent session; fall back to a random split.

    Consecutive frames are almost identical, so a random split leaks near-duplicates
    into the test set and inflates accuracy.
    """
    idx = np.arange(len(ds))
    test_mask = np.zeros(len(ds), dtype=bool)
    per_label_sessions = {lab: sorted(set(ds.sessions[ds.labels == lab]))
                          for lab in np.unique(ds.labels)}
    if all(len(s) >= 2 for s in per_label_sessions.values()):
        for lab, sessions in per_label_sessions.items():
            test_mask |= (ds.labels == lab) & (ds.sessions == sessions[-1])
        return idx[~test_mask], idx[test_mask], "held-out recording session per gesture"
    train, test = train_test_split(idx, test_size=0.2, stratify=ds.labels, random_state=seed)
    note = ("random stratified 80/20 split (record >= 2 sessions per gesture for an "
            "honest session-level split)")
    return train, test, note


def build_models(seed: int = 0) -> dict:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=800,
                          early_stopping=True, random_state=seed),
        ),
    }


def latency_ms(fn, n: int = 200) -> float:
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n * 1000


def train_and_evaluate(dataset_path: Path, model_path: Path, reports_dir: Path,
                       seed: int = 0) -> dict:
    ds = load_dataset(dataset_path)
    X, y = featurize(ds), ds.labels
    train_idx, test_idx, split_note = split_by_session(ds, seed)
    labels = sorted(np.unique(y))

    results: dict[str, dict] = {}
    rules = RuleBasedClassifier()
    rule_pred = np.array([rules.predict(ds.points[i], ds.handedness[i])[0] for i in test_idx])
    results["rules (baseline)"] = {"pred": rule_pred, "latency": latency_ms(
        lambda: rules.predict(ds.points[test_idx[0]], ds.handedness[test_idx[0]]))}

    fitted = {}
    for name, model in build_models(seed).items():
        model.fit(X[train_idx], y[train_idx])
        fitted[name] = model
        sample = X[test_idx[:1]]
        results[name] = {"pred": model.predict(X[test_idx]),
                         "latency": latency_ms(lambda m=model, s=sample: m.predict_proba(s))}

    for r in results.values():
        r["acc"] = accuracy_score(y[test_idx], r["pred"])
        r["f1"] = f1_score(y[test_idx], r["pred"], labels=labels, average="macro",
                           zero_division=0)

    best = max(fitted, key=lambda n: results[n]["f1"])
    LandmarkClassifier(fitted[best], labels).save(
        model_path, model_name=best, n_train=len(train_idx), split=split_note)

    reports_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y[test_idx], results[best]["pred"], labels=labels,
                                            ax=ax, cmap="Blues", xticks_rotation=45,
                                            colorbar=False)
    ax.set_title(f"Confusion matrix - {best}")
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    counts = {lab: int((y == lab).sum()) for lab in labels}
    lines = [
        "# Gesture classifier evaluation",
        "",
        f"- Samples: {len(y)} ({', '.join(f'{k}: {v}' for k, v in counts.items())})",
        f"- Recording sessions: {len(set(ds.sessions))}",
        f"- Split: {split_note} -> {len(train_idx)} train / {len(test_idx)} test",
        "",
        "| Model | Accuracy | Macro F1 | Latency / prediction |",
        "|---|---|---|---|",
    ]
    for name, r in results.items():
        mark = " **(saved)**" if name == best else ""
        lines.append(f"| {name}{mark} | {r['acc']:.3f} | {r['f1']:.3f} | {r['latency']:.2f} ms |")
    lines += ["", f"## Per-class report ({best})", "", "```",
              classification_report(y[test_idx], results[best]["pred"], labels=labels,
                                    zero_division=0).rstrip(),
              "```", "", "![Confusion matrix](confusion_matrix.png)", ""]
    (reports_dir / "metrics.md").write_text("\n".join(lines))
    return {"best": best, "results": results, "split": split_note}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATASET_PATH)
    parser.add_argument("--model", type=Path, default=CLASSIFIER_PATH)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = train_and_evaluate(args.data, args.model, args.reports, args.seed)
    for name, r in out["results"].items():
        print(f"{name:18s} acc={r['acc']:.3f}  macro-F1={r['f1']:.3f}  {r['latency']:.2f} ms")
    print(f"Saved {out['best']} to {args.model}; report in {args.reports / 'metrics.md'}")


if __name__ == "__main__":
    main()
