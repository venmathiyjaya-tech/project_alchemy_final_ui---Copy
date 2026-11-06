"""
ml/recommender/train_recommender.py

Training pipeline for the scent VAD recommender system.

Steps
-----
1. Load scents from data/scents.json
2. Instantiate MLRecommender and RuleBasedRecommender
3. Train three model types in sequence: knn, random_forest, mlp
4. For each model:
   - Print 5-fold cross-validation accuracy
   - Evaluate Mean Absolute Error in VAD distance vs RuleBasedRecommender
     on 100 random VAD inputs
5. Save the best model (highest CV accuracy) to models/saved_model.joblib
6. Write results to experiments/recommender_comparison.csv
7. Print a final summary table showing which model won and why
"""

from __future__ import annotations

import csv
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup – support running from project root or from this file's directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent           # ml/recommender/
_PROJECT_ROOT = _HERE.parent.parent               # project root

SCENTS_PATH    = _PROJECT_ROOT / "data"        / "scents.json"
MODEL_SAVE_PATH = _PROJECT_ROOT / "models"     / "saved_model.joblib"   # same file the app loads
CSV_SAVE_PATH   = _PROJECT_ROOT / "experiments" / "recommender_comparison.csv"

# Add project root to sys.path so `from ml.recommender.recommender import …`
# works regardless of how the script is invoked.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.recommender.recommender import MLRecommender, RuleBasedRecommender  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_TYPES   = ("knn", "random_forest", "mlp")
N_EVAL_INPUTS = 100
RNG_SEED      = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_vad_inputs(n: int, rng: np.random.Generator) -> list[dict]:
    """Generate *n* random VAD dicts with values uniformly in [0, 1]."""
    raw = rng.uniform(0.0, 1.0, size=(n, 3))
    return [
        {"valence": float(row[0]), "arousal": float(row[1]), "dominance": float(row[2])}
        for row in raw
    ]


def _vad_distance(a: dict, b: dict) -> float:
    """Euclidean distance between two VAD dicts."""
    return float(np.sqrt(
        (a["valence"]   - b["valence"])   ** 2
        + (a["arousal"]   - b["arousal"])   ** 2
        + (a["dominance"] - b["dominance"]) ** 2
    ))


def _mae_vs_rule_based(
    ml: MLRecommender,
    rb: RuleBasedRecommender,
    vad_inputs: list[dict],
) -> float:
    """
    Compute the Mean Absolute Error (in VAD Euclidean distance) between the
    ML model's top recommendation and the rule-based top recommendation over
    *vad_inputs*.

    For each input the "ground truth" VAD point is the scent returned by the
    RuleBasedRecommender; the "prediction" VAD point is the scent returned by
    the MLRecommender.  MAE = mean distance between the two recommended scents'
    VAD coordinates.

    A MAE of 0 means the two systems always agree on the same scent.
    """
    errors: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for vad in vad_inputs:
            rb_scent = rb.recommend(vad)
            ml_scent = ml.recommend(vad)
            # Distance between the two recommended scents in VAD space
            err = _vad_distance(rb_scent, ml_scent)
            errors.append(err)
    return float(np.mean(errors))


def _cv_top_k(ml: MLRecommender, model_type: str, k: int = 3, folds: int = 5):
    """
    Cross-validated top-k hit rate: how often the true scent is among the k most
    probable scents. This is the metric that matches the app's real goal (a short,
    sensible list), unlike top-1, which is capped by scents that overlap in VAD space.
    """
    from sklearn.model_selection import StratifiedKFold

    X, y = ml.build_training_data()
    y_enc = ml._label_encoder.transform(y)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in skf.split(X, y_enc):
            clf = MLRecommender._build_classifier(model_type)
            clf.fit(X[train_idx], y_enc[train_idx])
            top = np.argsort(-clf.predict_proba(X[test_idx]), axis=1)[:, :k]
            hit = (clf.classes_[top] == y_enc[test_idx][:, None]).any(axis=1)
            scores.append(float(hit.mean()))
    return float(np.mean(scores)), float(np.std(scores))


def _nearest_prototype_ceiling(ml: MLRecommender, n: int = 60_000, seed: int = 0):
    """
    Best accuracy ANY classifier can reach on this synthetic task.

    The test draws a noisy copy of a scent (Gaussian sigma, clipped to [0, 1]) and asks
    which scent it came from. For isotropic noise the nearest-prototype rule is the
    optimal answer, so its hit rate is the ceiling. Returns (top1, top3).
    """
    rng = np.random.default_rng(seed)
    protos = np.array([[s["valence"], s["arousal"], s["dominance"]] for s in ml.scents])
    truth = rng.integers(0, len(protos), size=n)
    noisy = np.clip(protos[truth] + rng.normal(0.0, MLRecommender._NOISE_STD, (n, 3)), 0.0, 1.0)
    order = np.argsort(((noisy[:, None, :] - protos[None]) ** 2).sum(axis=2), axis=1)
    return (float((order[:, 0] == truth).mean()),
            float((order[:, :3] == truth[:, None]).any(axis=1).mean()))


def _bar(value: float, max_val: float = 1.0, width: int = 20, fill: str = "█") -> str:
    """Simple ASCII progress bar."""
    filled = int(round(value / max_val * width))
    return fill * filled + "░" * (width - filled)


def _print_section(title: str, width: int = 72) -> None:
    print()
    print("╔" + "═" * (width - 2) + "╗")
    print("║  " + title.ljust(width - 4) + "║")
    print("╚" + "═" * (width - 2) + "╝")


def _print_model_result(
    model_type: str,
    cv_mean: float,
    cv_std: float,
    mae: float,
    elapsed: float,
    is_best: bool = False,
) -> None:
    tag = "  ★ BEST" if is_best else ""
    print(f"\n  ┌─ {model_type.upper()}{tag}")
    print(f"  │  CV Accuracy : {cv_mean:.4f} ± {cv_std:.4f}  "
          f"{_bar(cv_mean, width=18)}")
    print(f"  │  vs-Rule MAE : {mae:.6f}  "
          f"{_bar(mae, max_val=0.5, width=18, fill='▒')}  "
          f"(lower is better)")
    print(f"  └─ Training time: {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # ── 0. Validate paths & create output dirs ────────────────────────────
    if not SCENTS_PATH.is_file():
        sys.exit(f"[ERROR] Scents file not found: {SCENTS_PATH}\n"
                 "        Place scents.json in the data/ directory.")

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CSV_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    _print_section("Scent VAD Recommender – Training Pipeline")
    print(f"  Scents file : {SCENTS_PATH}")
    print(f"  Model output: {MODEL_SAVE_PATH}")
    print(f"  CSV output  : {CSV_SAVE_PATH}")

    # ── 1. Load recommenders ──────────────────────────────────────────────
    rb  = RuleBasedRecommender(str(SCENTS_PATH))
    ml  = MLRecommender(str(SCENTS_PATH))

    print(f"\n  Loaded {len(rb.scents)} scents.")
    X, y = ml.build_training_data()
    print(f"  Training set after augmentation: {X.shape[0]} samples "
          f"({MLRecommender._AUGMENT_N} synthetic + 1 real per scent).")

    # ── 2. Generate fixed evaluation inputs ───────────────────────────────
    rng = np.random.default_rng(RNG_SEED)
    eval_inputs = _random_vad_inputs(N_EVAL_INPUTS, rng)
    print(f"  Evaluation  : {N_EVAL_INPUTS} random VAD inputs (seed={RNG_SEED}).")

    # ── 3. Train each model type ──────────────────────────────────────────
    _print_section("Training & Evaluation")

    results: list[dict] = []

    for model_type in MODEL_TYPES:
        print(f"\n  ▶ Training [{model_type}] …", flush=True)
        t0 = time.perf_counter()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = ml.train(model_type=model_type)

        elapsed = time.perf_counter() - t0

        mae = _mae_vs_rule_based(ml, rb, eval_inputs)
        top3_mean, top3_std = _cv_top_k(ml, model_type, k=3)

        record = {
            "model_type"       : model_type,
            "cv_accuracy_mean" : scores["mean_accuracy"],
            "cv_accuracy_std"  : scores["std_accuracy"],
            "vs_rule_mae"      : mae,
            "cv_top3_mean"     : top3_mean,
            "cv_top3_std"      : top3_std,
            "elapsed_s"        : elapsed,
            "_model_snapshot"  : ml.model,          # keep for later save
            "_label_encoder"   : ml._label_encoder,
        }
        results.append(record)

        _print_model_result(
            model_type,
            scores["mean_accuracy"],
            scores["std_accuracy"],
            mae,
            elapsed,
        )
        print(f"     Top-3 hit rate: {top3_mean:.4f} ± {top3_std:.4f}")

    # Ceiling: the best any classifier could do on this synthetic task
    ceil_top1, ceil_top3 = _nearest_prototype_ceiling(ml)
    print(f"\n  Ceiling (nearest prototype, sigma={MLRecommender._NOISE_STD}): "
          f"top-1 {ceil_top1:.4f} | top-3 {ceil_top3:.4f}")

    # ── 4. Pick best model ────────────────────────────────────────────────
    best = max(results, key=lambda r: r["cv_accuracy_mean"])

    # ── 5. Save best model ────────────────────────────────────────────────
    _print_section("Saving Best Model")

    # Retrain best model type so ml.model is the winner
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.train(model_type=best["model_type"])

    ml.save(str(MODEL_SAVE_PATH))
    print(f"  Best model type : {best['model_type'].upper()}")
    print(f"  CV accuracy     : {best['cv_accuracy_mean']:.4f} "
          f"± {best['cv_accuracy_std']:.4f}")
    print(f"  Saved to        : {MODEL_SAVE_PATH}")

    # ── 6. Write CSV ──────────────────────────────────────────────────────
    _print_section("Writing Results CSV")

    csv_columns = [
        "model_type",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "vs_rule_mae",
        "cv_top3_mean",
        "cv_top3_std",
    ]

    with open(CSV_SAVE_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_columns)
        writer.writeheader()
        for r in results:
            writer.writerow({col: r[col] for col in csv_columns})
        writer.writerow({
            "model_type": "ceiling_nearest_prototype",
            "cv_accuracy_mean": ceil_top1, "cv_accuracy_std": "",
            "vs_rule_mae": "", "cv_top3_mean": ceil_top3, "cv_top3_std": "",
        })

    print(f"  Written {len(results)} model rows + 1 ceiling row → {CSV_SAVE_PATH}")

    # ── 7. Final summary table ────────────────────────────────────────────
    _print_section("Summary – Model Comparison")

    col_w = [16, 18, 18, 18]
    headers = ["Model", "CV Accuracy", "± Std Dev", "vs-Rule MAE"]
    divider  = "  ├" + "┼".join("─" * w for w in col_w) + "┤"
    top_line = "  ┌" + "┬".join("─" * w for w in col_w) + "┐"
    bot_line = "  └" + "┴".join("─" * w for w in col_w) + "┘"

    def _row(cells: list[str], bold_col: int = -1) -> str:
        parts = []
        for i, (cell, w) in enumerate(zip(cells, col_w)):
            parts.append(cell.center(w))
        return "  │" + "│".join(parts) + "│"

    print()
    print(top_line)
    print(_row(headers))
    print(divider)

    for r in sorted(results, key=lambda x: x["cv_accuracy_mean"], reverse=True):
        is_best = r["model_type"] == best["model_type"]
        name_cell = ("★ " if is_best else "  ") + r["model_type"].upper()
        cells = [
            name_cell,
            f"{r['cv_accuracy_mean']:.4f}",
            f"± {r['cv_accuracy_std']:.4f}",
            f"{r['vs_rule_mae']:.6f}",
        ]
        print(_row(cells))

    print(bot_line)

    # Winner rationale
    runner_up_accs = [
        r["cv_accuracy_mean"]
        for r in results
        if r["model_type"] != best["model_type"]
    ]
    margin = best["cv_accuracy_mean"] - max(runner_up_accs)

    print()
    print(f"  ★  Winner : {best['model_type'].upper()}")
    print(f"     Reason : Highest 5-fold CV accuracy "
          f"({best['cv_accuracy_mean']:.4f}), leading the next-best "
          f"model by {margin:+.4f} accuracy points.")
    print(f"     MAE vs rule-based : {best['vs_rule_mae']:.6f} "
          "(agreement with geometric nearest-neighbour baseline)")
    print()

    # Interpretation note
    print("  ─── Metric Guide " + "─" * 54)
    print("  CV Accuracy  │ Higher is better. % of augmented VAD samples "
          "correctly")
    print("               │ classified to the right scent under 5-fold CV.")
    print("  vs-Rule MAE  │ Lower is better. Mean VAD distance between the "
          "ML model's")
    print("               │ top recommendation and the rule-based baseline's "
          "top pick.")
    print("               │ Zero means perfect agreement; max meaningful "
          "value ≈ √3 ≈ 1.73.")
    print("  " + "─" * 69)
    print()


if __name__ == "__main__":
    main()