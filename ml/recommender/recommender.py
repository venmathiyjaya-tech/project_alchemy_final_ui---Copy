"""
ml/recommender/recommender.py

Scent recommendation engine using VAD (Valence-Arousal-Dominance) affective space.

Two recommenders are provided:
  - RuleBasedRecommender : pure Euclidean nearest-neighbour search (no ML).
  - MLRecommender        : sklearn classifier (KNN / Random Forest / MLP) trained
                           on augmented VAD data, with a rule-based fallback.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import joblib


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euclidean(a: dict, b: dict) -> float:
    """Return the Euclidean distance between two VAD dicts."""
    return math.sqrt(
        (a["valence"] - b["valence"]) ** 2
        + (a["arousal"] - b["arousal"]) ** 2
        + (a["dominance"] - b["dominance"]) ** 2
    )


def _vad_array(vad_dict: dict) -> np.ndarray:
    """Convert a VAD dict to a (1, 3) numpy array."""
    return np.array([[vad_dict["valence"], vad_dict["arousal"], vad_dict["dominance"]]])


def _load_scents(path: str) -> list[dict]:
    """Load and validate a JSON scents file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Scents file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        scents = json.load(fh)
    if not isinstance(scents, list) or len(scents) == 0:
        raise ValueError("Scents JSON must be a non-empty list of dicts.")
    required_keys = {"id", "valence", "arousal", "dominance"}
    for s in scents:
        missing = required_keys - s.keys()
        if missing:
            raise ValueError(f"Scent '{s}' is missing keys: {missing}")
    return scents


# ---------------------------------------------------------------------------
# CLASS 1 – RuleBasedRecommender
# ---------------------------------------------------------------------------

class RuleBasedRecommender:
    """
    Nearest-neighbour recommender operating purely in 3-D VAD space.

    No model is trained; recommendations are determined by Euclidean distance
    to each scent's labelled VAD coordinates.

    Parameters
    ----------
    scents_path : str
        Path to a JSON file containing a list of scent dicts. Each dict must
        have at minimum the keys: ``id``, ``valence``, ``arousal``,
        ``dominance``.
    """

    def __init__(self, scents_path: str) -> None:
        self.scents: list[dict] = _load_scents(scents_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, vad_dict: dict) -> dict:
        """
        Find the single closest scent to *vad_dict* in VAD space.

        Parameters
        ----------
        vad_dict : dict
            Must contain ``valence``, ``arousal``, and ``dominance`` keys
            (floats in [0, 1]).

        Returns
        -------
        dict
            A copy of the matched scent dict with an extra ``'distance'`` key
            (float, Euclidean distance).
        """
        self._validate_vad(vad_dict)
        best_scent, best_dist = self._rank_all(vad_dict)[0]
        return {**best_scent, "distance": round(best_dist, 6)}

    def recommend_top_n(self, vad_dict: dict, n: int = 3) -> list[dict]:
        """
        Return the *n* closest scents to *vad_dict* in VAD space.

        Parameters
        ----------
        vad_dict : dict
            VAD coordinates (keys ``valence``, ``arousal``, ``dominance``).
        n : int
            Number of recommendations to return (default 3).

        Returns
        -------
        list[dict]
            List of scent dicts, each augmented with a ``'distance'`` key,
            sorted ascending by distance.
        """
        self._validate_vad(vad_dict)
        ranked = self._rank_all(vad_dict)
        return [{**scent, "distance": round(dist, 6)} for scent, dist in ranked[:n]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_vad(vad_dict: dict) -> None:
        for key in ("valence", "arousal", "dominance"):
            if key not in vad_dict:
                raise KeyError(f"vad_dict is missing required key: '{key}'")
            val = vad_dict[key]
            if not isinstance(val, (int, float)):
                raise TypeError(f"vad_dict['{key}'] must be numeric, got {type(val)}.")

    def _rank_all(self, vad_dict: dict) -> list[tuple[dict, float]]:
        """Return all scents sorted by ascending distance to *vad_dict*."""
        scored = [
            (scent, _euclidean(vad_dict, scent))
            for scent in self.scents
        ]
        scored.sort(key=lambda pair: pair[1])
        return scored


# ---------------------------------------------------------------------------
# CLASS 2 – MLRecommender
# ---------------------------------------------------------------------------

class MLRecommender:
    """
    Sklearn-based scent recommender trained on augmented VAD data.

    A small Gaussian-noise augmentation step is applied to each scent's VAD
    coordinates to create a richer training set (50 synthetic samples per
    scent by default).

    Parameters
    ----------
    scents_path : str
        Path to a JSON file containing scent dicts (same format as
        ``RuleBasedRecommender``).
    model_path : str, optional
        If provided, a previously saved sklearn pipeline is loaded from this
        path via ``joblib``. ``train()`` is not required when a saved model
        is supplied.
    """

    _SUPPORTED_MODELS = ("knn", "random_forest", "mlp")
    _AUGMENT_N = 50          # synthetic samples per scent
    _NOISE_STD = 0.05        # Gaussian noise std for augmentation
    _CV_FOLDS = 5

    def __init__(self, scents_path: str, model_path: Optional[str] = None) -> None:
        self.scents: list[dict] = _load_scents(scents_path)
        self._scent_by_id: dict[str, dict] = {s["id"]: s for s in self.scents}

        self.model = None
        self._label_encoder = LabelEncoder()
        self._model_type: Optional[str] = None

        # Fit the label encoder on all known IDs so it is always available.
        self._label_encoder.fit([s["id"] for s in self.scents])

        if model_path is not None:
            self._load(model_path)

        # Fallback recommender (used when predict_proba is unavailable).
        self._rule_based = RuleBasedRecommender(scents_path)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def build_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct augmented training data from the loaded scents.

        Each scent contributes 1 original sample plus ``_AUGMENT_N`` synthetic
        samples created by adding zero-mean Gaussian noise (std ``_NOISE_STD``)
        to its VAD coordinates. Values are clipped to [0, 1].

        Returns
        -------
        X : np.ndarray, shape (n_scents * (1 + AUGMENT_N), 3)
            Feature matrix [valence, arousal, dominance].
        y : np.ndarray, shape (n_scents * (1 + AUGMENT_N),)
            Scent ID labels (strings).
        """
        rng = np.random.default_rng(seed=42)
        X_rows: list[list[float]] = []
        y_rows: list[str] = []

        for scent in self.scents:
            vad = np.array([scent["valence"], scent["arousal"], scent["dominance"]])
            scent_id = scent["id"]

            # Original sample
            X_rows.append(vad.tolist())
            y_rows.append(scent_id)

            # Synthetic augmented samples
            noise = rng.normal(loc=0.0, scale=self._NOISE_STD,
                               size=(self._AUGMENT_N, 3))
            synthetic = np.clip(vad + noise, 0.0, 1.0)
            X_rows.extend(synthetic.tolist())
            y_rows.extend([scent_id] * self._AUGMENT_N)

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_rows, dtype=object)
        return X, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, model_type: str = "knn") -> dict:
        """
        Train a classifier on the augmented VAD dataset.

        Parameters
        ----------
        model_type : {'knn', 'random_forest', 'mlp'}
            Which sklearn estimator to train.

        Returns
        -------
        dict
            ``{'model_type': str, 'cv_scores': list[float],
               'mean_accuracy': float, 'std_accuracy': float}``
        """
        model_type = model_type.lower()
        if model_type not in self._SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Choose from: {self._SUPPORTED_MODELS}"
            )

        X, y = self.build_training_data()
        y_enc = self._label_encoder.transform(y)

        clf = self._build_classifier(model_type)

        cv = StratifiedKFold(n_splits=self._CV_FOLDS, shuffle=True, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_scores = cross_val_score(clf, X, y_enc, cv=cv, scoring="accuracy")

        # Fit on full dataset after evaluation
        clf.fit(X, y_enc)
        self.model = clf
        self._model_type = model_type

        result = {
            "model_type": model_type,
            "cv_scores": cv_scores.tolist(),
            "mean_accuracy": float(cv_scores.mean()),
            "std_accuracy": float(cv_scores.std()),
        }
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the trained model and label encoder to *path* via joblib.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``'models/ml_recommender.joblib'``).
        """
        if self.model is None:
            raise RuntimeError("No model trained yet. Call train() first.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "model": self.model,
            "label_encoder": self._label_encoder,
            "model_type": self._model_type,
        }
        joblib.dump(payload, path)
        print(f"[MLRecommender] Model saved → {path}")

    def _load(self, path: str) -> None:
        """Load a previously saved model from *path*."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        payload = joblib.load(path)
        self.model = payload["model"]
        self._label_encoder = payload["label_encoder"]
        self._model_type = payload.get("model_type")
        print(f"[MLRecommender] Model loaded ← {path}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend(self, vad_dict: dict) -> dict:
        """
        Predict the best-matching scent for *vad_dict*.

        Parameters
        ----------
        vad_dict : dict
            VAD coordinates (``valence``, ``arousal``, ``dominance``).

        Returns
        -------
        dict
            The matched scent dict (no ``'distance'`` key; the model predicts
            a class label rather than measuring distance).

        Raises
        ------
        RuntimeError
            If no model has been trained or loaded.
        """
        self._require_model()
        RuleBasedRecommender._validate_vad(vad_dict)
        X = _vad_array(vad_dict)
        pred_enc = self.model.predict(X)[0]
        scent_id = self._label_encoder.inverse_transform([pred_enc])[0]
        return dict(self._scent_by_id[scent_id])

    def recommend_top_n(self, vad_dict: dict, n: int = 3) -> list[dict]:
        """
        Return the top *n* scent recommendations for *vad_dict*.

        If the underlying model exposes ``predict_proba``, class probabilities
        are used to rank results.  Otherwise falls back to the rule-based
        Euclidean distance ranking.

        Parameters
        ----------
        vad_dict : dict
            VAD coordinates (``valence``, ``arousal``, ``dominance``).
        n : int
            Number of recommendations (default 3).

        Returns
        -------
        list[dict]
            Scent dicts augmented with either a ``'probability'`` key (ML
            path) or a ``'distance'`` key (fallback path), sorted by
            relevance descending / distance ascending.
        """
        self._require_model()
        RuleBasedRecommender._validate_vad(vad_dict)

        if hasattr(self.model, "predict_proba"):
            return self._top_n_proba(vad_dict, n)

        # Fallback: rule-based distance ranking
        warnings.warn(
            f"Model '{self._model_type}' does not support predict_proba. "
            "Falling back to RuleBasedRecommender for top-N.",
            RuntimeWarning,
            stacklevel=2,
        )
        return self._rule_based.recommend_top_n(vad_dict, n=n)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "No model is available. Call train() or pass a model_path to __init__."
            )

    @staticmethod
    def _build_classifier(model_type: str):
        if model_type == "knn":
            return KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
            )
        if model_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=500,
                random_state=42,
            )
        raise ValueError(f"Unknown model_type: {model_type}")

    def _top_n_proba(self, vad_dict: dict, n: int) -> list[dict]:
        """Use class probabilities to rank top-N scents."""
        X = _vad_array(vad_dict)
        proba = self.model.predict_proba(X)[0]          # shape: (n_classes,)
        classes_enc = self.model.classes_               # encoded label array

        # Pair each class with its probability and sort descending
        ranked = sorted(zip(classes_enc, proba), key=lambda p: p[1], reverse=True)

        results = []
        for enc_label, prob in ranked[:n]:
            scent_id = self._label_encoder.inverse_transform([enc_label])[0]
            results.append({**self._scent_by_id[scent_id], "probability": round(float(prob), 6)})
        return results


# ---------------------------------------------------------------------------
# Main – comparison block
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    width = 70
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def _fmt_scent(s: dict, extra_key: Optional[str] = None) -> str:
    base = f"  [{s['id']}] {s['name']} ({s.get('family', '?')})"
    if extra_key and extra_key in s:
        base += f"  │  {extra_key}={s[extra_key]:.4f}"
    return base


if __name__ == "__main__":
    import sys

    SCENTS_PATH = os.path.join(os.path.dirname(__file__), "scents.json")
    MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "saved_model.joblib")

    # Five diverse test inputs spanning the VAD cube
    TEST_INPUTS: list[dict] = [
        {"valence": 0.85, "arousal": 0.20, "dominance": 0.50, "_label": "Calm & Happy"},
        {"valence": 0.90, "arousal": 0.90, "dominance": 0.75, "_label": "Excited & Energetic"},
        {"valence": 0.30, "arousal": 0.25, "dominance": 0.30, "_label": "Sad & Subdued"},
        {"valence": 0.50, "arousal": 0.55, "dominance": 0.85, "_label": "Confident & Neutral"},
        {"valence": 0.70, "arousal": 0.50, "dominance": 0.50, "_label": "Balanced & Neutral"},
    ]

    # ── 1. Rule-Based Recommender ─────────────────────────────────────────
    _print_header("Rule-Based Recommender")
    rb = RuleBasedRecommender(SCENTS_PATH)
    print(f"  Loaded {len(rb.scents)} scents from '{SCENTS_PATH}'.\n")

    for inp in TEST_INPUTS:
        vad = {k: v for k, v in inp.items() if not k.startswith("_")}
        label = inp["_label"]
        top3 = rb.recommend_top_n(vad, n=3)
        print(f"  ● {label}  VAD={list(vad.values())}")
        for rank, s in enumerate(top3, 1):
            print(f"    {rank}. {_fmt_scent(s, 'distance')}")
        print()

    # ── 2. ML Recommender – train all three model types ───────────────────
    _print_header("ML Recommender – Training")
    ml = MLRecommender(SCENTS_PATH)
    X, y = ml.build_training_data()
    print(f"  Training set: {X.shape[0]} samples × {X.shape[1]} features "
          f"({len(ml.scents)} scents × {MLRecommender._AUGMENT_N + 1} each)\n")

    for mtype in MLRecommender._SUPPORTED_MODELS:
        scores = ml.train(model_type=mtype)
        print(f"  {mtype.upper():>13s}  │  "
              f"CV accuracy: {scores['mean_accuracy']:.4f} "
              f"± {scores['std_accuracy']:.4f}  │  "
              f"folds: {[f'{s:.3f}' for s in scores['cv_scores']]}")

    # Persist last-trained model (MLP) for demonstration
    ml.save(MODEL_SAVE_PATH)

    # ── 3. ML Recommender – inference & comparison ────────────────────────
    _print_header("Side-by-Side Comparison (KNN model)")
    ml.train(model_type="knn")   # retrain with KNN for the comparison

    col_w = 34
    header_row = f"  {'Input State':<22} │ {'Rule-Based (top-1)':<{col_w}} │ ML Top-3 (predict_proba)"
    print(header_row)
    print("  " + "─" * (len(header_row) - 2))

    for inp in TEST_INPUTS:
        vad = {k: v for k, v in inp.items() if not k.startswith("_")}
        label = inp["_label"]

        rb_rec = rb.recommend(vad)
        ml_top3 = ml.recommend_top_n(vad, n=3)

        rb_str = f"{rb_rec['name']}  (d={rb_rec['distance']:.3f})"
        ml_strs = [f"{s['name']} ({s['probability']:.2f})" for s in ml_top3]

        print(f"  {label:<22} │ {rb_str:<{col_w}} │ {ml_strs[0]}")
        if len(ml_strs) > 1:
            for ms in ml_strs[1:]:
                print(f"  {'':<22}   {'':<{col_w}}   {ms}")
        print()

    # ── 4. Load from disk and verify ─────────────────────────────────────
    _print_header("Loading Saved Model and Verifying")
    ml2 = MLRecommender(SCENTS_PATH, model_path=MODEL_SAVE_PATH)
    sample_vad = TEST_INPUTS[0]
    vad = {k: v for k, v in sample_vad.items() if not k.startswith("_")}
    result = ml2.recommend(vad)
    print(f"  Test input : {vad}")
    print(f"  Recommended: [{result['id']}] {result['name']}")
    print()
    print("  Done. ✓")
    print("═" * 70)

    sys.exit(0)