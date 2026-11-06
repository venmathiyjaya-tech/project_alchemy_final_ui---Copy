"""
tests/test_recommender.py

Behavioural tests for RuleBasedRecommender and MLRecommender.

Five archetypal emotional states are tested, covering the corners and centre
of the VAD (Valence–Arousal–Dominance) cube:

    State         Valence  Arousal  Dominance
    ──────────────────────────────────────────
    Very Happy      high     high     high
    Very Calm       high     low      low
    Anxious         low      high     low
    Sad             low      low      low
    Neutral         mid      mid      mid

For each input both recommenders are queried and the result is printed with
an AGREE / DISAGREE verdict.

Usage
-----
    # Run as a plain script (no pytest needed):
    python tests/test_recommender.py

    # Or with pytest (verbose):
    pytest tests/test_recommender.py -v
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap – works whether called as a script or via pytest from root
# ---------------------------------------------------------------------------
_HERE         = Path(__file__).resolve().parent          # tests/
_PROJECT_ROOT = _HERE.parent                             # project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.recommender.recommender import MLRecommender, RuleBasedRecommender  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCENTS_PATH = _PROJECT_ROOT / "data"   / "scents.json"
MODEL_PATH  = _PROJECT_ROOT / "models" / "saved_model.joblib"   # canonical model path

# ---------------------------------------------------------------------------
# The five canonical test inputs
# ---------------------------------------------------------------------------
TEST_STATES: list[dict] = [
    {
        "label"      : "Very Happy",
        "description": "High valence + high arousal – elated, energised",
        "vad"        : {"valence": 0.90, "arousal": 0.88, "dominance": 0.80},
    },
    {
        "label"      : "Very Calm",
        "description": "High valence + low arousal – serene, at peace",
        "vad"        : {"valence": 0.88, "arousal": 0.12, "dominance": 0.35},
    },
    {
        "label"      : "Anxious",
        "description": "Low valence + high arousal – tense, on edge",
        "vad"        : {"valence": 0.18, "arousal": 0.85, "dominance": 0.20},
    },
    {
        "label"      : "Sad",
        "description": "Low valence + low arousal – melancholic, withdrawn",
        "vad"        : {"valence": 0.15, "arousal": 0.14, "dominance": 0.22},
    },
    {
        "label"      : "Neutral",
        "description": "All dimensions at midpoint – balanced, unremarkable",
        "vad"        : {"valence": 0.50, "arousal": 0.50, "dominance": 0.50},
    },
]

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rule_recommender() -> RuleBasedRecommender:
    """Module-scoped RuleBasedRecommender (loads once per test session)."""
    assert SCENTS_PATH.is_file(), (
        f"Scents file not found: {SCENTS_PATH}\n"
        "Run from the project root or ensure data/scents.json exists."
    )
    return RuleBasedRecommender(str(SCENTS_PATH))


@pytest.fixture(scope="module")
def ml_recommender() -> MLRecommender:
    """Module-scoped MLRecommender loaded from the saved model."""
    assert SCENTS_PATH.is_file(), f"Scents file not found: {SCENTS_PATH}"
    assert MODEL_PATH.is_file(), (
        f"Trained model not found: {MODEL_PATH}\n"
        "Run ml/recommender/train_recommender.py first."
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MLRecommender(str(SCENTS_PATH), model_path=str(MODEL_PATH))


# ---------------------------------------------------------------------------
# Formatting helpers (used by both the pytest tests and the __main__ block)
# ---------------------------------------------------------------------------
_W = 72   # total table width

def _header(title: str) -> None:
    print()
    print("╔" + "═" * (_W - 2) + "╗")
    print("║  " + title.ljust(_W - 4) + "║")
    print("╚" + "═" * (_W - 2) + "╝")


def _divider(char: str = "─") -> None:
    print("  " + char * (_W - 4))


def _scent_line(prefix: str, scent: dict, extra_key: Optional[str] = None) -> None:
    """Print one scent result line."""
    name   = scent.get("name", "?")
    brand  = scent.get("brand", "Vasya")
    sid    = scent.get("id", "?")
    tags   = ", ".join(scent.get("emotion_tags", []))
    extra  = ""
    if extra_key and extra_key in scent:
        extra = f"  [{extra_key}={scent[extra_key]:.4f}]"
    print(f"  {prefix:<14} [{sid}] {name} ({brand}){extra}")
    print(f"  {'':14}  tags: {tags}")


def _verdict(rb_scent: dict, ml_scent: dict) -> str:
    agree = rb_scent["id"] == ml_scent["id"]
    if agree:
        return "  ✅  AGREE   — both recommenders chose the same scent."
    else:
        return (
            f"  ⚡  DISAGREE — rule-based: '{rb_scent['name']}' "
            f"vs ML: '{ml_scent['name']}'"
        )


def _run_and_print_all(
    rb: RuleBasedRecommender,
    ml: MLRecommender,
) -> list[dict]:
    """
    Run both recommenders on every test state, print results, and return
    a list of result dicts for assertion use.
    """
    _header("Recommender Test Suite – Five Emotional States")

    outcomes: list[dict] = []

    for state in TEST_STATES:
        label = state["label"]
        desc  = state["description"]
        vad   = state["vad"]

        print(f"\n  {'●':1} {label.upper()}  —  {desc}")
        print(f"    VAD: valence={vad['valence']}, "
              f"arousal={vad['arousal']}, "
              f"dominance={vad['dominance']}")
        _divider()

        rb_rec = rb.recommend(vad)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ml_rec = ml.recommend(vad)

        _scent_line("Rule-Based →", rb_rec, extra_key="distance")
        _scent_line("ML Model   →", ml_rec)
        _divider("·")
        print(_verdict(rb_rec, ml_rec))

        outcomes.append({
            "label"  : label,
            "vad"    : vad,
            "rb_rec" : rb_rec,
            "ml_rec" : ml_rec,
            "agree"  : rb_rec["id"] == ml_rec["id"],
        })

    return outcomes


def _print_summary(outcomes: list[dict]) -> None:
    """Print the aggregate agreement summary table."""
    _header("Summary")

    col_label  = 16
    col_rb     = 22
    col_ml     = 22
    col_agree  = 10

    def _row(cells: list[str], widths: list[int]) -> str:
        parts = [c.ljust(w) for c, w in zip(cells, widths)]
        return "  │ " + " │ ".join(parts) + " │"

    top = "  ┌─" + "─┬─".join("─" * w for w in [col_label, col_rb, col_ml, col_agree]) + "─┐"
    mid = "  ├─" + "─┼─".join("─" * w for w in [col_label, col_rb, col_ml, col_agree]) + "─┤"
    bot = "  └─" + "─┴─".join("─" * w for w in [col_label, col_rb, col_ml, col_agree]) + "─┘"

    print()
    print(top)
    print(_row(["State", "Rule-Based Pick", "ML Pick", "Verdict"],
               [col_label, col_rb, col_ml, col_agree]))
    print(mid)

    n_agree = 0
    for o in outcomes:
        verdict = "✅ AGREE" if o["agree"] else "⚡ DISAGREE"
        if o["agree"]:
            n_agree += 1
        print(_row([
            o["label"],
            o["rb_rec"]["name"][:col_rb],
            o["ml_rec"]["name"][:col_ml],
            verdict,
        ], [col_label, col_rb, col_ml, col_agree]))

    print(bot)

    pct = 100 * n_agree / len(outcomes)
    print(f"\n  Agreement rate: {n_agree}/{len(outcomes)} states "
          f"({pct:.0f}%)")
    if n_agree == len(outcomes):
        print("  Both recommenders are in full agreement across all states. ✓")
    else:
        n_dis = len(outcomes) - n_agree
        print(f"  {n_dis} state(s) produced different recommendations — "
              "see per-state detail above.")
    print()


# ---------------------------------------------------------------------------
# pytest test functions
# ---------------------------------------------------------------------------

class TestRuleBasedRecommender:
    """Structural and behavioural tests for RuleBasedRecommender."""

    def test_loads_scents(self, rule_recommender):
        assert len(rule_recommender.scents) > 0, "Scent list must not be empty."

    @pytest.mark.parametrize("state", TEST_STATES, ids=[s["label"] for s in TEST_STATES])
    def test_recommend_returns_dict(self, rule_recommender, state):
        result = rule_recommender.recommend(state["vad"])
        assert isinstance(result, dict), "recommend() must return a dict."

    @pytest.mark.parametrize("state", TEST_STATES, ids=[s["label"] for s in TEST_STATES])
    def test_recommend_has_required_keys(self, rule_recommender, state):
        result = rule_recommender.recommend(state["vad"])
        for key in ("id", "name", "valence", "arousal", "dominance", "distance"):
            assert key in result, f"Missing key '{key}' in recommend() result."

    @pytest.mark.parametrize("state", TEST_STATES, ids=[s["label"] for s in TEST_STATES])
    def test_recommend_distance_is_non_negative(self, rule_recommender, state):
        result = rule_recommender.recommend(state["vad"])
        assert result["distance"] >= 0.0, "Distance must be non-negative."

    @pytest.mark.parametrize("state", TEST_STATES, ids=[s["label"] for s in TEST_STATES])
    def test_recommend_top3_length(self, rule_recommender, state):
        results = rule_recommender.recommend_top_n(state["vad"], n=3)
        assert len(results) == 3, "recommend_top_n(n=3) must return exactly 3 items."

    @pytest.mark.parametrize("state", TEST_STATES, ids=[s["label"] for s in TEST_STATES])
    def test_recommend_top3_sorted_ascending(self, rule_recommender, state):
        results = rule_recommender.recommend_top_n(state["vad"], n=3)
        dists = [r["distance"] for r in results]
        assert dists == sorted(dists), "Top-N results must be sorted by ascending distance."

    @pytest.mark.parametrize("state", TEST_STATES, ids=[s["label"] for s in TEST_STATES])
    def test_top1_matches_recommend(self, rule_recommender, state):
        """The top result from recommend_top_n should match recommend()."""
        top1  = rule_recommender.recommend_top_n(state["vad"], n=1)[0]
        best  = rule_recommender.recommend(state["vad"])
        assert top1["id"] == best["id"], (
            "Top-1 of recommend_top_n must match recommend()."
        )

    # ── Semantic sanity checks ────────────────────────────────────────────

    def test_very_happy_high_valence(self, rule_recommender):
        """Very-happy input should resolve to a high-valence scent (≥ 0.70)."""
        vad    = TEST_STATES[0]["vad"]   # Very Happy
        result = rule_recommender.recommend(vad)
        assert result["valence"] >= 0.70, (
            f"Expected a high-valence scent for 'Very Happy', "
            f"got valence={result['valence']} ({result['name']})."
        )

    def test_sad_low_valence(self, rule_recommender):
        """Sad input should resolve to one of the library's lowest-valence scents.

        The library has no scent below valence 0.55, so 'low' is relative: the pick must sit in
        the lowest quartile of scent valences (a fixed 0.55 cut-off only passed while the
        library's minimum happened to equal it)."""
        vad    = TEST_STATES[3]["vad"]   # Sad
        result = rule_recommender.recommend(vad)
        valences = sorted(s["valence"] for s in rule_recommender.scents)
        quartile = valences[len(valences) // 4]
        assert result["valence"] <= quartile, (
            f"Expected a low-valence scent for 'Sad' (<= {quartile}), "
            f"got valence={result['valence']} ({result['name']})."
        )

    def test_very_calm_low_arousal(self, rule_recommender):
        """Very-calm input should resolve to a low-arousal scent (≤ 0.50)."""
        vad    = TEST_STATES[1]["vad"]   # Very Calm
        result = rule_recommender.recommend(vad)
        assert result["arousal"] <= 0.50, (
            f"Expected a low-arousal scent for 'Very Calm', "
            f"got arousal={result['arousal']} ({result['name']})."
        )

    def test_very_happy_differs_from_sad(self, rule_recommender):
        """Very-happy and sad should not map to the same scent."""
        happy = rule_recommender.recommend(TEST_STATES[0]["vad"])
        sad   = rule_recommender.recommend(TEST_STATES[3]["vad"])
        assert happy["id"] != sad["id"], (
            "Very Happy and Sad should not recommend the same scent."
        )

    def test_very_happy_differs_from_very_calm(self, rule_recommender):
        """Very-happy (high arousal) and very-calm (low arousal) should differ."""
        happy = rule_recommender.recommend(TEST_STATES[0]["vad"])
        calm  = rule_recommender.recommend(TEST_STATES[1]["vad"])
        assert happy["id"] != calm["id"], (
            "Very Happy and Very Calm should not recommend the same scent "
            "(they differ primarily in arousal)."
        )


class TestMLRecommender:
    """Structural and behavioural tests for MLRecommender (pre-trained model)."""

    def test_model_loaded(self, ml_recommender):
        assert ml_recommender.model is not None, "Model must be loaded."

    @pytest.mark.parametrize("state", TEST_STATES, ids=[s["label"] for s in TEST_STATES])
    def test_recommend_returns_dict(self, ml_recommender, state):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ml_recommender.recommend(state["vad"])
        assert isinstance(result, dict), "recommend() must return a dict."

    @pytest.mark.parametrize("state", TEST_STATES, ids=[s["label"] for s in TEST_STATES])
    def test_recommend_has_required_keys(self, ml_recommender, state):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ml_recommender.recommend(state["vad"])
        for key in ("id", "name", "valence", "arousal", "dominance"):
            assert key in result, f"Missing key '{key}' in ML recommend() result."

    @pytest.mark.parametrize("state", TEST_STATES, ids=[s["label"] for s in TEST_STATES])
    def test_recommend_id_is_known(self, ml_recommender, state):
        """Predicted scent ID must exist in the loaded scents catalogue."""
        known_ids = {s["id"] for s in ml_recommender.scents}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ml_recommender.recommend(state["vad"])
        assert result["id"] in known_ids, (
            f"Predicted scent ID '{result['id']}' is not in the scents catalogue."
        )

    @pytest.mark.parametrize("state", TEST_STATES, ids=[s["label"] for s in TEST_STATES])
    def test_recommend_top3_length(self, ml_recommender, state):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = ml_recommender.recommend_top_n(state["vad"], n=3)
        assert len(results) == 3, "recommend_top_n(n=3) must return exactly 3 items."

    @pytest.mark.parametrize("state", TEST_STATES, ids=[s["label"] for s in TEST_STATES])
    def test_recommend_top3_probabilities_sum_le_1(self, ml_recommender, state):
        """Top-3 probabilities must sum to ≤ 1 (they are a subset of the full distribution)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = ml_recommender.recommend_top_n(state["vad"], n=3)
        if "probability" in results[0]:
            total = sum(r["probability"] for r in results)
            assert total <= 1.0 + 1e-6, (
                f"Top-3 probabilities sum to {total:.4f}, expected ≤ 1.0."
            )

    @pytest.mark.parametrize("state", TEST_STATES, ids=[s["label"] for s in TEST_STATES])
    def test_top1_matches_recommend(self, ml_recommender, state):
        """Top-1 of recommend_top_n should match recommend()."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            top3   = ml_recommender.recommend_top_n(state["vad"], n=3)
            single = ml_recommender.recommend(state["vad"])
        assert top3[0]["id"] == single["id"], (
            "Top-1 of recommend_top_n must match recommend()."
        )

    # ── Semantic sanity ───────────────────────────────────────────────────

    def test_very_happy_differs_from_sad(self, ml_recommender):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            happy = ml_recommender.recommend(TEST_STATES[0]["vad"])
            sad   = ml_recommender.recommend(TEST_STATES[3]["vad"])
        assert happy["id"] != sad["id"], (
            "ML model: Very Happy and Sad should not recommend the same scent."
        )


class TestCrossRecommenderAgreement:
    """Cross-system tests comparing rule-based and ML outputs."""

    @pytest.mark.parametrize("state", TEST_STATES, ids=[s["label"] for s in TEST_STATES])
    def test_both_return_known_scent_id(
        self, rule_recommender, ml_recommender, state
    ):
        known_ids = {s["id"] for s in rule_recommender.scents}
        rb_rec = rule_recommender.recommend(state["vad"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ml_rec = ml_recommender.recommend(state["vad"])
        assert rb_rec["id"] in known_ids
        assert ml_rec["id"] in known_ids

    def test_full_comparison_runs_without_error(
        self, rule_recommender, ml_recommender, capsys
    ):
        """
        Integration smoke-test: run the full print-and-compare loop on all
        five states and verify it completes without raising.
        """
        outcomes = _run_and_print_all(rule_recommender, ml_recommender)
        _print_summary(outcomes)
        assert len(outcomes) == len(TEST_STATES)


# ---------------------------------------------------------------------------
# __main__ block – rich printed output when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  Loading recommenders …", flush=True)

    if not SCENTS_PATH.is_file():
        sys.exit(f"\n[ERROR] Scents file not found: {SCENTS_PATH}")
    if not MODEL_PATH.is_file():
        sys.exit(
            f"\n[ERROR] Trained model not found: {MODEL_PATH}\n"
            "        Run ml/recommender/train_recommender.py first."
        )

    rb = RuleBasedRecommender(str(SCENTS_PATH))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml = MLRecommender(str(SCENTS_PATH), model_path=str(MODEL_PATH))

    print(f"  Rule-Based  : {len(rb.scents)} scents loaded.")
    print(f"  ML Model    : {ml._model_type.upper() if ml._model_type else 'loaded'} "
          f"from {MODEL_PATH.name}")

    outcomes = _run_and_print_all(rb, ml)
    _print_summary(outcomes)