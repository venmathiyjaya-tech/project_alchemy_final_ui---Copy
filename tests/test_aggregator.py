"""
tests/test_aggregator.py
========================
Standalone test script for :class:`ml.temporal.aggregator.TemporalAggregator`.

Run directly::

    python tests/test_aggregator.py

Each test is a plain function that prints a structured report and returns
``True`` (pass) or ``False`` (fail).  A final summary shows the overall
result.  No external test framework is required.

Test catalogue
--------------
1.  Default weights                  – constructor stores the correct ramp.
2.  Custom weights                   – constructor accepts a valid custom list.
3.  Invalid weights (length)         – ValueError on wrong-length list.
4.  Invalid weights (negative)       – ValueError on negative element.
5.  aggregate_emotions (7 days)      – recency bias: result closer to day 7.
6.  aggregate_vad (7 days)           – recency bias: result closer to day 7.
7.  aggregate_emotions (short window)– 3-day slice uses the correct tail weights.
8.  aggregate_emotions (empty)       – empty input → empty output.
9.  aggregate_vad (empty)            – empty input → empty output.
10. get_dominant_emotions            – correct top-N ranking.
11. get_dominant_emotions (top_n > len)– returns all labels, no crash.
12. summarise (7 entries)            – aggregated output structure is correct.
13. summarise (>7 entries)           – only last 7 are used (entry_count == 7).
14. Uniform weights baseline         – equal weights → result equals simple mean.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or from inside tests/
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.temporal.aggregator import TemporalAggregator  # noqa: E402

# ============================================================
# Shared fixtures
# ============================================================

# 7 daily emotion dicts (oldest → newest).
# Day 1: heavily sad.  Day 7: heavily joyful.
# The progression is deliberate so recency-bias assertions are obvious.
EMOTION_DAYS: list[dict[str, float]] = [
    # Day 1 – dominated by sadness
    {"sadness": 0.75, "anger": 0.10, "fear": 0.08, "joy": 0.05, "surprise": 0.02},
    # Day 2 – still sad but anger rising
    {"sadness": 0.60, "anger": 0.18, "fear": 0.10, "joy": 0.08, "surprise": 0.04},
    # Day 3 – mixed, sadness fading
    {"sadness": 0.40, "anger": 0.20, "fear": 0.15, "joy": 0.18, "surprise": 0.07},
    # Day 4 – neutral transition
    {"sadness": 0.25, "anger": 0.15, "fear": 0.12, "joy": 0.38, "surprise": 0.10},
    # Day 5 – joy taking over
    {"sadness": 0.12, "anger": 0.08, "fear": 0.08, "joy": 0.60, "surprise": 0.12},
    # Day 6 – mostly joyful
    {"sadness": 0.06, "anger": 0.04, "fear": 0.05, "joy": 0.75, "surprise": 0.10},
    # Day 7 (today) – overwhelmingly joyful
    {"sadness": 0.02, "anger": 0.02, "fear": 0.03, "joy": 0.88, "surprise": 0.05},
]

# 7 daily VAD dicts (oldest → newest).
# Day 1: low valence, high arousal (distressed).  Day 7: high valence, calm.
VAD_DAYS: list[dict[str, float]] = [
    {"valence": 0.15, "arousal": 0.80, "dominance": 0.25},  # Day 1
    {"valence": 0.22, "arousal": 0.72, "dominance": 0.30},  # Day 2
    {"valence": 0.35, "arousal": 0.65, "dominance": 0.38},  # Day 3
    {"valence": 0.50, "arousal": 0.55, "dominance": 0.50},  # Day 4
    {"valence": 0.65, "arousal": 0.45, "dominance": 0.60},  # Day 5
    {"valence": 0.78, "arousal": 0.35, "dominance": 0.68},  # Day 6
    {"valence": 0.90, "arousal": 0.28, "dominance": 0.75},  # Day 7
]

DAY1_EMOTIONS = EMOTION_DAYS[0]
DAY7_EMOTIONS = EMOTION_DAYS[-1]
DAY1_VAD = VAD_DAYS[0]
DAY7_VAD = VAD_DAYS[-1]

# ============================================================
# Helpers
# ============================================================

def _dist(a: float, b: float) -> float:
    """Absolute distance between two scalars."""
    return abs(a - b)


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _result(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    line = f"  {status}  {name}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    return passed


def _safe_run(fn):
    """Execute *fn()* and return (return_value, exception_or_None)."""
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001
        return None, exc


# ============================================================
# Individual tests
# ============================================================

def test_default_weights() -> None:
    _section("Test 1 · Default weights")
    agg = TemporalAggregator()
    expected = [0.4, 0.5, 0.6, 0.7, 0.85, 0.95, 1.0]
    ok = agg.weights == expected
    print(f"  weights : {agg.weights}")
    print(f"  expected: {expected}")
    assert _result("Default weights match expected ramp", ok)


def test_custom_weights() -> None:
    _section("Test 2 · Custom weights")
    custom = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    agg = TemporalAggregator(weights=custom)
    ok = agg.weights == custom
    print(f"  stored weights: {agg.weights}")
    assert _result("Custom weights stored correctly", ok)


def test_invalid_weights_length() -> None:
    _section("Test 3 · Invalid weights – wrong length")
    _, exc = _safe_run(lambda: TemporalAggregator(weights=[1.0, 2.0]))
    ok = isinstance(exc, ValueError)
    print(f"  exception raised: {type(exc).__name__ if exc else 'None'}")
    print(f"  message         : {exc}")
    assert _result("ValueError raised for wrong-length weights", ok)


def test_invalid_weights_negative() -> None:
    _section("Test 4 · Invalid weights – negative value")
    bad = [0.4, 0.5, 0.6, -0.1, 0.85, 0.95, 1.0]
    _, exc = _safe_run(lambda: TemporalAggregator(weights=bad))
    ok = isinstance(exc, ValueError)
    print(f"  exception raised: {type(exc).__name__ if exc else 'None'}")
    print(f"  message         : {exc}")
    assert _result("ValueError raised for negative weight", ok)


def test_aggregate_emotions_recency_bias() -> None:
    """
    Core recency-bias test for emotions.

    Recency weighting guarantees that the weighted average is pulled toward
    Day 7 *relative to the uniform/simple mean*.  Concretely:

    - For labels that **rise** across the week (joy: 0.05→0.88) the
      weighted average must be *above* the simple arithmetic mean.
    - For labels that **fall** across the week (sadness: 0.75→0.02) the
      weighted average must be *below* the simple arithmetic mean.

    Labels that are nearly flat (anger, fear, surprise) are shown in the
    table but not used in the pass/fail decision, because for flat series
    the weighted average is legitimately indistinguishable from the mean.

    Additionally we verify the two dominant labels directly:
      - ``joy``     (the rising label) should be the #1 dominant emotion.
      - ``sadness`` (the falling label) should not be #1 dominant emotion.
    """
    _section("Test 5 · aggregate_emotions – recency bias (7 days)")
    agg      = TemporalAggregator()
    uniform  = TemporalAggregator(weights=[1.0] * 7)   # baseline: simple mean

    result      = agg.aggregate_emotions(EMOTION_DAYS)
    simple_mean = uniform.aggregate_emotions(EMOTION_DAYS)

    # Classify each label by its trend across the week
    # A label "rises" if day7 > day1 by more than a small epsilon.
    TREND_THRESHOLD = 0.05
    rising  = {k for k in result if DAY7_EMOTIONS.get(k, 0.0) - DAY1_EMOTIONS.get(k, 0.0) >  TREND_THRESHOLD}
    falling = {k for k in result if DAY1_EMOTIONS.get(k, 0.0) - DAY7_EMOTIONS.get(k, 0.0) >  TREND_THRESHOLD}

    print(f"\n  Trend classification  (threshold Δ>{TREND_THRESHOLD}):")
    print(f"    Rising  labels: {sorted(rising)}")
    print(f"    Falling labels: {sorted(falling)}")

    print(f"\n  {'Label':<12} {'Day1':>8} {'SimpleMean':>12} {'Weighted':>12} {'Day7':>8}  "
          f"{'Trend':>8}  {'Check':>24}  {'OK?':>5}")
    print(f"  {'─'*12} {'─'*8} {'─'*12} {'─'*12} {'─'*8}  {'─'*8}  {'─'*24}  {'─'*5}")

    checks: list[bool] = []
    for label in sorted(result):
        wt_val  = result[label]
        sm_val  = simple_mean[label]
        d1_val  = DAY1_EMOTIONS.get(label, 0.0)
        d7_val  = DAY7_EMOTIONS.get(label, 0.0)

        if label in rising:
            trend    = "rising"
            ok_label = wt_val > sm_val          # weighted > mean  (pulled toward high day7)
            check    = f"weighted({wt_val:.4f}) > mean({sm_val:.4f})"
        elif label in falling:
            trend    = "falling"
            ok_label = wt_val < sm_val          # weighted < mean  (pulled toward low day7)
            check    = f"weighted({wt_val:.4f}) < mean({sm_val:.4f})"
        else:
            trend    = "flat"
            ok_label = True                     # flat series: no directional requirement
            check    = "flat – not tested"

        checks.append(ok_label)
        flag = "✅" if ok_label else "❌"
        print(f"  {label:<12} {d1_val:>8.3f} {sm_val:>12.6f} {wt_val:>12.6f} {d7_val:>8.3f}  "
              f"{trend:>8}  {check:>24}  {flag}")

    # Dominant-emotion sanity check
    dominant = agg.get_dominant_emotions(result, top_n=1)[0]
    joy_is_top = dominant == "joy"
    checks.append(joy_is_top)
    print(f"\n  Dominant emotion: '{dominant}'  (expected 'joy')  {'✅' if joy_is_top else '❌'}")
    print(f"\n  Aggregated emotions: {result}")

    all_ok = all(checks)
    assert _result(
        "Recency bias pulls rising labels above mean and falling labels below mean",
        all_ok,
        detail=(
            "Rising labels must have weighted > simple-mean; "
            "falling labels must have weighted < simple-mean. "
            "'joy' must be the dominant emotion."
        ),
    )


def test_aggregate_vad_recency_bias() -> None:
    """
    Core recency-bias test for VAD dimensions.

    Valence rises from 0.15 → 0.90 across the week; the weighted average
    must sit in the upper half of that range.  Similarly arousal falls and
    the average must be in the lower half.
    """
    _section("Test 6 · aggregate_vad – recency bias (7 days)")
    agg = TemporalAggregator()
    result = agg.aggregate_vad(VAD_DAYS)

    print(f"\n  {'Dim':<12} {'Day1':>8} {'Aggregated':>12} {'Day7':>8}  "
          f"{'Closer to':>12}  {'OK?':>5}")
    print(f"  {'─'*12} {'─'*8} {'─'*12} {'─'*8}  {'─'*12}  {'─'*5}")

    all_ok = True
    for dim in ("valence", "arousal", "dominance"):
        agg_val  = result[dim]
        day1_val = DAY1_VAD[dim]
        day7_val = DAY7_VAD[dim]
        d1       = _dist(agg_val, day1_val)
        d7       = _dist(agg_val, day7_val)
        closer   = "day7" if d7 < d1 else "day1"
        ok_label = closer == "day7"
        all_ok   = all_ok and ok_label
        flag     = "✅" if ok_label else "❌"
        print(f"  {dim:<12} {day1_val:>8.3f} {agg_val:>12.6f} {day7_val:>8.3f}  "
              f"{closer:>12}  {flag}")

    print(f"\n  Aggregated VAD: {result}")
    assert _result(
        "Every VAD dimension is closer to day 7 than day 1",
        all_ok,
    )


def test_aggregate_emotions_short_window() -> None:
    """
    Verify that a 3-day window uses weights[-3:] = [0.85, 0.95, 1.0].

    We hand-compute the expected result and compare with the method output.
    """
    _section("Test 7 · aggregate_emotions – short window (3 days)")
    agg = TemporalAggregator()
    three_days = EMOTION_DAYS[-3:]          # days 5, 6, 7
    w = [0.85, 0.95, 1.0]
    total_w = sum(w)

    all_labels = set()
    for d in three_days:
        all_labels.update(d.keys())

    expected: dict[str, float] = {}
    for label in all_labels:
        ws = sum(d.get(label, 0.0) * wi for d, wi in zip(three_days, w))
        expected[label] = round(ws / total_w, 6)

    result = agg.aggregate_emotions(three_days)

    print(f"\n  Expected : {expected}")
    print(f"  Got      : {result}")

    ok = all(abs(result.get(k, 0.0) - expected[k]) < 1e-9 for k in expected)
    assert _result(
        "3-day result matches hand-computed weighted average",
        ok,
        detail="weights used: [0.85, 0.95, 1.0]",
    )


def test_aggregate_emotions_empty() -> None:
    _section("Test 8 · aggregate_emotions – empty input")
    agg = TemporalAggregator()
    result = agg.aggregate_emotions([])
    ok = result == {}
    print(f"  result: {result}")
    assert _result("Empty input returns empty dict", ok)


def test_aggregate_vad_empty() -> None:
    _section("Test 9 · aggregate_vad – empty input")
    agg = TemporalAggregator()
    result = agg.aggregate_vad([])
    ok = result == {}
    print(f"  result: {result}")
    assert _result("Empty input returns empty dict", ok)


def test_get_dominant_emotions_ranking() -> None:
    _section("Test 10 · get_dominant_emotions – ranking")
    agg = TemporalAggregator()
    scores = {"joy": 0.70, "sadness": 0.05, "anger": 0.15, "fear": 0.08, "surprise": 0.02}
    top3 = agg.get_dominant_emotions(scores, top_n=3)
    expected = ["joy", "anger", "fear"]
    ok = top3 == expected
    print(f"  input scores : {scores}")
    print(f"  top-3 result : {top3}")
    print(f"  expected     : {expected}")
    assert _result("Top-3 labels returned in correct descending order", ok)


def test_get_dominant_emotions_top_n_exceeds_size() -> None:
    _section("Test 11 · get_dominant_emotions – top_n larger than dict")
    agg = TemporalAggregator()
    scores = {"joy": 0.8, "sadness": 0.2}
    result = agg.get_dominant_emotions(scores, top_n=10)
    ok = set(result) == {"joy", "sadness"} and len(result) == 2
    print(f"  input  : {scores}")
    print(f"  result : {result}")
    assert _result("Returns all labels without error when top_n > len(dict)", ok)


def test_summarise_structure() -> None:
    """
    Build 7 fake JournalEntry-like objects and call summarise().
    Verify the returned dict has the expected keys, types, and entry_count.
    """
    _section("Test 12 · summarise – output structure (7 entries)")

    class FakeEntry:
        def __init__(self, ev, v, a, d):
            self.emotion_vector = json.dumps(ev)
            self.vad_valence    = v
            self.vad_arousal    = a
            self.vad_dominance  = d

    entries = [
        FakeEntry(EMOTION_DAYS[i], VAD_DAYS[i]["valence"],
                  VAD_DAYS[i]["arousal"], VAD_DAYS[i]["dominance"])
        for i in range(7)
    ]

    agg    = TemporalAggregator()
    result = agg.summarise(entries)

    import pprint
    print()
    pprint.pprint(result, indent=4)

    checks = {
        "has 'aggregated_emotions' key"  : "aggregated_emotions" in result,
        "aggregated_emotions is dict"    : isinstance(result.get("aggregated_emotions"), dict),
        "has 'aggregated_vad' key"       : "aggregated_vad" in result,
        "aggregated_vad has 3 dims"      : set(result.get("aggregated_vad", {}).keys()) >= {"valence","arousal","dominance"},
        "has 'dominant_emotions' key"    : "dominant_emotions" in result,
        "dominant_emotions is list"      : isinstance(result.get("dominant_emotions"), list),
        "dominant_emotions has 3 items"  : len(result.get("dominant_emotions", [])) == 3,
        "entry_count == 7"               : result.get("entry_count") == 7,
    }

    print()
    all_ok = True
    for desc, ok in checks.items():
        flag    = "✅" if ok else "❌"
        all_ok  = all_ok and ok
        print(f"  {flag}  {desc}")

    assert _result("summarise() returns well-formed summary dict", all_ok)


def test_summarise_truncates_to_7() -> None:
    """
    Passing more than 7 entries must silently use only the last 7.
    """
    _section("Test 13 · summarise – >7 entries are truncated")

    class FakeEntry:
        def __init__(self, ev, v, a, d):
            self.emotion_vector = json.dumps(ev)
            self.vad_valence    = v
            self.vad_arousal    = a
            self.vad_dominance  = d

    # 10 entries: first 3 are dummies, last 7 are our real data
    dummy = FakeEntry({"joy": 0.5, "sadness": 0.5}, 0.5, 0.5, 0.5)
    real_entries = [
        FakeEntry(EMOTION_DAYS[i], VAD_DAYS[i]["valence"],
                  VAD_DAYS[i]["arousal"], VAD_DAYS[i]["dominance"])
        for i in range(7)
    ]
    entries_10 = [dummy, dummy, dummy] + real_entries

    agg    = TemporalAggregator()
    result = agg.summarise(entries_10)

    ok = result["entry_count"] == 7
    print(f"  total entries passed : {len(entries_10)}")
    print(f"  entry_count in result: {result['entry_count']}")
    assert _result("entry_count == 7 even when >7 entries are supplied", ok)


def test_uniform_weights_equals_simple_mean() -> None:
    """
    When all weights are equal (1.0) the weighted average must equal the
    arithmetic mean to within floating-point precision.
    """
    _section("Test 14 · Uniform weights → result equals simple mean")
    uniform_agg = TemporalAggregator(weights=[1.0] * 7)
    result = uniform_agg.aggregate_emotions(EMOTION_DAYS)

    # Compute plain arithmetic mean manually
    all_labels = set()
    for d in EMOTION_DAYS:
        all_labels.update(d.keys())
    simple_mean = {
        label: round(sum(d.get(label, 0.0) for d in EMOTION_DAYS) / 7, 6)
        for label in all_labels
    }

    print(f"\n  Uniform-weight result : {result}")
    print(f"  Simple mean (expected): {simple_mean}")

    ok = all(abs(result.get(k, 0.0) - simple_mean[k]) < 1e-9 for k in simple_mean)
    assert _result(
        "Uniform weights yield the arithmetic mean",
        ok,
        detail="Confirms the weighting formula is mathematically correct.",
    )


# ============================================================
# Runner
# ============================================================

def main() -> None:
    print("=" * 60)
    print("  TemporalAggregator – Test Suite")
    print("=" * 60)

    tests = [
        test_default_weights,
        test_custom_weights,
        test_invalid_weights_length,
        test_invalid_weights_negative,
        test_aggregate_emotions_recency_bias,
        test_aggregate_vad_recency_bias,
        test_aggregate_emotions_short_window,
        test_aggregate_emotions_empty,
        test_aggregate_vad_empty,
        test_get_dominant_emotions_ranking,
        test_get_dominant_emotions_top_n_exceeds_size,
        test_summarise_structure,
        test_summarise_truncates_to_7,
        test_uniform_weights_equals_simple_mean,
    ]

    results: list[tuple[str, bool]] = []
    for test_fn in tests:
        try:
            test_fn()          # the tests now use assert, so no exception means PASS
            passed = True
        except Exception:  # noqa: BLE001
            print(f"\n  💥 EXCEPTION in {test_fn.__name__}:")
            traceback.print_exc()
            passed = False
        results.append((test_fn.__name__, passed))

    # ── Final summary ──────────────────────────────────────────
    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed

    print(f"\n{'=' * 60}")
    print(f"  RESULTS  –  {passed}/{total} passed   {failed} failed")
    print(f"{'=' * 60}")
    for name, ok in results:
        flag = "✅" if ok else "❌"
        print(f"  {flag}  {name}")

    print()
    if failed == 0:
        print("  🎉  ALL TESTS PASSED")
    else:
        print(f"  ⚠️   {failed} TEST(S) FAILED – see details above")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()