"""
ml/temporal/aggregator.py
=========================
Temporal aggregation utilities for emotion and VAD (Valence-Arousal-Dominance)
signals extracted from journal entries over a rolling 7-day window.

The core idea is that more-recent days should carry more influence when
forming a summary of the user's emotional state, so every aggregation
method applies a configurable per-day weight vector (oldest → newest).
"""

from __future__ import annotations

import json
from typing import Any


class TemporalAggregator:
    """Aggregate emotion and VAD data across multiple days with recency weighting.

    Each position in the weight vector corresponds to one day in a chronological
    window, index 0 being the *oldest* day and index -1 being *today*.  The
    default profile ramps from 0.4 (one week ago) to 1.0 (today), giving the
    most-recent day roughly 2.5× the influence of the oldest day.

    Parameters
    ----------
    weights : list[float] | None
        A list of exactly 7 non-negative floats, ordered oldest-first.
        When *None* the default ramp ``[0.4, 0.5, 0.6, 0.7, 0.85, 0.95, 1.0]``
        is used.

    Raises
    ------
    ValueError
        If *weights* is supplied but does not contain exactly 7 elements, or if
        any element is negative.

    Examples
    --------
    >>> agg = TemporalAggregator()
    >>> agg.weights
    [0.4, 0.5, 0.6, 0.7, 0.85, 0.95, 1.0]
    """

    DEFAULT_WEIGHTS: list[float] = [0.4, 0.5, 0.6, 0.7, 0.85, 0.95, 1.0]

    def __init__(self, weights: list[float] | None = None) -> None:
        if weights is None:
            self.weights: list[float] = list(self.DEFAULT_WEIGHTS)
        else:
            if len(weights) != 7:
                raise ValueError(
                    f"weights must contain exactly 7 elements, got {len(weights)}."
                )
            if any(w < 0 for w in weights):
                raise ValueError("All weights must be non-negative.")
            self.weights = list(weights)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_weights(self, n: int) -> list[float]:
        """Return the last *n* weights from the full weight vector.

        When fewer than 7 days of data are available we simply slice the tail
        of the weight vector so that the most-recent entry always maps to
        ``weights[-1]``.

        Parameters
        ----------
        n : int
            Number of data points (days) actually available.  Must satisfy
            ``1 <= n <= 7``.

        Returns
        -------
        list[float]
            A sub-list of length *n* taken from the end of ``self.weights``.
        """
        if n <= 0:
            return []
        return self.weights[-n:]

    @staticmethod
    def _weighted_average_dicts(
        dicts: list[dict[str, float]],
        weights: list[float],
    ) -> dict[str, float]:
        """Compute a weighted average over a list of label→score dicts.

        The union of all keys found across *dicts* is used.  Missing keys in
        any individual dict are treated as 0.0.

        Parameters
        ----------
        dicts : list[dict[str, float]]
            Input dicts, one per time step, ordered oldest-first.
        weights : list[float]
            Per-step weights aligned with *dicts*; must have the same length.

        Returns
        -------
        dict[str, float]
            Weighted-average scores, rounded to 6 decimal places.
        """
        if not dicts:
            return {}

        all_keys: set[str] = set()
        for d in dicts:
            all_keys.update(d.keys())

        total_weight = sum(weights)
        if total_weight == 0:
            return {k: 0.0 for k in all_keys}

        result: dict[str, float] = {}
        for key in all_keys:
            weighted_sum = sum(
                d.get(key, 0.0) * w for d, w in zip(dicts, weights)
            )
            result[key] = round(weighted_sum / total_weight, 6)

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate_emotions(
        self,
        emotion_dicts: list[dict[str, float]],
    ) -> dict[str, float]:
        """Compute a recency-weighted average over a sequence of emotion dicts.

        Each element of *emotion_dicts* represents one day's emotion vector
        (e.g. ``{'joy': 0.8, 'sadness': 0.1, 'anger': 0.05, ...}``).  The
        list is assumed to be ordered oldest-first with the last element
        representing today.

        When fewer than 7 days are supplied the method automatically selects
        the *last N* weights from the configured weight vector so that the
        most-recent entry always receives the highest weight.

        Parameters
        ----------
        emotion_dicts : list[dict[str, float]]
            Between 1 and 7 daily emotion dicts, ordered oldest → newest.
            An empty list returns an empty dict.

        Returns
        -------
        dict[str, float]
            A single emotion dict whose values are the weighted averages of
            the corresponding scores across all input days.

        Examples
        --------
        >>> agg = TemporalAggregator()
        >>> days = [
        ...     {'joy': 0.6, 'sadness': 0.3},
        ...     {'joy': 0.8, 'sadness': 0.1},
        ... ]
        >>> agg.aggregate_emotions(days)
        {'joy': 0.737..., 'sadness': 0.168...}
        """
        if not emotion_dicts:
            return {}
        n = min(len(emotion_dicts), 7)
        data = emotion_dicts[-n:]  # guard against lists longer than 7
        w = self._effective_weights(n)
        return self._weighted_average_dicts(data, w)

    def aggregate_vad(
        self,
        vad_dicts: list[dict[str, float]],
    ) -> dict[str, float]:
        """Compute a recency-weighted average over a sequence of VAD dicts.

        VAD (Valence-Arousal-Dominance) is a three-dimensional model of
        affect.  Each input dict must contain the keys ``'valence'``,
        ``'arousal'``, and ``'dominance'``.  Any extra keys are preserved
        in the output using the same weighted-average logic.

        Parameters
        ----------
        vad_dicts : list[dict[str, float]]
            Between 1 and 7 daily VAD dicts, ordered oldest → newest.
            An empty list returns an empty dict.

        Returns
        -------
        dict[str, float]
            A single VAD dict with weighted-average values for each
            dimension.  Keys: ``'valence'``, ``'arousal'``, ``'dominance'``.

        Examples
        --------
        >>> agg = TemporalAggregator()
        >>> days = [
        ...     {'valence': 0.4, 'arousal': 0.6, 'dominance': 0.5},
        ...     {'valence': 0.7, 'arousal': 0.3, 'dominance': 0.6},
        ... ]
        >>> result = agg.aggregate_vad(days)
        >>> 0.4 < result['valence'] < 0.7
        True
        """
        if not vad_dicts:
            return {}
        n = min(len(vad_dicts), 7)
        data = vad_dicts[-n:]
        w = self._effective_weights(n)
        return self._weighted_average_dicts(data, w)

    def get_dominant_emotions(
        self,
        emotion_dict: dict[str, float],
        top_n: int = 3,
    ) -> list[str]:
        """Return the labels of the highest-scoring emotions.

        Parameters
        ----------
        emotion_dict : dict[str, float]
            A single (possibly already aggregated) emotion dict mapping
            label strings to float scores.
        top_n : int, optional
            How many labels to return.  Defaults to 3.  If *emotion_dict*
            contains fewer than *top_n* entries, all labels are returned.

        Returns
        -------
        list[str]
            Emotion labels sorted by descending score, truncated to *top_n*.

        Examples
        --------
        >>> agg = TemporalAggregator()
        >>> scores = {'joy': 0.7, 'sadness': 0.1, 'anger': 0.05, 'fear': 0.15}
        >>> agg.get_dominant_emotions(scores, top_n=2)
        ['joy', 'fear']
        """
        if not emotion_dict:
            return []
        sorted_labels = sorted(
            emotion_dict.keys(),
            key=lambda label: emotion_dict[label],
            reverse=True,
        )
        return sorted_labels[:top_n]

    def summarise(self, journal_entries: list[Any]) -> dict:
        """Produce a temporal summary from a list of JournalEntry-like objects.

        Each element of *journal_entries* must expose:

        * ``.emotion_vector`` – a JSON string encoding a ``dict[str, float]``
          (e.g. ``'{"joy": 0.8, "sadness": 0.1}'``).
        * ``.vad_valence`` – float in roughly ``[−1, 1]`` or ``[0, 1]``.
        * ``.vad_arousal`` – float.
        * ``.vad_dominance`` – float.

        The list is treated as ordered oldest-first.  Only the last 7 entries
        are considered; earlier entries are silently discarded.

        Parameters
        ----------
        journal_entries : list
            Sequence of objects with the attributes described above.

        Returns
        -------
        dict
            A summary dictionary with the following keys:

            ``'aggregated_emotions'`` : dict[str, float]
                Weighted-average emotion vector across all entries.
            ``'aggregated_vad'`` : dict[str, float]
                Weighted-average VAD dimensions across all entries.
            ``'dominant_emotions'`` : list[str]
                Top-3 emotion labels from the aggregated emotion vector.
            ``'entry_count'`` : int
                Number of entries that were actually used (≤ 7).

        Raises
        ------
        AttributeError
            If any entry is missing the required attributes.
        json.JSONDecodeError
            If any ``.emotion_vector`` value is not valid JSON.

        Examples
        --------
        >>> class Entry:
        ...     def __init__(self, ev, v, a, d):
        ...         self.emotion_vector = ev
        ...         self.vad_valence = v
        ...         self.vad_arousal = a
        ...         self.vad_dominance = d
        >>> entries = [Entry('{"joy": 0.9, "sadness": 0.1}', 0.8, 0.4, 0.6)]
        >>> agg = TemporalAggregator()
        >>> result = agg.summarise(entries)
        >>> result['dominant_emotions']
        ['joy', 'sadness']
        """
        # Use only the most-recent 7 entries
        entries = journal_entries[-7:] if len(journal_entries) > 7 else journal_entries
        entry_count = len(entries)

        emotion_dicts: list[dict[str, float]] = []
        vad_dicts: list[dict[str, float]] = []

        for entry in entries:
            # Parse emotion vector from JSON string
            ev: dict[str, float] = json.loads(entry.emotion_vector)
            emotion_dicts.append(ev)

            vad_dicts.append(
                {
                    "valence": float(entry.vad_valence),
                    "arousal": float(entry.vad_arousal),
                    "dominance": float(entry.vad_dominance),
                }
            )

        aggregated_emotions = self.aggregate_emotions(emotion_dicts)
        aggregated_vad = self.aggregate_vad(vad_dicts)
        dominant_emotions = self.get_dominant_emotions(aggregated_emotions, top_n=3)

        return {
            "aggregated_emotions": aggregated_emotions,
            "aggregated_vad": aggregated_vad,
            "dominant_emotions": dominant_emotions,
            "entry_count": entry_count,
        }


# ---------------------------------------------------------------------------
# Demonstration / smoke-test
# ---------------------------------------------------------------------------

class _FakeJournalEntry:
    """Minimal stand-in for a real ORM JournalEntry model."""

    def __init__(
        self,
        emotion_vector: dict[str, float],
        vad_valence: float,
        vad_arousal: float,
        vad_dominance: float,
    ) -> None:
        # Store as JSON string to match the real model's TextField behaviour
        self.emotion_vector: str = json.dumps(emotion_vector)
        self.vad_valence = vad_valence
        self.vad_arousal = vad_arousal
        self.vad_dominance = vad_dominance


if __name__ == "__main__":
    import pprint

    # ------------------------------------------------------------------
    # Build 7 days of fake data (oldest → newest)
    # Day 0 = one week ago; Day 6 = today
    # ------------------------------------------------------------------
    fake_entries = [
        _FakeJournalEntry(
            emotion_vector={"joy": 0.30, "sadness": 0.40, "anger": 0.15, "fear": 0.10, "surprise": 0.05},
            vad_valence=0.30, vad_arousal=0.55, vad_dominance=0.40,
        ),
        _FakeJournalEntry(
            emotion_vector={"joy": 0.35, "sadness": 0.35, "anger": 0.10, "fear": 0.15, "surprise": 0.05},
            vad_valence=0.35, vad_arousal=0.50, vad_dominance=0.42,
        ),
        _FakeJournalEntry(
            emotion_vector={"joy": 0.45, "sadness": 0.25, "anger": 0.10, "fear": 0.12, "surprise": 0.08},
            vad_valence=0.48, vad_arousal=0.48, vad_dominance=0.50,
        ),
        _FakeJournalEntry(
            emotion_vector={"joy": 0.55, "sadness": 0.20, "anger": 0.08, "fear": 0.10, "surprise": 0.07},
            vad_valence=0.55, vad_arousal=0.45, vad_dominance=0.55,
        ),
        _FakeJournalEntry(
            emotion_vector={"joy": 0.65, "sadness": 0.15, "anger": 0.05, "fear": 0.08, "surprise": 0.07},
            vad_valence=0.62, vad_arousal=0.42, vad_dominance=0.60,
        ),
        _FakeJournalEntry(
            emotion_vector={"joy": 0.72, "sadness": 0.10, "anger": 0.04, "fear": 0.07, "surprise": 0.07},
            vad_valence=0.70, vad_arousal=0.40, vad_dominance=0.65,
        ),
        _FakeJournalEntry(
            emotion_vector={"joy": 0.80, "sadness": 0.05, "anger": 0.03, "fear": 0.05, "surprise": 0.07},
            vad_valence=0.78, vad_arousal=0.38, vad_dominance=0.70,
        ),
    ]

    agg = TemporalAggregator()

    print("=" * 60)
    print("TemporalAggregator — 7-day demonstration")
    print("=" * 60)

    # 1. Full summarise()
    summary = agg.summarise(fake_entries)
    print("\n[summarise() — all 7 days]")
    pprint.pprint(summary)

    # 2. Only 3 days (shorter window)
    short_summary = agg.summarise(fake_entries[-3:])
    print("\n[summarise() — last 3 days only]")
    pprint.pprint(short_summary)

    # 3. Stand-alone aggregate_emotions()
    emotion_dicts = [json.loads(e.emotion_vector) for e in fake_entries]
    agg_emotions = agg.aggregate_emotions(emotion_dicts)
    print("\n[aggregate_emotions() — raw output]")
    pprint.pprint(agg_emotions)

    # 4. Stand-alone get_dominant_emotions()
    top5 = agg.get_dominant_emotions(agg_emotions, top_n=5)
    print(f"\n[get_dominant_emotions() top-5]: {top5}")

    # 5. Stand-alone aggregate_vad()
    vad_dicts = [
        {"valence": e.vad_valence, "arousal": e.vad_arousal, "dominance": e.vad_dominance}
        for e in fake_entries
    ]
    agg_vad = agg.aggregate_vad(vad_dicts)
    print("\n[aggregate_vad() — raw output]")
    pprint.pprint(agg_vad)

    print("\nDone.")