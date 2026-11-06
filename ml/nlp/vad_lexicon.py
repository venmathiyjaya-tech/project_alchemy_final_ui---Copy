"""
vad_lexicon.py
NRC VAD Lexicon-based VAD estimator.

Provides a second, lexicon-grounded VAD signal to complement the RoBERTa
emotion classifier. Works by looking up every word in the entry against the
~20,000-word NRC VAD Lexicon (Mohammad, 2018) and averaging the matched scores.

Reference:
    Mohammad, S. M. (2018). Obtaining Reliable Human Ratings of Valence,
    Arousal, and Dominance for 20,000 English Words. In Proceedings of ACL.
    https://saifmohammad.com/WebPages/nrc-vad.html

Usage:
    lexicon = VADLexicon('data/NRC-VAD-Lexicon.txt')
    vad = lexicon.score("I felt completely overwhelmed today but somehow hopeful")
    # → {'valence': 0.52, 'arousal': 0.61, 'dominance': 0.44,
    #     'coverage': 0.38, 'word_count': 7, 'matched_count': 4}
"""

import re
import os
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


class VADLexicon:
    """
    NRC VAD Lexicon scorer.

    Attributes:
        lexicon (dict): Maps lowercase word → (valence, arousal, dominance).
        FALLBACK_VAD (dict): Returned when no words match the lexicon.
    """

    # Returned when coverage is too low to be meaningful.
    # Represents a neutral emotional state — matches VAD_TABLE['neutral'].
    FALLBACK_VAD = {'valence': 0.5, 'arousal': 0.5, 'dominance': 0.5}

    # Minimum fraction of words that must match before we trust the result.
    # Below this threshold we blend more heavily toward the model signal.
    MIN_COVERAGE = 0.10   # at least 10% of words must be in the lexicon

    def __init__(self, lexicon_path: str):
        """
        Load the NRC VAD Lexicon from a tab-separated file.

        Expected format (one word per line, no header):
            word<TAB>valence<TAB>arousal<TAB>dominance

        Args:
            lexicon_path: Absolute or relative path to NRC-VAD-Lexicon.txt.

        Raises:
            FileNotFoundError: If the lexicon file does not exist.
            ValueError: If the file cannot be parsed.
        """
        if not os.path.isfile(lexicon_path):
            raise FileNotFoundError(
                f"NRC VAD Lexicon not found at: {lexicon_path}\n"
                f"Download it from: https://saifmohammad.com/WebPages/nrc-vad.html\n"
                f"and place NRC-VAD-Lexicon.txt in your data/ folder."
            )

        self.lexicon: dict[str, tuple[float, float, float]] = {}
        self._load(lexicon_path)
        logger.info("[VADLexicon] Loaded %d entries from %s", len(self.lexicon), lexicon_path)

    def _load(self, path: str) -> None:
        """Parse the lexicon file into self.lexicon."""
        with open(path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue  # skip blank lines and comments

                parts = line.split('\t')
                if len(parts) < 4:
                    # Some distributions use spaces — try splitting on whitespace
                    parts = line.split()

                if len(parts) < 4:
                    logger.debug("[VADLexicon] Skipping malformed line %d: %r", line_no, line)
                    continue

                word = parts[0].lower().strip()
                try:
                    valence   = float(parts[1])
                    arousal   = float(parts[2])
                    dominance = float(parts[3])
                except ValueError:
                    logger.debug("[VADLexicon] Non-numeric scores on line %d: %r", line_no, line)
                    continue

                self.lexicon[word] = (valence, arousal, dominance)

        self._normalise_scale(path)

    def _normalise_scale(self, path: str) -> None:
        """
        Make sure every score is on [0, 1], which is what score()/blend_vad()
        assume. NRC releases differ, so detect the range instead of trusting it:
          * [0, 1]   -> used as is
          * [-1, 1]  -> rescaled with (x + 1) / 2
          * anything else (e.g. a 1-9 scale) -> refuse to load
        """
        if not self.lexicon:
            raise ValueError(f"No entries could be parsed from {path}")
        lo = min(min(t) for t in self.lexicon.values())
        hi = max(max(t) for t in self.lexicon.values())
        if lo < -0.001:
            if lo < -1.001 or hi > 1.001:
                raise ValueError(f"Unexpected score range [{lo}, {hi}] in {path}")
            self.lexicon = {w: tuple((x + 1.0) / 2.0 for x in t)
                            for w, t in self.lexicon.items()}
            logger.info("[VADLexicon] Scores were on [-1, 1]; rescaled to [0, 1].")
        elif hi > 1.001:
            raise ValueError(
                f"Unexpected score range [{lo}, {hi}] in {path} "
                f"(expected [0, 1] or [-1, 1])."
            )

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """
        Lowercase and tokenise text into alphabetic words.
        Strips punctuation, numbers, and whitespace artefacts.
        """
        return re.findall(r"[a-z']+", text.lower())

    def score(self, text: str) -> dict:
        """
        Compute lexicon-based VAD scores for a piece of text.

        Each word is looked up in the lexicon. Words not found are skipped.
        The final scores are the arithmetic mean of all matched words.

        Args:
            text: Raw journal entry string.

        Returns:
            dict with keys:
                valence (float)     : 0–1, higher = more positive
                arousal (float)     : 0–1, higher = more activated/energetic
                dominance (float)   : 0–1, higher = more in-control
                coverage (float)    : fraction of words that matched the lexicon
                word_count (int)    : total words in the text
                matched_count (int) : words found in the lexicon
        """
        tokens = self._tokenise(text)
        if not tokens:
            return {**self.FALLBACK_VAD, 'coverage': 0.0,
                    'word_count': 0, 'matched_count': 0}

        v_sum = a_sum = d_sum = 0.0
        matched = 0

        for token in tokens:
            entry = self.lexicon.get(token)
            if entry is None:
                # Try stemmed form: strip trailing 's', 'ing', 'ed', 'ly'
                for suffix in ('ing', 'ed', 'ly', 'er', 'est', 's'):
                    if token.endswith(suffix) and len(token) > len(suffix) + 2:
                        stem = token[:-len(suffix)]
                        entry = self.lexicon.get(stem)
                        if entry:
                            break

            if entry:
                v, a, d  = entry
                v_sum   += v
                a_sum   += a
                d_sum   += d
                matched += 1

        word_count = len(tokens)
        coverage   = matched / word_count if word_count > 0 else 0.0

        if matched == 0:
            return {**self.FALLBACK_VAD, 'coverage': 0.0,
                    'word_count': word_count, 'matched_count': 0}

        return {
            'valence':       round(v_sum / matched, 4),
            'arousal':       round(a_sum / matched, 4),
            'dominance':     round(d_sum / matched, 4),
            'coverage':      round(coverage, 4),
            'word_count':    word_count,
            'matched_count': matched,
        }

    def is_reliable(self, score_result: dict) -> bool:
        """Return True if the lexicon coverage is above MIN_COVERAGE."""
        return score_result.get('coverage', 0.0) >= self.MIN_COVERAGE


# ── Blending utility ──────────────────────────────────────────────────────

def blend_vad(
    model_vad:   dict[str, float],
    lexicon_vad: dict,
    model_weight: float = 0.65,
) -> dict[str, float]:
    """
    Blend model VAD and lexicon VAD into a single estimate.

    The NRC lexicon scores are in [0, 1] while the model's VAD_TABLE
    is in [-1, 1].  We normalise the lexicon output to [-1, 1] before
    blending so both signals are on the same scale.

    Args:
        model_vad:    {'valence': float, 'arousal': float, 'dominance': float}
                      from EmotionClassifier.predict_vad()   — scale [-1, 1]
        lexicon_vad:  output of VADLexicon.score()            — scale [0, 1]
        model_weight: weight given to the model signal (0–1).
                      lexicon gets (1 - model_weight).
                      Default 0.65 means model drives 65% of the result.

    Returns:
        Blended VAD dict on scale [-1, 1], matching the model's output format.
    """
    lw = 1.0 - model_weight   # lexicon weight

    # Normalise lexicon [0,1] → [-1,1]
    lv = lexicon_vad.get('valence',   0.5)
    la = lexicon_vad.get('arousal',   0.5)
    ld = lexicon_vad.get('dominance', 0.5)
    lv_norm = (lv - 0.5) * 2
    la_norm = (la - 0.5) * 2
    ld_norm = (ld - 0.5) * 2

    # If lexicon coverage is very low, trust the model almost entirely
    coverage = lexicon_vad.get('coverage', 0.0)
    if coverage < VADLexicon.MIN_COVERAGE:
        effective_lw = lw * (coverage / VADLexicon.MIN_COVERAGE)
        effective_mw = 1.0 - effective_lw
    else:
        effective_mw = model_weight
        effective_lw = lw

    blended = {
        'valence':   round(effective_mw * model_vad['valence']   + effective_lw * lv_norm, 4),
        'arousal':   round(effective_mw * model_vad['arousal']   + effective_lw * la_norm, 4),
        'dominance': round(effective_mw * model_vad['dominance'] + effective_lw * ld_norm, 4),
    }

    return blended


# ── Smoke test ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    lexicon_path = sys.argv[1] if len(sys.argv) > 1 else 'data/NRC-VAD-Lexicon.txt'

    print("=" * 60)
    print("VADLexicon — smoke test")
    print("=" * 60)

    try:
        lex = VADLexicon(lexicon_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    tests = [
        "0000",
        "I felt completely overwhelmed today but somehow hopeful",
        "Everything is wonderful and I am so grateful",
        "I am furious and cannot stop crying",
        "Just another day. Nothing happened.",
    ]

    for text in tests:
        result = lex.score(text)
        print(f"\nText:     {text!r}")
        print(f"Valence:  {result['valence']:.3f}  "
              f"Arousal: {result['arousal']:.3f}  "
              f"Dominance: {result['dominance']:.3f}")
        print(f"Coverage: {result['matched_count']}/{result['word_count']} words "
              f"({result['coverage']:.0%})")
        print(f"Reliable: {lex.is_reliable(result)}")
