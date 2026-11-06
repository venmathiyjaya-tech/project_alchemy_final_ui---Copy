"""
EmotionClassifier: Multi-label emotion detection using GoEmotions taxonomy.
Maps text → 27 emotion probabilities → VAD (Valence-Arousal-Dominance) coordinates.
"""

import json
import logging
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

logger = logging.getLogger(__name__)


class EmotionClassifier:
    """
    Multi-label emotion classifier using the GoEmotions 27-label taxonomy.

    Loads a RoBERTa-based sequence classification model and provides:
      - predict(text)      → per-emotion probability scores
      - predict_vad(dict)  → Valence-Arousal-Dominance affective coordinates
    """

    # Canonical GoEmotions order = the SamLowe / Hugging Face head order
    # ('neutral' is LAST, index 27). New checkpoints ship a labels.json that is
    # the source of truth; this list is only the expected set / default.
    LABELS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness',
        'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise', 'neutral',
    ]

    # Order used by checkpoints trained with the OLD notebook (alphabetical,
    # 'neutral' at index 20). Such a checkpoint has no labels.json and its
    # config.json id2label is stale, so we must NOT trust the config for it.
    LEGACY_LABELS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'neutral',
        'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise',
    ]

    # Research-grounded VAD values for all 27 GoEmotions labels.
    # Based on: Warriner et al. (2013) affective norms + Cowen & Keltner (2017)
    # GoEmotions mapping. Scale: Valence [-1, 1], Arousal [-1, 1], Dominance [-1, 1]
    VAD_TABLE = {
        #                     V       A       D
        'admiration':      ( 0.82,  0.41,   0.36),
        'amusement':       ( 0.76,  0.57,   0.46),
        'anger':           (-0.64,  0.73,   0.51),
        'annoyance':       (-0.48,  0.45,   0.29),
        'approval':        ( 0.69,  0.27,   0.55),
        'caring':          ( 0.80,  0.28,   0.44),
        'confusion':       (-0.22,  0.35,  -0.26),
        'curiosity':       ( 0.42,  0.52,   0.18),
        'desire':          ( 0.58,  0.64,   0.21),
        'disappointment':  (-0.61,  0.23,  -0.38),
        'disapproval':     (-0.52,  0.38,   0.32),
        'disgust':         (-0.72,  0.52,   0.39),
        'embarrassment':   (-0.44,  0.49,  -0.49),
        'excitement':      ( 0.80,  0.84,   0.54),
        'fear':            (-0.70,  0.72,  -0.57),
        'gratitude':       ( 0.87,  0.35,   0.29),
        'grief':           (-0.82,  0.28,  -0.48),
        'joy':             ( 0.88,  0.72,   0.57),
        'love':            ( 0.90,  0.55,   0.40),
        'nervousness':     (-0.38,  0.68,  -0.44),
        'neutral':         ( 0.00,  0.00,   0.00),
        'optimism':        ( 0.74,  0.45,   0.52),
        'pride':           ( 0.77,  0.55,   0.72),
        'realization':     ( 0.18,  0.30,   0.16),
        'relief':          ( 0.72, -0.20,   0.38),
        'remorse':         (-0.68,  0.25,  -0.36),
        'sadness':         (-0.76,  0.21,  -0.42),
        'surprise':        ( 0.22,  0.68,  -0.06),
    }

    # ── Emotion grouping: 27 GoEmotions → 8 parent categories ──────────
    # The model still classifies all 27 labels internally.
    # group_emotions() collapses them for display and dominant-emotion
    # reporting while preserving full VAD fidelity in the pipeline.
    EMOTION_GROUPS: dict[str, list[str]] = {
        'joy':       ['joy', 'amusement', 'excitement', 'gratitude',
                      'relief', 'optimism', 'pride', 'admiration'],
        'love':      ['love', 'caring', 'desire'],
        'sadness':   ['sadness', 'grief', 'disappointment', 'remorse'],
        'anger':     ['anger', 'annoyance', 'disapproval', 'disgust'],
        'fear':      ['fear', 'nervousness'],
        'surprise':  ['surprise', 'realization', 'confusion'],
        'neutral':   ['neutral', 'approval'],
        'curiosity': ['curiosity', 'embarrassment'],
    }

    # ── Input validation thresholds ───────────────────────────────────
    MIN_WORD_COUNT   = 3      # must have at least 3 words
    MIN_ALPHA_RATIO  = 0.50   # at least 50% of characters must be letters
    MIN_UNIQUE_WORDS = 2      # at least 2 distinct alphabetic words

    class InvalidEntryError(ValueError):
        """Raised when submitted text is unsuitable for emotion analysis."""
        pass

    def __init__(self, model_path: str = 'models/roberta_finetuned'):
        """
        Initialise tokenizer and model from `model_path`.

        Falls back to 'roberta-base' if the requested path cannot be loaded
        (e.g. a fine-tuned checkpoint that does not exist yet).

        Args:
            model_path: Local directory path or HuggingFace Hub model ID.
        """
        self.model_path = model_path

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            print(f"[EmotionClassifier] Loaded model from '{model_path}'.")
        except (OSError, EnvironmentError) as exc:
            fallback = 'roberta-base'
            print(
                f"[EmotionClassifier] Could not load '{model_path}' ({exc}). "
                f"Falling back to '{fallback}'."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(fallback)
            self.model = AutoModelForSequenceClassification.from_pretrained(fallback)
            self.model_path = fallback

        self.model.eval()
        self.labels = self._resolve_labels()

    # ------------------------------------------------------------------
    # Label order
    # ------------------------------------------------------------------

    def _resolve_labels(self) -> list[str]:
        """
        Return the label order matching the classifier head.

        1. <model_path>/labels.json  -> authoritative (written by the v4 notebook).
        2. Local folder without it   -> legacy order (old checkpoints; config is stale).
        3. Hub id / other            -> config.id2label if it holds the GoEmotions
                                        names, otherwise the legacy order.
        """
        expected = set(self.LABELS)
        labels_file = (os.path.join(self.model_path, 'labels.json')
                       if os.path.isdir(self.model_path) else None)

        if labels_file and os.path.isfile(labels_file):
            with open(labels_file, encoding='utf-8') as f:
                labels = list(json.load(f)['labels'])
            if set(labels) != expected or len(labels) != len(expected):
                raise ValueError(f'{labels_file} does not list the 28 GoEmotions labels.')
            if self.model.config.num_labels != len(labels):
                raise ValueError(
                    f'labels.json has {len(labels)} labels but the model head has '
                    f'{self.model.config.num_labels} outputs.')
            return labels

        if os.path.isdir(self.model_path):
            logger.warning(
                "No labels.json in '%s' - assuming the LEGACY label order. "
                "Re-run the v4 fine-tuning notebook to remove this warning.",
                self.model_path)
            return list(self.LEGACY_LABELS)

        id2label = getattr(self.model.config, 'id2label', {}) or {}
        cfg = [id2label.get(i) for i in range(len(id2label))]
        if len(cfg) == len(expected) and set(cfg) == expected:
            return cfg
        logger.warning("Model '%s' has no usable id2label - assuming LEGACY order.",
                       self.model_path)
        return list(self.LEGACY_LABELS)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> dict[str, float]:
        """
        Run multi-label emotion classification on `text`.

        Tokenizes the input (max 512 tokens), runs a forward pass under
        torch.no_grad(), and applies sigmoid to convert logits to independent
        per-label probabilities.

        Args:
            text: Raw input string.

        Returns:
            Dict mapping each of the 27 GoEmotions labels to its probability
            in [0, 1].
        """
        self._validate_text(text.strip())

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits                     # (1, num_labels)
            probs = torch.sigmoid(logits).squeeze(0)    # (num_labels,)

        num_model_labels = probs.shape[0]
        num_expected = len(self.labels)

        # If the loaded model has fewer outputs than our label list (e.g. when
        # falling back to base roberta with only 1 or 2 outputs for demo
        # purposes), pad with zeros so callers always receive all 27 keys.
        if num_model_labels < num_expected:
            padding = torch.zeros(num_expected - num_model_labels)
            probs = torch.cat([probs, padding])
        elif num_model_labels > num_expected:
            probs = probs[:num_expected]

        return {label: float(probs[i]) for i, label in enumerate(self.labels)}

    # ------------------------------------------------------------------
    # VAD projection
    # ------------------------------------------------------------------

    def predict_vad(self, emotion_dict: dict[str, float]) -> dict[str, float]:
        """
        Map a probability-weighted emotion distribution to VAD coordinates.

        Each dimension is computed as the probability-weighted average of the
        per-emotion VAD values in VAD_TABLE, then normalised by the total
        probability mass so the result is always in [-1, 1].

        Args:
            emotion_dict: Output of predict() — {label: probability}.

        Returns:
            {'valence': float, 'arousal': float, 'dominance': float}
        """
        total_weight = 0.0
        v_acc = a_acc = d_acc = 0.0

        for label, prob in emotion_dict.items():
            if label not in self.VAD_TABLE:
                continue
            v, a, d = self.VAD_TABLE[label]
            v_acc += prob * v
            a_acc += prob * a
            d_acc += prob * d
            total_weight += prob

        if total_weight == 0.0:
            return {'valence': 0.0, 'arousal': 0.0, 'dominance': 0.0}

        return {
            'valence':   round(v_acc / total_weight, 4),
            'arousal':   round(a_acc / total_weight, 4),
            'dominance': round(d_acc / total_weight, 4),
        }


    def _validate_text(self, text: str) -> None:
        """
        Check that text is meaningful enough for emotion classification.

        Uses regex tokenisation so punctuation-heavy entries like
        "Yay!!! I'm happy!!!!" are correctly parsed as 3 words, not 0.

        Raises:
            InvalidEntryError: With a human-readable explanation.
        """
        import re as _re
        if not text or not text.strip():
            raise EmotionClassifier.InvalidEntryError(
                "Entry is empty. Please write something meaningful."
            )

        # Tokenise with regex so "Yay!!!" → "Yay", "I'm" → "I'm"
        tokens = _re.findall(r"[a-zA-Z\']+", text)
        if len(tokens) < self.MIN_WORD_COUNT:
            raise EmotionClassifier.InvalidEntryError(
                f"Entry is too short — please write at least "
                f"{self.MIN_WORD_COUNT} words. (Got {len(tokens)}.)"
            )

        # Alpha ratio check (guards against "0000", "123 456 789")
        alpha_chars = sum(c.isalpha() for c in text)
        total_chars = len(text.replace(' ', ''))
        if total_chars > 0 and (alpha_chars / total_chars) < self.MIN_ALPHA_RATIO:
            raise EmotionClassifier.InvalidEntryError(
                "Entry appears to contain mostly numbers or symbols. "
                "Please write in natural language so the archive can read you."
            )

        # Unique word check — strip leading/trailing apostrophes before comparing
        unique_words = {_re.sub(r"^\'+ |\'+ $", "", t).lower()
                        for t in tokens if len(t) > 1}
        if len(unique_words) < self.MIN_UNIQUE_WORDS:
            raise EmotionClassifier.InvalidEntryError(
                "Entry does not contain enough distinct words. "
                "Please write at least one full sentence."
            )

    def group_emotions(self, emotion_dict: dict[str, float]) -> dict[str, float]:
        """
        Collapse 27 GoEmotions labels into 8 parent categories.

        Each group score is the MAX of its member scores (not average),
        so a strong sub-emotion is not diluted by weaker siblings.
        The full 27-label dict is still used internally for VAD projection;
        this method is only used for display and dominant-emotion reporting.

        Args:
            emotion_dict: Output of predict() — {label: probability}.

        Returns:
            Dict mapping 8 group names to their max member score.
        """
        grouped: dict[str, float] = {}
        for group, members in self.EMOTION_GROUPS.items():
            scores = [emotion_dict.get(m, 0.0) for m in members]
            grouped[group] = round(max(scores), 4)
        return grouped

# ----------------------------------------------------------------------
# Quick smoke-test
# ----------------------------------------------------------------------

if __name__ == '__main__':
    TEST_SENTENCE = 'I felt overwhelmed today but my friends cheered me up.'

    print('=' * 60)
    print('EmotionClassifier — smoke test')
    print('=' * 60)
    print(f'Input: "{TEST_SENTENCE}"\n')

    classifier = EmotionClassifier()          # uses default 'roberta-base'

    emotion_scores = classifier.predict(TEST_SENTENCE)

    # Show top-5 emotions by probability
    top5 = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print('Top-5 emotion probabilities:')
    for label, score in top5:
        bar = '█' * int(score * 30)
        print(f'  {label:<18} {score:.4f}  {bar}')

    vad = classifier.predict_vad(emotion_scores)
    print('\nVAD coordinates:')
    for dim, val in vad.items():
        polarity = '+' if val >= 0 else ''
        print(f'  {dim:<12} {polarity}{val:.4f}')

    print('\n✓ Smoke test complete.')