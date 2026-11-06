"""
lexicon_ablation.py — does blending the NRC VAD lexicon actually help? (issue 3)

Run from the project root (needs the fine-tuned model, the journal CSV and the
NRC lexicon file):

    python lexicon_ablation.py --csv "journal entries.csv" \
        --model models/roberta_finetuned --lexicon data/NRC-VAD-Lexicon.txt

It scores the SAME 221 validation entries the notebook held out (same split,
random_state=42) with: model VAD only, lexicon VAD only, and several blends.

HONEST LIMITATION: nobody has rated the VAD of these journal entries. The only
reference available is a PROXY: the average VAD of the emotions each writer
ticked (mapped to GoEmotions labels, valued with EmotionClassifier.VAD_TABLE).
That proxy shares its VAD_TABLE with the model signal, so it slightly favours
the model. Read the result as "do the two signals agree with the writers' own
labels", not as proof of accuracy.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.nlp.emotion_classifier import EmotionClassifier
from ml.nlp.vad_lexicon import VADLexicon, blend_vad

# Same mapping as the fine-tuning notebook.
CSV_TO_GO = {
    'afraid':     ['fear', 'nervousness'],
    'angry':      ['anger', 'annoyance'],
    'anxious':    ['nervousness', 'fear'],
    'ashamed':    ['remorse', 'embarrassment'],
    'awkward':    ['embarrassment', 'confusion'],
    'bored':      ['annoyance'],
    'calm':       ['relief', 'neutral'],
    'confused':   ['confusion'],
    'disgusted':  ['disgust'],
    'excited':    ['excitement', 'joy'],
    'frustrated': ['anger', 'annoyance', 'disapproval'],
    'happy':      ['joy', 'amusement', 'admiration'],
    'jealous':    ['annoyance', 'disapproval'],
    'nostalgic':  ['realization', 'sadness'],
    'proud':      ['pride', 'admiration'],
    'sad':        ['sadness', 'grief'],
    'satisfied':  ['approval', 'gratitude', 'relief'],
    'surprised':  ['surprise'],
}
MODEL_WEIGHTS = [1.0, 0.8, 0.65, 0.5, 0.0]   # 1.0 = model only, 0.0 = lexicon only
DIMS = ['valence', 'arousal', 'dominance']


def validation_indices(n_rows: int) -> np.ndarray:
    """Row positions of the notebook's validation split (test_size=0.15, seed 42)."""
    _, val_idx = train_test_split(np.arange(n_rows), test_size=0.15, random_state=42)
    return val_idx


def proxy_vad(row: pd.Series) -> np.ndarray:
    """Average VAD_TABLE value of the GoEmotions labels implied by the ticked emotions."""
    active = set()
    for csv_emotion, go_labels in CSV_TO_GO.items():
        if str(row.get(f'Answer.f1.{csv_emotion}.raw', '')).strip().upper() == 'TRUE':
            active.update(go_labels)
    if not active:
        active = {'neutral'}
    return np.mean([EmotionClassifier.VAD_TABLE[l] for l in active], axis=0)


def _corr(a, b):
    return float('nan') if np.std(a) < 1e-9 or np.std(b) < 1e-9 else float(np.corrcoef(a, b)[0, 1])


def evaluate(model_vad: np.ndarray, lex_scores: list, proxy: np.ndarray) -> pd.DataFrame:
    """One row per blend weight: MAE / Pearson r vs the proxy, plus valence-sign accuracy."""
    rows = []
    for w in MODEL_WEIGHTS:
        blended = np.array([
            [b[d] for d in DIMS]
            for b in (blend_vad(dict(zip(DIMS, m)), lx, model_weight=w)
                      for m, lx in zip(model_vad, lex_scores))
        ])
        err = np.abs(blended - proxy)
        rows.append({
            'model_weight': w,
            'MAE_V': err[:, 0].mean(), 'MAE_A': err[:, 1].mean(), 'MAE_D': err[:, 2].mean(),
            'MAE_mean': err.mean(),
            'r_V': _corr(blended[:, 0], proxy[:, 0]),
            'r_A': _corr(blended[:, 1], proxy[:, 1]),
            'valence_sign_acc': float((np.sign(blended[:, 0]) == np.sign(proxy[:, 0])).mean()),
        })
    return pd.DataFrame(rows).round(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--model', default='models/roberta_finetuned')
    ap.add_argument('--lexicon', default='data/NRC-VAD-Lexicon.txt')
    ap.add_argument('--out', default='lexicon_ablation_results.csv')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    val = df.iloc[validation_indices(len(df))].reset_index(drop=True)
    print(f'Validation entries: {len(val)}')

    clf, lex = EmotionClassifier(args.model), VADLexicon(args.lexicon)
    texts = val['Answer'].astype(str).tolist()
    model_vad = np.array([[clf.predict_vad(clf.predict(t))[d] for d in DIMS] for t in texts])
    lex_scores = [lex.score(t) for t in texts]
    proxy = np.array([proxy_vad(r) for _, r in val.iterrows()])

    cov = np.array([s['coverage'] for s in lex_scores])
    print(f'Lexicon coverage: mean {cov.mean():.2f}, '
          f'{(cov >= VADLexicon.MIN_COVERAGE).mean() * 100:.0f}% of entries above the '
          f'{VADLexicon.MIN_COVERAGE:.0%} reliability threshold')
    lex_v = np.array([(s['valence'] - 0.5) * 2 for s in lex_scores])
    print(f'Model vs lexicon valence agreement (Pearson r): {_corr(model_vad[:, 0], lex_v):.3f}\n')

    table = evaluate(model_vad, lex_scores, proxy)
    print(table.to_string(index=False))
    table.to_csv(args.out, index=False)
    print(f'\nSaved {args.out}')
    print('Lower MAE and higher r are better. If the blends are no better than model_weight=1.0,')
    print('the lexicon adds nothing measurable: set USE_LEXICON_BLEND = False and drop it from the report.')


if __name__ == '__main__':
    main()
