"""
scent_coverage.py — where does the Vasya library cover the emotional map, and where not?

Run from the project root:   python scent_coverage.py

For each of the 28 emotions it shows the scent the system would pick when matching the
emotion directly and when using the adaptive policy, and how far the closest scent is.
Writes experiments/scent_coverage.csv. Use the output as evidence for the report's
"scent library coverage" limitation, and to describe which VAD profiles the catalogue
would need to cover to serve negative states better. It does not add scents.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ml.nlp.emotion_classifier import EmotionClassifier           # noqa: E402
from ml.recommender.policy import apply_policy                    # noqa: E402
from ml.recommender.calibration import DEFAULT_AROUSAL_OFFSET, to_scent_space  # noqa: E402
from ml.recommender.recommender import RuleBasedRecommender       # noqa: E402

FAR = 0.35        # a scent further than this from the user's state is a poor match
KEYS = ('valence', 'arousal', 'dominance')


def main():
    offset = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AROUSAL_OFFSET
    print(f'Arousal offset: {offset} (pass a number as the first argument to change it; 0 = off)\n')
    rb = RuleBasedRecommender(str(ROOT / 'data' / 'scents.json'))
    protos = np.array([[s[k] for k in KEYS] for s in rb.scents])

    print('Library bounds (valence / arousal / dominance):')
    for i, k in enumerate(KEYS):
        print(f'  {k:<10} {protos[:, i].min():.2f} - {protos[:, i].max():.2f}')
    print(f'  scents with valence >= 0.70 and arousal >= 0.70: '
          f'{int(((protos[:, 0] >= .7) & (protos[:, 1] >= .7)).sum())}')
    print(f'  scents with valence < 0.55: {int((protos[:, 0] < .55).sum())}')
    print(f'  scents with dominance < 0.50: {int((protos[:, 2] < .5).sum())}\n')

    rows = []
    for label, (v, a, d) in EmotionClassifier.VAD_TABLE.items():
        state = to_scent_space(dict(zip(KEYS, (v, a, d))), arousal_offset=offset)
        match = rb.recommend(state)
        adapt = rb.recommend(apply_policy(state, 'adaptive'))
        rows.append({
            'emotion': label,
            'valence': round(state['valence'], 2), 'arousal': round(state['arousal'], 2),
            'dominance': round(state['dominance'], 2),
            'match_scent': match['name'], 'match_distance': round(match['distance'], 2),
            'adaptive_scent': adapt['name'],
        })

    print(f"{'emotion':<15}{'V':>5}{'A':>5}{'D':>5}  {'match -> scent (distance)':<44}adaptive -> scent")
    for r in sorted(rows, key=lambda r: r['valence']):
        flag = ' *' if r['match_distance'] > FAR else '  '
        print(f"{r['emotion']:<15}{r['valence']:>5}{r['arousal']:>5}{r['dominance']:>5}  "
              f"{r['match_scent'][:28]:<28} ({r['match_distance']:.2f}){flag}   {r['adaptive_scent']}")

    dist = np.array([r['match_distance'] for r in rows])
    print(f'\n* = nearest scent is further than {FAR} from the emotion')
    print(f'Mean distance to nearest scent: {dist.mean():.2f} | worst: {dist.max():.2f} | '
          f'emotions beyond {FAR}: {int((dist > FAR).sum())} of {len(rows)}')
    print(f"Distinct scents reached: match {len({r['match_scent'] for r in rows})} | "
          f"adaptive {len({r['adaptive_scent'] for r in rows})} of {len(rb.scents)}")

    out = ROOT / 'experiments'
    out.mkdir(exist_ok=True)
    with open(out / 'scent_coverage.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\nSaved {out / "scent_coverage.csv"}')


if __name__ == '__main__':
    main()
