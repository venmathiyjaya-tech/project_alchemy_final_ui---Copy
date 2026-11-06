"""
check_model.py — run from the project root after swapping in the v5 model:

    python check_model.py                      # checks models/roberta_finetuned
    python check_model.py models/roberta_finetuned_v5

Confirms the checkpoint loads, that labels.json is being used, and that a few
sentences give sensible emotions.
"""
import os
import sys

import transformers

from ml.nlp.emotion_classifier import EmotionClassifier

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'models/roberta_finetuned'

SENTENCES = [
    'I felt overwhelmed today but somehow hopeful.',
    'Everything went wrong and I just want to cry.',
    'My friends threw me a surprise party and I am so grateful.',
    'Just another ordinary day, nothing special happened.',
]

print('transformers  ', transformers.__version__)
print('model folder  ', os.path.abspath(MODEL))
has_labels = os.path.isfile(os.path.join(MODEL, 'labels.json'))
print('labels.json   ', 'FOUND' if has_labels else 'MISSING -> legacy label order will be assumed')

clf = EmotionClassifier(MODEL)
print('neutral index ', clf.labels.index('neutral'),
      '(expect 27 for the v5 model, 20 for the old one)')

for text in SENTENCES:
    probs = clf.predict(text)
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:4]
    vad = clf.predict_vad(probs)
    print(f'\n"{text}"')
    print('  top emotions:', ', '.join(f'{k} {v:.2f}' for k, v in top))
    print('  VAD:', vad)
