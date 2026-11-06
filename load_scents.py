#!/usr/bin/env python3
import json
import os

from app import create_app  
from app.models import db, Scent

def load_scents():
    # Path to data/scents.json relative to this script (project root)
    json_path = os.path.join(os.path.dirname(__file__), 'data', 'scents.json')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        scents_data = json.load(f)

    for item in scents_data:
        scent = Scent(
            id=item['id'],
            name=item['name'],
            description=item['description'],
            vad_valence=item['vad_valence'],
            vad_arousal=item['vad_arousal'],
            vad_dominance=item['vad_dominance'],
            brand=item['brand'],
            scientific_prove_reference=item['scientific_prove_reference']
        )
        scent.tags_list = item['emotion_tags']
        db.session.merge(scent)

    db.session.commit()
    print(f" Loaded {len(scents_data)} scents.")

if __name__ == '__main__':
    app = create_app()
    with app.app_context():
        load_scents()