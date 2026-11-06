import os
from dotenv import load_dotenv 

load_dotenv() #reads the .env file

class Config:
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-change-in-production')

    # Database
    SQLALCHEMY_DATABASE_URI = 'sqlite:///alchemy.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ML Model Paths
    NLP_MODEL_PATH = 'roberta-base'
    NLP_FINETUNED_PATH = 'models/roberta_finetuned'
    RECOMMENDER_PATH = 'models/saved_model.joblib'   # one canonical path (app, tests, training)

    # Temporal Aggregation
    JOURNAL_WINDOW_DAYS = 7
    # Weights for days 1-7 (day 7 = most recent = highest weight)
    RECENCY_WEIGHTS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Lexicon blending (NRC VAD). OFF by default: the ablation (lexicon_ablation.py)
    # showed no measurable benefit over the RoBERTa VAD alone. Set the environment
    # variable USE_LEXICON_BLEND=1 to switch it on.
    USE_LEXICON_BLEND = os.environ.get('USE_LEXICON_BLEND', '0') == '1'
    NRC_VAD_PATH = 'data/NRC-VAD-Lexicon.txt'      # relative paths resolve from the project root
    LEXICON_MODEL_WEIGHT = 0.65                    # share of the model signal in the blend

    # Recommendation policy: 'adaptive' nudges clearly negative states toward a calming or
    # uplifting scent; 'match' aims at the user's own state (see ml/recommender/policy.py).
    RECOMMENDATION_POLICY = os.environ.get('RECOMMENDATION_POLICY', 'adaptive')
    POLICY_STRENGTH = float(os.environ.get('POLICY_STRENGTH', '1.0'))

    # Arousal calibration (ml/recommender/calibration.py): subtracted from the user's arousal
    # after converting to the scent library's 0-1 range. 0 switches it off.
    AROUSAL_OFFSET = float(os.environ.get('AROUSAL_OFFSET', '0.25'))

    # Scent Library
    SCENTS_DATA_PATH = 'data/scents.json'
