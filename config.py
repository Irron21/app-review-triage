import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "super_secret_key")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY")
    MODEL_PATH = 'sentiment_model.joblib'
    VEC_PATH = 'vectorizer.joblib'
    METADATA_PATH = 'googleplaystore.csv'
