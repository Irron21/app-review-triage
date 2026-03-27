import joblib
import numpy as np

class MLService:
    def __init__(self, model_path, vec_path):
        self.is_trained = False
        self.ml_model = None
        self.vectorizer = None
        self.load_models(model_path, vec_path)

    def load_models(self, model_path, vec_path):
        try:
            self.ml_model = joblib.load(model_path)
            self.vectorizer = joblib.load(vec_path)
            self.is_trained = True
            print("ML Models loaded successfully via MLService.")
        except Exception as e:
            self.is_trained = False
            print(f"Error loading ML models: {e}")

    def analyze_payload(self, cleaned_text, raw_text, app_name):
        """Processes payload via SKLearn LogisticRegression matrices returning core dict parameters."""
        if not self.is_trained:
            raise ValueError("ML Models are not trained or missing from the disk context limits.")
            
        # Target processed variables with safe raw fallback options mapped automatically
        used_text = cleaned_text if cleaned_text else raw_text
        vec_text = self.vectorizer.transform([used_text])
        prediction = self.ml_model.predict(vec_text)[0]
        
        probs = self.ml_model.predict_proba(vec_text)[0]
        confidence = np.max(probs)
        
        return {
            'app_name': str(app_name),
            'raw_review': str(raw_text),
            'sentiment': str(prediction),
            'confidence': round(float(confidence) * 100, 2)
        }
