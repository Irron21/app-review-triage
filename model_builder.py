import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Ensure VADER lexicon is downloaded
def setup_nltk():
    nltk.download('vader_lexicon', quiet=True)

class HybridSentimentAnalyzer:
    def __init__(self):
        setup_nltk()
        self.vader = SentimentIntensityAnalyzer()
        # Adjusted vectorizer to capture 1 and 2-word phrases (ngrams) scaling context
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        # Replaced Naive Bayes with heavily iterated Logistic Regression for deeper convergence
        self.ml_model = LogisticRegression(max_iter=1000)
        self.is_trained = False
        
    def train_ml_model(self, X_train, y_train):
        """Train the underlying ML model using TF-IDF and Logistic Regression."""
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.ml_model.fit(X_train_vec, y_train)
        self.is_trained = True
        
    def get_vader_sentiment(self, text):
        """Returns sentiment string based on VADER compound scorings."""
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'Positive'
        elif compound <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
            
    def get_ml_sentiment(self, text):
        """Returns the predicted sentiment and the maximum probability from the LogisticRegression array."""
        if not self.is_trained:
            raise ValueError("ML model must be trained before prediction.")
            
        vec_text = self.vectorizer.transform([text])
        prediction = self.ml_model.predict(vec_text)[0]
        
        # Extract maximum probability
        probs = self.ml_model.predict_proba(vec_text)[0]
        max_prob = np.max(probs)
        
        return prediction, max_prob
        
    def manual_ensemble_predict(self, review_text):
        """
        Combines VADER and ML model predictions.
        If they agree, return that sentiment.
        If they disagree, use the ML model's confidence probability to break the tie.
        """
        if not isinstance(review_text, str) or not review_text.strip():
            return 'Neutral'
            
        vader_pred = self.get_vader_sentiment(review_text)
        ml_pred, ml_confidence = self.get_ml_sentiment(review_text)
        
        # Scenario 1: Agreement
        if vader_pred == ml_pred:
            return vader_pred
            
        # Scenario 2: Disagreement, Tie-Breaking Logic
        if ml_confidence > 0.65:
            return ml_pred
        else:
            return vader_pred

    def evaluate_model(self, X_test, y_test):
        """
        Tests the hybrid ensemble against a holdout testing set and calculates accuracy metrics.
        """
        print("Evaluating Hybrid Manual Ensemble Model (Logistic Regression + N-Grams)...")
        predictions = []
        for text in X_test:
            pred = self.manual_ensemble_predict(text)
            predictions.append(pred)
            
        acc = accuracy_score(y_test, predictions)
        print(f"Overall Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, zero_division=0))
        return acc

    def save_models(self, model_path='sentiment_model.joblib', vec_path='vectorizer.joblib'):
        """Saves the ML model and vectorizer to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save an untrained model.")
            
        joblib.dump(self.ml_model, model_path)
        joblib.dump(self.vectorizer, vec_path)
        print(f"Models saved successfully to {model_path} and {vec_path}")


def load_data(filepath='cleaned_reviews.csv'):
    if not os.path.exists(filepath):
        print(f"Error: '{filepath}' not found. Please run data_prep.py first.")
        return None, None
        
    df = pd.read_csv(filepath)
    # Ensure no missing values in needed columns
    df = df.dropna(subset=['cleaned_review', 'Sentiment'])
    return df['cleaned_review'].values, df['Sentiment'].values


def main():
    print("Loading prepared dataset...")
    X, y = load_data('cleaned_reviews.csv')
    
    if X is None:
        return
        
    # Standardize labels to match outputs
    y = [str(label).capitalize() for label in y]
        
    print("Splitting data into train/test sets (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    analyzer = HybridSentimentAnalyzer()
    
    print("Training the Logistic Regression Classifier using TF-IDF (NGrams 1-2)...")
    analyzer.train_ml_model(X_train, y_train)
    
    analyzer.evaluate_model(X_test, y_test)
    
    print("\nSaving trained models via joblib...")
    analyzer.save_models()
    
    # Run an inference example showcasing negation 
    sample_text = "The UI is not good at all, it's very terrible and confusing."
    print(f"\nExample Inference (Negation Test):")
    print(f"Review: '{sample_text}'")
    print(f"Predicted Sentiment: {analyzer.manual_ensemble_predict(sample_text)}")

if __name__ == "__main__":
    main()
