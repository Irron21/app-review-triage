import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import os

# Download necessary NLTK corpora
def setup_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    """
    Cleans text by lowering casing, removing punctuation & stopwords, and applying lemmatization.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase
    text = text.lower()
    
    # 2. Tokenize
    tokens = word_tokenize(text)
    
    # 3. Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    
    # 4. Remove stopwords (Excluding crucial negators and intensifiers to preserve contextual meaning)
    stop_words = set(stopwords.words('english'))
    exclusions = {'not', 'no', 'nor', 'against', 'too', 'very'}
    stop_words = stop_words - exclusions
    
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Apply lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

def main():
    setup_nltk()
    
    dataset_path = 'googleplaystore_user_reviews.csv'
    
    # 1. Read the kaggle CSV
    if not os.path.exists(dataset_path):
        print(f"Error: Could not find '{dataset_path}'. Please ensure the true dataset is downloaded and present in the directory.")
        return

    print(f"Loading dataset from '{dataset_path}'...")
    df = pd.read_csv(dataset_path)
    
    # 2. Data Cleaning: Drop rows where Translated_Review or Sentiment is NaN
    initial_rows = len(df)
    df = df.dropna(subset=['Translated_Review', 'Sentiment'])
    print(f"Dropped missing values. Shape went from {initial_rows} to {len(df)} rows.")
    
    print("\n--- Original Sentiment Distribution ---")
    print(df['Sentiment'].value_counts())
    
    # 3. Sampling: 5,000 rows stratified for an equal mix of Positive, Negative, and Neutral
    n_classes = df['Sentiment'].nunique()
    
    if n_classes == 0:
        print("No valid sentiment classes found.")
        return
        
    target_per_class = 15000 // n_classes
    remainder = 15000 % n_classes
    
    sampled_frames = []
    # Loop over all unique classes to ensure an equal mix
    for i, sentiment in enumerate(df['Sentiment'].unique()):
        subset = df[df['Sentiment'] == sentiment]
        # Distribute any remainder to ensure exactly 5000 total (if available)
        n = target_per_class + (1 if i < remainder else 0)
        # Cannot sample more than what's available
        n = min(len(subset), n)
        
        sampled_frames.append(subset.sample(n=n, random_state=42))
        
    # Combine and shuffle the stratified subsets
    df_sampled = pd.concat(sampled_frames).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\n--- Sampled Sentiment Distribution (Target: 5,000 equal mix) ---")
    print(df_sampled['Sentiment'].value_counts())
    print(f"Total rows in sample dataset: {len(df_sampled)}")
    
    # 4. Text Preprocessing
    print("\nApplying NLP text preprocessing (lowercase, punctuation/stopwords removal, lemmatization)...")
    print("Please wait, this will take a moment for 5,000 rows...")
    df_sampled['cleaned_review'] = df_sampled['Translated_Review'].apply(preprocess_text)
    
    # 5. Save the output
    output_path = 'cleaned_reviews.csv'
    df_sampled.to_csv(output_path, index=False)
    print(f"\nPreprocessing pipeline complete! Saved final 5,000 rows to '{output_path}'.")

if __name__ == "__main__":
    main()
