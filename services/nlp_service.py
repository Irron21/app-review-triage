import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def get_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

# Securely extract resources asynchronously globally upon instantiation
get_nltk_resources()

class NLPService:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Retain vital sentiment modifiers seamlessly against stop-word removal arrays
        self.exclusions = {'not', 'no', 'nor', 'against', 'too', 'very'}
        self.stop_words = self.stop_words - self.exclusions
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in string.punctuation]
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return " ".join(tokens)
