"""
preprocessor.py - Text cleaning, tokenization, and sequence padding.
Used by the BiLSTM baseline model (Stage 1).
BERT uses its own HuggingFace tokenizer (handled in transformer_model.py).
"""

import re
import json
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src import config

# Download NLTK data on first run
for resource in ["stopwords", "punkt"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

STOPWORDS = set(stopwords.words("english"))


# ─────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────

def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Clean a single email string.

    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove email addresses
      4. Remove HTML tags
      5. Remove punctuation & special characters
      6. Collapse whitespace
      7. Optionally remove stopwords
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)              # URLs
    text = re.sub(r"\S+@\S+", " EMAIL ", text)                     # email addresses
    text = re.sub(r"<[^>]+>", " ", text)                           # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                          # keep only letters
    text = re.sub(r"\s+", " ", text).strip()                       # whitespace

    if remove_stopwords:
        text = " ".join(w for w in text.split() if w not in STOPWORDS)
    return text


def clean_texts(texts, remove_stopwords: bool = False):
    return [clean_text(t, remove_stopwords) for t in texts]


# ─────────────────────────────────────────────
# Keras Tokenizer (baseline model)
# ─────────────────────────────────────────────

class SpamTokenizer:
    """Wraps Keras Tokenizer with save/load support."""

    def __init__(self):
        self.tokenizer = Tokenizer(
            num_words=config.MAX_VOCAB_SIZE,
            oov_token="<OOV>"
        )
        self.fitted = False

    def fit(self, texts: list):
        """Fit tokenizer on training texts."""
        cleaned = clean_texts(texts)
        self.tokenizer.fit_on_texts(cleaned)
        self.fitted = True
        vocab_size = min(config.MAX_VOCAB_SIZE, len(self.tokenizer.word_index) + 1)
        print(f"Tokenizer fitted. Vocabulary size: {vocab_size:,}")
        return self

    def encode(self, texts: list, max_len: int = config.MAX_SEQ_LEN) -> np.ndarray:
        """Convert texts → padded integer sequences."""
        if not self.fitted:
            raise RuntimeError("Call .fit() before .encode()")
        cleaned = clean_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
        return padded

    def save(self, path: str = config.TOKENIZER_PATH):
        """Persist tokenizer to JSON."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tokenizer_json = self.tokenizer.to_json()
        with open(path, "w") as f:
            json.dump(tokenizer_json, f)
        print(f"Tokenizer saved → {path}")

    @classmethod
    def load(cls, path: str = config.TOKENIZER_PATH):
        """Load tokenizer from JSON."""
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        with open(path) as f:
            tokenizer_json = json.load(f)
        obj = cls()
        obj.tokenizer = tokenizer_from_json(tokenizer_json)
        obj.fitted = True
        print(f"Tokenizer loaded ← {path}")
        return obj

    @property
    def vocab_size(self) -> int:
        return min(config.MAX_VOCAB_SIZE, len(self.tokenizer.word_index) + 1)


if __name__ == "__main__":
    sample = ["FREE MONEY! Click here now!!!", "Meeting at 3pm, see you soon."]
    print("Original:", sample)
    print("Cleaned :", clean_texts(sample))

    tok = SpamTokenizer()
    tok.fit(sample)
    encoded = tok.encode(sample, max_len=20)
    print("Encoded :", encoded)
