"""
predict.py - Run inference on new emails using a trained model.

Usage (CLI):
    python src/predict.py --model baseline --text "You won a free prize!"
    python src/predict.py --model transformer --text "See you at the meeting."

Usage (Python API):
    from src.predict import SpamPredictor
    predictor = SpamPredictor(model_type="baseline")
    result = predictor.predict("Congratulations! Click here to claim your reward.")
    print(result)  # {'label': 'SPAM', 'confidence': 0.9987, 'spam_prob': 0.9987}
"""

import argparse
import numpy as np
import tensorflow as tf
from src import config
from src.preprocessor import SpamTokenizer
from src.models.baseline_model import load_baseline_model


class SpamPredictor:
    """
    Unified predictor for both model stages.

    Args:
        model_type: "baseline" or "transformer"
        bert_variant: "standard" or "custom" (only for transformer)
    """

    def __init__(self, model_type: str = "baseline", bert_variant: str = "standard"):
        self.model_type = model_type
        self.bert_variant = bert_variant
        self._load()

    def _load(self):
        if self.model_type == "baseline":
            self.tokenizer = SpamTokenizer.load()
            self.model = load_baseline_model()
            print("Baseline predictor ready.")

        elif self.model_type == "transformer":
            from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
            from src.models.transformer_model import build_custom_bert_model

            if self.bert_variant == "custom":
                model_path = config.TRANSFORMER_MODEL_PATH + "_custom/best_model"
                self.model = tf.keras.models.load_model(model_path)
            else:
                model_path = config.TRANSFORMER_MODEL_PATH + "/best_model"
                self.model = TFAutoModelForSequenceClassification.from_pretrained(model_path)

            self.hf_tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER["model_name"])
            print("Transformer predictor ready.")
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _predict_baseline(self, texts: list) -> np.ndarray:
        """Return spam probabilities using BiLSTM model."""
        encoded = self.tokenizer.encode(texts)
        probs = self.model.predict(encoded, verbose=0).flatten()
        return probs

    def _predict_transformer(self, texts: list) -> np.ndarray:
        """Return spam probabilities using BERT model."""
        cfg = config.TRANSFORMER
        encodings = self.hf_tokenizer(
            texts,
            max_length=cfg["max_length"],
            truncation=True,
            padding="max_length",
            return_tensors="tf"
        )

        if self.bert_variant == "custom":
            probs = self.model.predict(
                {
                    "input_ids"      : encodings["input_ids"],
                    "attention_mask" : encodings["attention_mask"],
                    "token_type_ids" : encodings["token_type_ids"],
                },
                verbose=0
            ).flatten()
        else:
            outputs = self.model(encodings)
            probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[:, 1]

        return probs

    def predict(self, text: str) -> dict:
        """
        Predict a single email.

        Returns:
            dict with keys: label, confidence, spam_prob
        """
        texts = [text]
        if self.model_type == "baseline":
            probs = self._predict_baseline(texts)
        else:
            probs = self._predict_transformer(texts)

        spam_prob = float(probs[0])
        label = "SPAM" if spam_prob >= 0.5 else "HAM"
        confidence = spam_prob if label == "SPAM" else 1 - spam_prob

        return {
            "label"      : label,
            "confidence" : round(confidence, 4),
            "spam_prob"  : round(spam_prob, 4),
            "ham_prob"   : round(1 - spam_prob, 4),
        }

    def predict_batch(self, texts: list) -> list:
        """Predict a list of emails. Returns list of dicts."""
        if self.model_type == "baseline":
            probs = self._predict_baseline(texts)
        else:
            probs = self._predict_transformer(texts)

        results = []
        for spam_prob in probs:
            spam_prob = float(spam_prob)
            label = "SPAM" if spam_prob >= 0.5 else "HAM"
            confidence = spam_prob if label == "SPAM" else 1 - spam_prob
            results.append({
                "label"      : label,
                "confidence" : round(confidence, 4),
                "spam_prob"  : round(spam_prob, 4),
                "ham_prob"   : round(1 - spam_prob, 4),
            })
        return results


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict spam/ham for an email")
    parser.add_argument("--model", choices=["baseline", "transformer"], default="baseline")
    parser.add_argument("--bert-variant", choices=["standard", "custom"], default="standard")
    parser.add_argument("--text", type=str, required=True, help="Email text to classify")
    args = parser.parse_args()

    predictor = SpamPredictor(model_type=args.model, bert_variant=args.bert_variant)
    result = predictor.predict(args.text)

    print("\n" + "="*40)
    print(f"  Input : {args.text[:80]}...")
    print(f"  Label : {result['label']}")
    print(f"  Spam  : {result['spam_prob']*100:.1f}%")
    print(f"  Ham   : {result['ham_prob']*100:.1f}%")
    print("="*40)
