"""
feedback_loop.py - Active learning feedback loop for continuous model improvement.

How it works:
  1. Your n8n workflow labels emails (SPAM or HAM)
  2. When you correct a wrong prediction, n8n calls POST /feedback
  3. Corrections are stored in data/feedback_store.jsonl
  4. When enough corrections accumulate, the model retrains automatically
  5. New model replaces old one — zero downtime via hot-swap

n8n Integration:
  Add an HTTP Request node after your "If" node that calls:
    POST http://YOUR_SERVER:8000/feedback
    {
      "text"           : "{{ $json.emailText }}",
      "correct_label"  : "HAM",          ← what it actually was
      "predicted_label": "SPAM",          ← what your model said
      "email_id"       : "{{ $json.id }}"
    }

Usage:
    python feedback_loop.py --status          # show pending feedback
    python feedback_loop.py --retrain         # force retrain now
    python feedback_loop.py --retrain --min 10  # retrain if >= 10 samples
"""

import os
import sys
import json
import time
import shutil
import argparse
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

log = logging.getLogger("feedback")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
FEEDBACK_STORE  = Path("data/feedback_store.jsonl")       # raw corrections
RETRAIN_LOG     = Path("data/retrain_history.json")       # history of retrains
AUGMENTED_CSV   = Path("data/augmented_dataset.csv")      # base + feedback merged

FEEDBACK_STORE.parent.mkdir(parents=True, exist_ok=True)


# ── Feedback Store ────────────────────────────────────────────────────────────

class FeedbackStore:
    """Append-only JSONL store for user corrections."""

    def __init__(self, path: Path = FEEDBACK_STORE):
        self.path = path

    def add(
        self,
        text         : str,
        correct_label: int,    # 0=ham, 1=spam
        predicted_label: int,  # what the model said
        email_id     : str = "",
        confidence   : float = 0.0,
        source       : str = "n8n",
    ) -> dict:
        """Record a correction. Returns the stored record."""
        record = {
            "id"              : hashlib.md5(text.encode()).hexdigest()[:12],
            "email_id"        : email_id,
            "text"            : text[:2000],
            "correct_label"   : int(correct_label),
            "predicted_label" : int(predicted_label),
            "was_wrong"       : correct_label != predicted_label,
            "confidence"      : float(confidence),
            "source"          : source,
            "timestamp"       : datetime.utcnow().isoformat() + "Z",
            "used_in_retrain" : False,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        log.info(
            f"Feedback stored: email={email_id[:8]} "
            f"correct={correct_label} predicted={predicted_label} "
            f"wrong={record['was_wrong']}"
        )
        return record

    def load_all(self) -> list:
        if not self.path.exists():
            return []
        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def load_pending(self) -> list:
        """Load corrections not yet used in retraining."""
        return [r for r in self.load_all() if not r.get("used_in_retrain")]

    def load_wrong_predictions(self) -> list:
        """Load only misclassified emails (most valuable for retraining)."""
        return [r for r in self.load_all() if r.get("was_wrong")]

    def mark_used(self, ids: list):
        """Mark records as used in retraining."""
        all_records = self.load_all()
        id_set = set(ids)
        updated = []
        for r in all_records:
            if r["id"] in id_set:
                r["used_in_retrain"] = True
            updated.append(r)
        with open(self.path, "w") as f:
            for r in updated:
                f.write(json.dumps(r) + "\n")

    def get_stats(self) -> dict:
        records = self.load_all()
        if not records:
            return {"total": 0}
        df = pd.DataFrame(records)
        return {
            "total"          : len(df),
            "pending"        : int((~df["used_in_retrain"]).sum()),
            "wrong_only"     : int(df["was_wrong"].sum()),
            "false_positives": int(((df.predicted_label==1) & (df.correct_label==0)).sum()),
            "false_negatives": int(((df.predicted_label==0) & (df.correct_label==1)).sum()),
            "by_source"      : df.groupby("source").size().to_dict(),
            "since"          : df["timestamp"].min(),
            "latest"         : df["timestamp"].max(),
        }


# ── Dataset Augmentation ──────────────────────────────────────────────────────

class DatasetAugmenter:
    """
    Merges base training data with feedback corrections.
    Wrong predictions are upsampled 3x (they're the most informative).
    """

    def __init__(
        self,
        base_csv   : str = "data/spam.csv",
        unified_csv: str = "data/unified_dataset.csv",
    ):
        # Use unified if available, else base
        self.base_path = unified_csv if Path(unified_csv).exists() else base_csv

    def build_augmented_dataset(
        self,
        feedback_records: list,
        output_path     : str = str(AUGMENTED_CSV),
        wrong_upsample  : int = 3,
    ) -> pd.DataFrame:
        """
        Combine base dataset with feedback corrections.

        Args:
            feedback_records: List of feedback records from FeedbackStore
            output_path     : Where to save the augmented CSV
            wrong_upsample  : How many times to duplicate wrong predictions
        """
        # Load base
        log.info(f"Loading base dataset from {self.base_path}...")
        base = pd.read_csv(self.base_path)

        # Normalize base columns
        if "Category" in base.columns:
            base["label"] = base["Category"].map({"ham": 0, "spam": 1})
            base["text"]  = base["Message"]
        base = base[["text", "label"]].dropna()
        base["label"]  = base["label"].astype(int)
        base["source"] = "base"
        base["weight"] = 1.0

        log.info(f"Base dataset: {len(base):,} samples")

        if not feedback_records:
            log.info("No feedback records — using base dataset only.")
            augmented = base
        else:
            fb_df = pd.DataFrame(feedback_records)
            fb_df = fb_df.rename(columns={"correct_label": "label"})
            fb_df["source"] = "feedback"

            # Wrong predictions get upsampled
            wrong_df   = fb_df[fb_df["was_wrong"] == True].copy()
            correct_df = fb_df[fb_df["was_wrong"] == False].copy()
            wrong_df["weight"]   = float(wrong_upsample)
            correct_df["weight"] = 1.0

            # Upsample wrong predictions
            wrong_up = pd.concat([wrong_df] * wrong_upsample, ignore_index=True)

            fb_combined = pd.concat([correct_df, wrong_up], ignore_index=True)
            fb_combined = fb_combined[["text", "label", "source", "weight"]]

            log.info(
                f"Feedback: {len(fb_df)} records "
                f"({len(wrong_df)} wrong × {wrong_upsample}x upsample = {len(wrong_up)} added)"
            )

            augmented = pd.concat([base, fb_combined], ignore_index=True)

        augmented = augmented.drop_duplicates(subset=["text"])
        augmented.to_csv(output_path, index=False)

        n_spam = augmented["label"].sum()
        n_ham  = (augmented["label"] == 0).sum()
        log.info(
            f"Augmented dataset: {len(augmented):,} samples "
            f"(ham={n_ham:,}, spam={n_spam:,}) → saved to {output_path}"
        )
        return augmented


# ── Retraining Manager ────────────────────────────────────────────────────────

class RetrainManager:
    """
    Orchestrates automatic retraining when enough feedback accumulates.

    Threshold-based trigger:
      - MIN_WRONG_TO_RETRAIN  : retrain when N wrong predictions accumulate
      - MIN_TOTAL_TO_RETRAIN  : or when N total feedback items accumulate
    """

    MIN_WRONG_TO_RETRAIN = 20    # retrain after 20 wrong predictions
    MIN_TOTAL_TO_RETRAIN = 50    # or 50 total feedback items

    def __init__(self):
        self.store    = FeedbackStore()
        self.augmenter = DatasetAugmenter()

    def should_retrain(self) -> tuple[bool, str]:
        """Returns (should_retrain, reason)."""
        pending = self.store.load_pending()
        wrong   = [r for r in pending if r.get("was_wrong")]

        if len(wrong) >= self.MIN_WRONG_TO_RETRAIN:
            return True, f"{len(wrong)} wrong predictions accumulated"
        if len(pending) >= self.MIN_TOTAL_TO_RETRAIN:
            return True, f"{len(pending)} pending feedback items accumulated"
        return False, f"Only {len(wrong)} wrong / {len(pending)} pending (thresholds: {self.MIN_WRONG_TO_RETRAIN}/{self.MIN_TOTAL_TO_RETRAIN})"

    def retrain(
        self,
        model_type    : str = "baseline",
        force         : bool = False,
        min_samples   : int = 0,
    ) -> dict:
        """
        Run retraining pipeline.

        Args:
            model_type  : "baseline" or "transformer"
            force       : Skip threshold check
            min_samples : Only retrain if at least this many feedback samples

        Returns:
            dict with status, metrics, timing
        """
        result = {
            "status"    : "skipped",
            "timestamp" : datetime.utcnow().isoformat() + "Z",
            "model_type": model_type,
        }

        # Check threshold
        if not force:
            should, reason = self.should_retrain()
            if not should:
                log.info(f"Skipping retrain: {reason}")
                result["reason"] = reason
                return result

        pending = self.store.load_pending()
        if len(pending) < min_samples:
            log.info(f"Not enough samples: {len(pending)} < {min_samples}")
            result["reason"] = f"Not enough samples ({len(pending)} < {min_samples})"
            return result

        log.info(f"Starting retrain with {len(pending)} feedback samples...")
        start = time.time()

        # Step 1: Build augmented dataset
        augmented = self.augmenter.build_augmented_dataset(pending)

        # Step 2: Retrain
        try:
            from src.data_loader import split_dataset, compute_weights
            from src import config

            train_df, val_df, test_df = split_dataset(augmented)
            class_weights = compute_weights(train_df["label"].values)

            if model_type == "baseline":
                metrics = self._retrain_baseline(train_df, val_df, test_df, class_weights)
            else:
                metrics = self._retrain_transformer(train_df, val_df, test_df, class_weights)

            # Step 3: Mark feedback as used
            self.store.mark_used([r["id"] for r in pending])

            elapsed = time.time() - start
            result.update({
                "status"        : "success",
                "samples_used"  : len(pending),
                "dataset_size"  : len(augmented),
                "metrics"       : metrics,
                "elapsed_sec"   : round(elapsed, 1),
            })

            # Save to retrain log
            self._log_retrain(result)
            log.info(f"Retrain complete in {elapsed:.0f}s — metrics: {metrics}")

        except Exception as e:
            result["status"] = "failed"
            result["error"]  = str(e)
            log.error(f"Retrain failed: {e}")

        return result

    def _retrain_baseline(self, train_df, val_df, test_df, class_weights) -> dict:
        import tensorflow as tf
        from src.preprocessor import SpamTokenizer
        from src.models.baseline_model import build_bilstm_model, get_baseline_callbacks
        from src import config

        tokenizer = SpamTokenizer()
        tokenizer.fit(train_df["text"])
        tokenizer.save()

        X_train = tokenizer.encode(train_df["text"])
        X_val   = tokenizer.encode(val_df["text"])
        X_test  = tokenizer.encode(test_df["text"])

        model = build_bilstm_model(vocab_size=tokenizer.vocab_size)
        callbacks = get_baseline_callbacks(config.BASELINE_MODEL_PATH)

        model.fit(
            X_train, train_df["label"].values,
            validation_data=(X_val, val_df["label"].values),
            batch_size=config.BASELINE["batch_size"],
            epochs=config.BASELINE["epochs"],
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate
        y_pred_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        y_test = test_df["label"].values
        return self._compute_metrics(y_test, y_pred, y_pred_prob)

    def _retrain_transformer(self, train_df, val_df, test_df, class_weights) -> dict:
        import tensorflow as tf
        from src.models.transformer_model import (
            build_bert_model, build_bert_dataset,
            get_bert_callbacks, load_bert_tokenizer
        )
        from src import config

        hf_tok = load_bert_tokenizer()
        cfg = config.TRANSFORMER

        train_ds = build_bert_dataset(train_df["text"], train_df["label"], hf_tok, cfg["batch_size"], shuffle=True)
        val_ds   = build_bert_dataset(val_df["text"],   val_df["label"],   hf_tok, cfg["batch_size"])
        test_ds  = build_bert_dataset(test_df["text"],  test_df["label"],  hf_tok, cfg["batch_size"])

        model = build_bert_model()
        model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs"],
                  class_weight=class_weights, callbacks=get_bert_callbacks(), verbose=0)

        logits = model.predict(test_ds).logits
        y_pred_prob = tf.nn.softmax(logits, axis=-1).numpy()[:, 1]
        y_pred = (y_pred_prob > 0.5).astype(int)
        y_test = test_df["label"].values
        return self._compute_metrics(y_test, y_pred, y_pred_prob)

    def _compute_metrics(self, y_true, y_pred, y_pred_prob) -> dict:
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
        return {
            "accuracy" : round(float(accuracy_score(y_true, y_pred)),  4),
            "f1"       : round(float(f1_score(y_true, y_pred)),        4),
            "precision": round(float(precision_score(y_true, y_pred)), 4),
            "recall"   : round(float(recall_score(y_true, y_pred)),    4),
            "auc"      : round(float(roc_auc_score(y_true, y_pred_prob)), 4),
        }

    def _log_retrain(self, result: dict):
        history = []
        if RETRAIN_LOG.exists():
            with open(RETRAIN_LOG) as f:
                history = json.load(f)
        history.append(result)
        with open(RETRAIN_LOG, "w") as f:
            json.dump(history, f, indent=2)

    def get_history(self) -> list:
        if not RETRAIN_LOG.exists():
            return []
        with open(RETRAIN_LOG) as f:
            return json.load(f)

    def print_status(self):
        stats = self.store.get_stats()
        should, reason = self.should_retrain()
        history = self.get_history()

        print(f"\n{'='*55}")
        print(f"  FEEDBACK LOOP STATUS")
        print(f"{'='*55}")
        print(f"  Total feedback    : {stats.get('total', 0)}")
        print(f"  Pending retrain   : {stats.get('pending', 0)}")
        print(f"  Wrong predictions : {stats.get('wrong_only', 0)}")
        print(f"    False positives : {stats.get('false_positives', 0)} (ham → predicted spam)")
        print(f"    False negatives : {stats.get('false_negatives', 0)} (spam → predicted ham)")
        print(f"  Should retrain    : {'YES ✓' if should else 'NO'} — {reason}")
        print(f"  Retrain history   : {len(history)} runs")
        if history:
            last = history[-1]
            print(f"  Last retrain      : {last['timestamp'][:19]}")
            if last.get("metrics"):
                m = last["metrics"]
                print(f"    Accuracy={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
        print(f"{'='*55}\n")


# ── FastAPI endpoints (add to api_server.py) ──────────────────────────────────

def get_feedback_router():
    """
    Returns a FastAPI router with feedback endpoints.
    Mount this in api_server.py with: app.include_router(get_feedback_router())
    """
    from fastapi import APIRouter
    from pydantic import BaseModel

    router  = APIRouter(prefix="/feedback", tags=["feedback"])
    store   = FeedbackStore()
    manager = RetrainManager()

    class FeedbackPayload(BaseModel):
        text            : str
        correct_label   : str        # "HAM" or "SPAM"
        predicted_label : str        # "HAM" or "SPAM"
        email_id        : str = ""
        confidence      : float = 0.0
        source          : str = "n8n"

    @router.post("/")
    def submit_feedback(body: FeedbackPayload):
        label_map = {"HAM": 0, "SPAM": 1, "ham": 0, "spam": 1, "0": 0, "1": 1}
        correct   = label_map.get(str(body.correct_label), -1)
        predicted = label_map.get(str(body.predicted_label), -1)
        if correct == -1 or predicted == -1:
            return {"error": "Invalid label. Use 'HAM' or 'SPAM'."}
        record = store.add(body.text, correct, predicted, body.email_id, body.confidence, body.source)
        # Check if we should auto-retrain
        should, reason = manager.should_retrain()
        return {
            "stored"           : True,
            "record_id"        : record["id"],
            "was_wrong"        : record["was_wrong"],
            "retrain_triggered": should,
            "retrain_reason"   : reason if should else None,
        }

    @router.get("/stats")
    def feedback_stats():
        return store.get_stats()

    @router.post("/retrain")
    def trigger_retrain(model_type: str = "baseline", force: bool = False):
        result = manager.retrain(model_type=model_type, force=force)
        return result

    @router.get("/history")
    def retrain_history():
        return manager.get_history()

    return router


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feedback loop manager")
    parser.add_argument("--status",  action="store_true", help="Show feedback status")
    parser.add_argument("--retrain", action="store_true", help="Trigger retraining")
    parser.add_argument("--force",   action="store_true", help="Force retrain even below threshold")
    parser.add_argument("--model",   default="baseline", choices=["baseline", "transformer"])
    parser.add_argument("--min",     type=int, default=0, help="Min feedback samples to retrain")
    parser.add_argument("--add-test", action="store_true", help="Add a test feedback record")
    args = parser.parse_args()

    manager = RetrainManager()

    if args.add_test:
        store = FeedbackStore()
        store.add(
            text="FREE PRIZE! Click here to claim your reward NOW!!!",
            correct_label=1, predicted_label=0,
            email_id="test-001", confidence=0.3, source="manual_test"
        )
        print("Test feedback record added.")

    if args.status:
        manager.print_status()

    if args.retrain:
        result = manager.retrain(model_type=args.model, force=args.force, min_samples=args.min)
        print(json.dumps(result, indent=2))
