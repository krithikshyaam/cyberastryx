"""
trainer.py - Full training pipeline for both model stages.
Fixed for transformers >= 5.0 (PyTorch) + tensorflow >= 2.16

Usage:
    python -m src.trainer --model baseline
    python -m src.trainer --model transformer
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from src import config
from src.data_loader import load_dataset, split_dataset, compute_weights
from src.preprocessor import SpamTokenizer
from src.models.baseline_model import build_bilstm_model, get_baseline_callbacks
from src.evaluator import evaluate_model, plot_training_history


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU detected: {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU detected — training on CPU.")


# ─────────────────────────────────────────────
# Stage 1: BiLSTM Baseline
# ─────────────────────────────────────────────

def train_baseline():
    print("\n" + "="*55)
    print("  STAGE 1: BiLSTM Baseline Training")
    print("="*55)

    df = load_dataset()
    train_df, val_df, test_df = split_dataset(df)
    class_weights = compute_weights(train_df["label"].values) if config.USE_CLASS_WEIGHTS else None

    tokenizer = SpamTokenizer()
    tokenizer.fit(train_df["text"])
    tokenizer.save()

    X_train = tokenizer.encode(train_df["text"])
    X_val   = tokenizer.encode(val_df["text"])
    X_test  = tokenizer.encode(test_df["text"])
    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    model = build_bilstm_model(vocab_size=tokenizer.vocab_size)
    os.makedirs(config.BASELINE_MODEL_PATH, exist_ok=True)
    callbacks = get_baseline_callbacks(config.BASELINE_MODEL_PATH)

    cfg = config.BASELINE
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "="*55)
    print("  BASELINE — TEST EVALUATION")
    print("="*55)
    y_pred_prob = model.predict(X_test, batch_size=256).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    evaluate_model(y_test, y_pred, y_pred_prob, model_name="BiLSTM Baseline")
    plot_training_history(history, save_path=f"{config.REPORTS_DIR}/baseline_training.png")

    print(f"\n✓ Baseline model saved -> {config.BASELINE_MODEL_PATH}")
    return model


# ─────────────────────────────────────────────
# Stage 2: BERT Fine-tuning (PyTorch)
# ─────────────────────────────────────────────

def train_transformer(variant="standard"):
    import torch
    from torch.optim import AdamW
    from src.models.transformer_model import (
        build_bert_model, build_bert_dataset, load_bert_tokenizer
    )

    print("\n" + "="*55)
    print(f"  STAGE 2: BERT Fine-tuning (PyTorch)")
    print("="*55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    df = load_dataset()
    train_df, val_df, test_df = split_dataset(df)

    hf_tokenizer = load_bert_tokenizer()
    cfg = config.TRANSFORMER

    train_loader = build_bert_dataset(train_df["text"], train_df["label"], hf_tokenizer, cfg["batch_size"], shuffle=True)
    val_loader   = build_bert_dataset(val_df["text"],   val_df["label"],   hf_tokenizer, cfg["batch_size"])
    test_loader  = build_bert_dataset(test_df["text"],  test_df["label"],  hf_tokenizer, cfg["batch_size"])

    model = build_bert_model().to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"])

    best_val_acc = 0
    save_path = config.TRANSFORMER_MODEL_PATH
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(cfg["epochs"]):
        # Train
        model.train()
        total_loss = correct = total = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validate
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{cfg['epochs']} — loss: {total_loss/len(train_loader):.4f} — acc: {train_acc:.4f} — val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(save_path)
            hf_tokenizer.save_pretrained(save_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

    # Evaluate on test set
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    evaluate_model(np.array(all_labels), np.array(all_preds), np.array(all_probs), model_name="BERT")
    print(f"\n✓ BERT model saved -> {save_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train spam detection model")
    parser.add_argument("--model", choices=["baseline", "transformer"], default="baseline")
    args = parser.parse_args()

    setup_gpu()
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    if args.model == "baseline":
        train_baseline()
    else:
        train_transformer()
