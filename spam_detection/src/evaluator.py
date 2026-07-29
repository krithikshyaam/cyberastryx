"""
evaluator.py - Evaluation metrics, confusion matrix, and training plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)
from src import config


def evaluate_model(y_true, y_pred, y_pred_prob=None, model_name: str = "Model"):
    """
    Print full evaluation report and save plots.

    Args:
        y_true:      Ground truth labels (0/1)
        y_pred:      Predicted labels (0/1)
        y_pred_prob: Predicted probabilities for class 1 (optional)
        model_name:  Name for display and file naming
    """
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    print(f"\n{'─'*50}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'─'*50}")

    # Classification report
    print(classification_report(y_true, y_pred, target_names=["Ham (0)", "Spam (1)"]))

    # F1
    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score (spam): {f1:.4f}")

    # ROC AUC
    if y_pred_prob is not None:
        auc = roc_auc_score(y_true, y_pred_prob)
        print(f"ROC AUC Score  : {auc:.4f}")
        ap  = average_precision_score(y_true, y_pred_prob)
        print(f"Avg Precision  : {ap:.4f}")

    # Confusion matrix
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    cm_path = os.path.join(config.REPORTS_DIR, f"{safe_name}_confusion.png")
    _plot_confusion_matrix(y_true, y_pred, model_name, save_path=cm_path)
    print(f"Confusion matrix saved → {cm_path}")

    # ROC curve
    if y_pred_prob is not None:
        roc_path = os.path.join(config.REPORTS_DIR, f"{safe_name}_roc.png")
        _plot_roc_curve(y_true, y_pred_prob, model_name, save_path=roc_path)
        print(f"ROC curve saved        → {roc_path}")

        pr_path = os.path.join(config.REPORTS_DIR, f"{safe_name}_pr.png")
        _plot_pr_curve(y_true, y_pred_prob, model_name, save_path=pr_path)
        print(f"PR curve saved         → {pr_path}")

    print(f"{'─'*50}\n")


def _plot_confusion_matrix(y_true, y_pred, title: str, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {title}", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_roc_curve(y_true, y_pred_prob, title: str, save_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {title}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_pr_curve(y_true, y_pred_prob, title: str, save_path: str):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    ap = average_precision_score(y_true, y_pred_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall Curve — {title}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_history(history, save_path: str):
    """Plot and save train/val accuracy and loss curves."""
    metrics = [k for k in history.history if not k.startswith("val_")]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.plot(history.history[metric], label="Train")
        val_key = f"val_{metric}"
        if val_key in history.history:
            ax.plot(history.history[val_key], label="Val")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training history saved → {save_path}")
