"""
Evaluate the /v1/classify API against a labelled CSV of emails.

Usage:
    python evaluate_classifier_api.py --input emails.csv --api-key sk-spam-...

CSV columns required: subject, from, snippet, true_label (1 = spam, 0 = legit).
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm

DOWNGRADE_SENDER_PATTERN = re.compile(
    r"no-?reply|noreply|notifications?@|updates?@|hello@|info@", re.IGNORECASE
)
DOWNGRADE_PROB_THRESHOLD = 0.99


def build_email_text(row: pd.Series) -> str:
    subject = str(row.get("subject", "") or "")
    sender = str(row.get("from", "") or "")
    snippet = str(row.get("snippet", "") or "")
    return f"Subject: {subject}\nFrom: {sender}\n\n{snippet}"


def classify_email(session: requests.Session, url: str, api_key: str, text: str, timeout: float):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = session.post(url, json={"text": text}, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def classify_dataframe(df: pd.DataFrame, url: str, api_key: str, timeout: float) -> pd.DataFrame:
    df = df.copy()
    labels, spam_probs, confidences, errors = [], [], [], []

    with requests.Session() as session:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
            text = build_email_text(row)
            try:
                result = classify_email(session, url, api_key, text, timeout)
                labels.append(result["label"])
                spam_probs.append(float(result["spam_prob"]))
                confidences.append(float(result.get("confidence", result["spam_prob"])))
                errors.append(None)
            except Exception as exc:
                labels.append(None)
                spam_probs.append(None)
                confidences.append(None)
                errors.append(str(exc))

    df["pred_label"] = labels
    df["spam_prob"] = spam_probs
    df["confidence"] = confidences
    df["api_error"] = errors
    return df


def apply_sender_downgrade_rule(df: pd.DataFrame) -> pd.DataFrame:
    def downgrade(row):
        if row["pred_label"] != "SPAM":
            return row["pred_label"]
        sender = str(row.get("from", "") or "")
        if row["spam_prob"] < DOWNGRADE_PROB_THRESHOLD and DOWNGRADE_SENDER_PATTERN.search(sender):
            return "HAM"
        return row["pred_label"]

    df = df.copy()
    df["pred_label_adjusted"] = df.apply(downgrade, axis=1)
    return df


def compute_metrics(y_true, pred_labels) -> dict:
    y_pred = [1 if label == "SPAM" else 0 for label in pred_labels]
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def write_markdown_report(metrics_baseline, metrics_adjusted, n_total, n_scored, n_errors, out_path):
    def cm_rows(cm):
        tn, fp, fn, tp = cm.ravel()
        return tn, fp, fn, tp

    tn_b, fp_b, fn_b, tp_b = cm_rows(metrics_baseline["confusion_matrix"])
    tn_a, fp_a, fn_a, tp_a = cm_rows(metrics_adjusted["confusion_matrix"])

    lines = [
        "# Spam Classifier API Evaluation",
        "",
        f"- Emails in CSV: {n_total}",
        f"- Successfully scored: {n_scored}",
        f"- API errors (excluded from metrics): {n_errors}",
        "",
        "## Metrics",
        "",
        "| Metric | Baseline | Adjusted (sender downgrade rule) |",
        "|---|---|---|",
        f"| Precision | {metrics_baseline['precision']:.4f} | {metrics_adjusted['precision']:.4f} |",
        f"| Recall | {metrics_baseline['recall']:.4f} | {metrics_adjusted['recall']:.4f} |",
        f"| F1 | {metrics_baseline['f1']:.4f} | {metrics_adjusted['f1']:.4f} |",
        "",
        "## Confusion Matrix — Baseline",
        "",
        "| | Predicted HAM | Predicted SPAM |",
        "|---|---|---|",
        f"| True HAM | {tn_b} | {fp_b} |",
        f"| True SPAM | {fn_b} | {tp_b} |",
        "",
        "## Confusion Matrix — Adjusted",
        "",
        "| | Predicted HAM | Predicted SPAM |",
        "|---|---|---|",
        f"| True HAM | {tn_a} | {fp_a} |",
        f"| True SPAM | {fn_a} | {tp_a} |",
        "",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_confusion_matrices(cm_baseline, cm_adjusted, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, cm, title in zip(
        axes, [cm_baseline, cm_adjusted], ["Baseline", "Adjusted (sender downgrade rule)"]
    ):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["HAM", "SPAM"],
            yticklabels=["HAM", "SPAM"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the spam classifier API against a labelled CSV.")
    parser.add_argument("--input", required=True, help="Path to CSV with subject, from, snippet, true_label")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/classify")
    parser.add_argument("--api-key", default=os.environ.get("SPAM_API_KEY"))
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--out-dir", default="outputs/evaluation")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("No API key provided. Pass --api-key or set SPAM_API_KEY.")

    df = pd.read_csv(args.input)
    required_cols = {"subject", "from", "snippet", "true_label"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing required columns: {sorted(missing)}")

    df = classify_dataframe(df, args.url, args.api_key, args.timeout)
    df = apply_sender_downgrade_rule(df)

    scored = df[df["api_error"].isna()].copy()
    n_errors = len(df) - len(scored)

    y_true = scored["true_label"].astype(int)
    metrics_baseline = compute_metrics(y_true, scored["pred_label"])
    metrics_adjusted = compute_metrics(y_true, scored["pred_label_adjusted"])

    os.makedirs(args.out_dir, exist_ok=True)
    md_path = os.path.join(args.out_dir, "metrics.md")
    png_path = os.path.join(args.out_dir, "confusion_matrix.png")

    write_markdown_report(metrics_baseline, metrics_adjusted, len(df), len(scored), n_errors, md_path)
    plot_confusion_matrices(
        metrics_baseline["confusion_matrix"], metrics_adjusted["confusion_matrix"], png_path
    )

    print(f"Scored {len(scored)}/{len(df)} emails ({n_errors} API errors excluded).")
    print(f"Baseline  -> precision={metrics_baseline['precision']:.4f} recall={metrics_baseline['recall']:.4f} f1={metrics_baseline['f1']:.4f}")
    print(f"Adjusted  -> precision={metrics_adjusted['precision']:.4f} recall={metrics_adjusted['recall']:.4f} f1={metrics_adjusted['f1']:.4f}")
    print(f"Report written to {md_path}")
    print(f"Confusion matrix plot written to {png_path}")


if __name__ == "__main__":
    main()
