"""
monitoring.py - Production monitoring for your spam detection API.

Logs every prediction and provides:
  - Real-time accuracy tracking
  - Confidence distribution analysis
  - Drift detection (model degrading over time?)
  - Daily/weekly summary reports
  - HTML dashboard (open in browser)

Usage:
    python monitoring.py --dashboard          # generate HTML dashboard
    python monitoring.py --report             # print text report
    python monitoring.py --stats --days 7     # last 7 days stats
    python monitoring.py --export             # export logs to CSV
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

log = logging.getLogger("monitoring")

# ── Paths ─────────────────────────────────────────────────────────────────────
LOG_DIR       = Path("outputs/monitoring")
PRED_LOG      = LOG_DIR / "predictions.jsonl"
DAILY_LOG     = LOG_DIR / "daily_stats.json"
DASHBOARD_HTML= LOG_DIR / "dashboard.html"

LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Prediction Logger ─────────────────────────────────────────────────────────

class PredictionLogger:
    """
    Logs every API prediction to a JSONL file.
    Thread-safe via file append (each write is atomic on most OS).
    """

    def __init__(self, path: Path = PRED_LOG):
        self.path = path

    def log(
        self,
        text          : str,
        spam_prob     : float,
        label         : str,
        confidence    : float,
        model_type    : str,
        response_ms   : float = 0.0,
        email_id      : str = "",
        correct_label : Optional[str] = None,   # filled in after feedback
    ):
        record = {
            "ts"           : datetime.utcnow().isoformat() + "Z",
            "email_id"     : email_id,
            "label"        : label,
            "spam_prob"    : round(spam_prob, 4),
            "confidence"   : round(confidence, 4),
            "model_type"   : model_type,
            "response_ms"  : round(response_ms, 1),
            "text_len"     : len(text),
            "correct_label": correct_label,
            "was_correct"  : (correct_label == label) if correct_label else None,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load(self, days: int = None) -> pd.DataFrame:
        """Load prediction logs, optionally filtered to last N days."""
        if not self.path.exists():
            return pd.DataFrame()
        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["ts"] = pd.to_datetime(df["ts"])

        if days:
            from datetime import timezone
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df[df["ts"] >= cutoff]

        return df

    def update_correct_label(self, email_id: str, correct_label: str):
        """Update a prediction record with the ground truth label (from feedback)."""
        records = []
        updated = 0
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("email_id") == email_id:
                        r["correct_label"] = correct_label
                        r["was_correct"]   = (correct_label == r["label"])
                        updated += 1
                    records.append(r)
                except json.JSONDecodeError:
                    pass
        with open(self.path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        if updated:
            log.info(f"Updated {updated} record(s) for email_id={email_id}")


# ── Stats Computer ────────────────────────────────────────────────────────────

class MonitoringStats:
    def __init__(self):
        self.logger = PredictionLogger()

    def compute(self, days: int = 30) -> dict:
        df = self.logger.load(days=days)
        if df.empty:
            return {"error": "No prediction logs found.", "days": days}

        stats = {
            "period_days"    : days,
            "total_requests" : len(df),
            "spam_count"     : int((df.label == "SPAM").sum()),
            "ham_count"      : int((df.label == "HAM").sum()),
            "spam_rate"      : round(float((df.label == "SPAM").mean()), 4),
            "avg_confidence" : round(float(df.confidence.mean()), 4),
            "avg_spam_prob"  : round(float(df.spam_prob.mean()), 4),
            "avg_response_ms": round(float(df.response_ms.mean()), 1),
            "p95_response_ms": round(float(df.response_ms.quantile(0.95)), 1),
            "model_types"    : df.model_type.value_counts().to_dict(),
            "computed_at"    : datetime.utcnow().isoformat() + "Z",
        }

        # Accuracy (only where ground truth is known)
        labelled = df[df.was_correct.notna()]
        if len(labelled) > 0:
            stats["labelled_count"]   = len(labelled)
            stats["accuracy"]         = round(float(labelled.was_correct.mean()), 4)
            stats["false_positive_rate"] = round(
                float(((labelled.label=="SPAM") & (~labelled.was_correct)).sum() / max(1, (labelled.label=="SPAM").sum())), 4
            )
            stats["false_negative_rate"] = round(
                float(((labelled.label=="HAM") & (~labelled.was_correct)).sum() / max(1, (labelled.label=="HAM").sum())), 4
            )

        # Confidence distribution buckets
        buckets = [0, 0.6, 0.7, 0.8, 0.9, 1.01]
        labels  = ["<60%", "60-70%", "70-80%", "80-90%", ">90%"]
        counts, _ = np.histogram(df.confidence, bins=buckets)
        stats["confidence_distribution"] = dict(zip(labels, counts.tolist()))

        # Drift: compare first-half vs second-half spam rate
        half = len(df) // 2
        if half > 10:
            first_spam  = float((df.iloc[:half].label == "SPAM").mean())
            second_spam = float((df.iloc[half:].label == "SPAM").mean())
            stats["drift"] = {
                "first_half_spam_rate" : round(first_spam, 4),
                "second_half_spam_rate": round(second_spam, 4),
                "delta"                : round(second_spam - first_spam, 4),
                "alert"                : abs(second_spam - first_spam) > 0.15,
            }

        # Daily breakdown
        df["date"] = df["ts"].dt.date.astype(str)
        daily = df.groupby("date").agg(
            requests   =("label", "count"),
            spam       =("label", lambda x: (x=="SPAM").sum()),
            avg_conf   =("confidence", "mean"),
            avg_resp_ms=("response_ms", "mean"),
        ).round(3)
        stats["daily"] = daily.to_dict(orient="index")

        return stats

    def print_report(self, days: int = 7):
        stats = self.compute(days=days)
        if "error" in stats:
            print(f"\n  {stats['error']}")
            return

        print(f"\n{'='*60}")
        print(f"  SPAM DETECTION MONITORING — Last {days} days")
        print(f"{'='*60}")
        print(f"  Total requests   : {stats['total_requests']:,}")
        print(f"  Spam detected    : {stats['spam_count']:,} ({stats['spam_rate']*100:.1f}%)")
        print(f"  Ham passed       : {stats['ham_count']:,}")
        print(f"  Avg confidence   : {stats['avg_confidence']*100:.1f}%")
        print(f"  Avg response     : {stats['avg_response_ms']:.0f}ms (p95: {stats['p95_response_ms']:.0f}ms)")

        if "accuracy" in stats:
            print(f"\n  Accuracy (labelled {stats['labelled_count']} samples):")
            print(f"    Overall        : {stats['accuracy']*100:.2f}%")
            print(f"    False positive : {stats['false_positive_rate']*100:.2f}% (ham→spam)")
            print(f"    False negative : {stats['false_negative_rate']*100:.2f}% (spam→ham)")

        if "drift" in stats:
            d = stats["drift"]
            alert = "⚠ DRIFT ALERT" if d["alert"] else "✓ Stable"
            print(f"\n  Drift detection : {alert}")
            print(f"    First half spam rate  : {d['first_half_spam_rate']*100:.1f}%")
            print(f"    Second half spam rate : {d['second_half_spam_rate']*100:.1f}%")
            print(f"    Delta                 : {d['delta']*100:+.1f}%")

        print(f"\n  Confidence distribution:")
        for bucket, count in stats["confidence_distribution"].items():
            bar = "█" * (count * 20 // max(1, stats["total_requests"]))
            print(f"    {bucket:8s} {bar} {count}")

        if stats.get("daily"):
            print(f"\n  Daily breakdown:")
            print(f"    {'Date':<12} {'Requests':>9} {'Spam':>6} {'Spam%':>6} {'Conf%':>6} {'ms':>6}")
            print(f"    {'─'*12} {'─'*9} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
            for date, row in list(stats["daily"].items())[-7:]:
                spam_pct = row["spam"] / max(1, row["requests"]) * 100
                print(f"    {date:<12} {row['requests']:>9,} {row['spam']:>6} {spam_pct:>5.1f}% {row['avg_conf']*100:>5.1f}% {row['avg_resp_ms']:>5.0f}")

        print(f"{'='*60}\n")


# ── HTML Dashboard ────────────────────────────────────────────────────────────

def generate_dashboard(days: int = 30, output: Path = DASHBOARD_HTML):
    """Generate a standalone HTML monitoring dashboard."""
    monitor = MonitoringStats()
    stats   = monitor.compute(days=days)

    # Daily chart data
    daily_dates  = list(stats.get("daily", {}).keys())
    daily_counts = [v["requests"] for v in stats.get("daily", {}).values()]
    daily_spam   = [v["spam"]     for v in stats.get("daily", {}).values()]
    daily_ham    = [c - s for c, s in zip(daily_counts, daily_spam)]

    conf_labels = list(stats.get("confidence_distribution", {}).keys())
    conf_counts = list(stats.get("confidence_distribution", {}).values())

    accuracy  = stats.get("accuracy", "N/A")
    acc_text  = f"{accuracy*100:.2f}%" if isinstance(accuracy, float) else "N/A (awaiting feedback)"
    drift     = stats.get("drift", {})
    drift_alert = drift.get("alert", False)
    drift_delta = drift.get("delta", 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spam Detection Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }}
  .header {{ background: linear-gradient(135deg, #1a1f2e, #16213e); padding: 24px 32px; border-bottom: 1px solid #2d3748; }}
  .header h1 {{ font-size: 22px; font-weight: 700; color: #63b3ed; }}
  .header p  {{ font-size: 13px; color: #718096; margin-top: 4px; }}
  .container {{ padding: 24px 32px; max-width: 1400px; margin: 0 auto; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .kpi {{ background: #1a1f2e; border: 1px solid #2d3748; border-radius: 12px; padding: 20px; }}
  .kpi .label {{ font-size: 12px; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; }}
  .kpi .value {{ font-size: 28px; font-weight: 700; margin-top: 6px; }}
  .kpi .sub   {{ font-size: 12px; color: #718096; margin-top: 4px; }}
  .green {{ color: #68d391; }} .red {{ color: #fc8181; }} .blue {{ color: #63b3ed; }} .yellow {{ color: #fbd38d; }}
  .charts-grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-bottom: 24px; }}
  .chart-card {{ background: #1a1f2e; border: 1px solid #2d3748; border-radius: 12px; padding: 20px; }}
  .chart-card h3 {{ font-size: 14px; font-weight: 600; color: #a0aec0; margin-bottom: 16px; }}
  .alert-banner {{ background: #7b341e; border: 1px solid #c05621; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; color: #fbd38d; }}
  .ok-banner    {{ background: #1c4532; border: 1px solid #276749; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; color: #68d391; }}
  .footer {{ text-align: center; padding: 16px; color: #4a5568; font-size: 12px; }}
  canvas {{ max-height: 260px; }}
</style>
</head>
<body>
<div class="header">
  <h1>🛡️ Spam Detection — Monitoring Dashboard</h1>
  <p>Period: Last {days} days &nbsp;|&nbsp; Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</p>
</div>
<div class="container">

  {"<div class='alert-banner'>⚠️ DRIFT ALERT: Spam rate shifted by " + f"{drift_delta*100:+.1f}% between first and second half of the period. Consider retraining.</div>" if drift_alert else "<div class='ok-banner'>✓ No significant drift detected — model appears stable.</div>"}

  <div class="kpi-grid">
    <div class="kpi"><div class="label">Total Requests</div><div class="value blue">{stats.get('total_requests',0):,}</div><div class="sub">Last {days} days</div></div>
    <div class="kpi"><div class="label">Spam Detected</div><div class="value red">{stats.get('spam_count',0):,}</div><div class="sub">{stats.get('spam_rate',0)*100:.1f}% of traffic</div></div>
    <div class="kpi"><div class="label">Ham Passed</div><div class="value green">{stats.get('ham_count',0):,}</div><div class="sub">{(1-stats.get('spam_rate',0))*100:.1f}% of traffic</div></div>
    <div class="kpi"><div class="label">Avg Confidence</div><div class="value yellow">{stats.get('avg_confidence',0)*100:.1f}%</div><div class="sub">Higher = more certain</div></div>
    <div class="kpi"><div class="label">Accuracy</div><div class="value green">{acc_text}</div><div class="sub">From user feedback</div></div>
    <div class="kpi"><div class="label">Avg Response</div><div class="value blue">{stats.get('avg_response_ms',0):.0f}ms</div><div class="sub">p95: {stats.get('p95_response_ms',0):.0f}ms</div></div>
  </div>

  <div class="charts-grid">
    <div class="chart-card">
      <h3>Daily Volume — Spam vs Ham</h3>
      <canvas id="dailyChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Confidence Distribution</h3>
      <canvas id="confChart"></canvas>
    </div>
  </div>

</div>
<div class="footer">Spam Detection Monitoring · Auto-refreshes on page reload · Data from {PRED_LOG}</div>

<script>
const dailyCtx = document.getElementById('dailyChart').getContext('2d');
new Chart(dailyCtx, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(daily_dates)},
    datasets: [
      {{ label: 'Spam', data: {json.dumps(daily_spam)}, backgroundColor: 'rgba(252,129,129,0.8)', borderRadius: 4 }},
      {{ label: 'Ham',  data: {json.dumps(daily_ham)},  backgroundColor: 'rgba(104,211,145,0.8)', borderRadius: 4 }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{
      x: {{ stacked: true, ticks: {{ color: '#718096' }}, grid: {{ color: '#2d3748' }} }},
      y: {{ stacked: true, ticks: {{ color: '#718096' }}, grid: {{ color: '#2d3748' }} }}
    }},
    plugins: {{ legend: {{ labels: {{ color: '#a0aec0' }} }} }}
  }}
}});

const confCtx = document.getElementById('confChart').getContext('2d');
new Chart(confCtx, {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(conf_labels)},
    datasets: [{{ data: {json.dumps(conf_counts)},
      backgroundColor: ['#fc8181','#f6ad55','#fbd38d','#68d391','#63b3ed'],
      borderWidth: 2, borderColor: '#1a1f2e'
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'right', labels: {{ color: '#a0aec0', font: {{ size: 11 }} }} }} }}
  }}
}});
</script>
</body>
</html>"""

    with open(output, "w") as f:
        f.write(html)
    print(f"Dashboard generated → {output}")
    print(f"Open in browser:     file://{output.absolute()}")
    return output


# ── Seed demo data ─────────────────────────────────────────────────────────────

def seed_demo_logs(n: int = 200):
    """Generate realistic demo prediction logs for testing the dashboard."""
    logger = PredictionLogger()
    rng = np.random.default_rng(42)

    spam_texts = [
        "FREE ENTRY! Win cash prizes now!!!",
        "URGENT: Your account has been suspended",
        "Congratulations! You have been selected",
        "Click here to claim your reward",
        "Limited time offer - act now!",
    ]
    ham_texts = [
        "Hi, can we meet tomorrow at 3pm?",
        "Please review the attached report",
        "Thanks for the quick turnaround",
        "Let me know if you have any questions",
        "See you at the conference next week",
    ]

    for i in range(n):
        is_spam = rng.random() < 0.14   # 14% spam rate
        text    = rng.choice(spam_texts if is_spam else ham_texts)
        prob    = rng.beta(9, 1) if is_spam else rng.beta(1, 9)
        label   = "SPAM" if prob > 0.5 else "HAM"
        conf    = prob if label == "SPAM" else 1 - prob

        # Backfill timestamps over last 30 days
        ts_offset = rng.integers(0, 30 * 24 * 3600)
        fake_ts   = datetime.utcnow() - timedelta(seconds=int(ts_offset))

        record = {
            "ts"           : fake_ts.isoformat() + "Z",
            "email_id"     : f"demo-{i:04d}",
            "label"        : label,
            "spam_prob"    : round(float(prob), 4),
            "confidence"   : round(float(conf), 4),
            "model_type"   : "baseline",
            "response_ms"  : round(float(rng.normal(45, 10)), 1),
            "text_len"     : len(text),
            "correct_label": None,
            "was_correct"  : None,
        }
        with open(PRED_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")

    print(f"Seeded {n} demo prediction logs → {PRED_LOG}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitoring dashboard")
    parser.add_argument("--dashboard", action="store_true", help="Generate HTML dashboard")
    parser.add_argument("--report",    action="store_true", help="Print text report")
    parser.add_argument("--stats",     action="store_true", help="Print raw stats JSON")
    parser.add_argument("--export",    action="store_true", help="Export logs to CSV")
    parser.add_argument("--seed-demo", action="store_true", help="Seed demo data for testing")
    parser.add_argument("--days",      type=int, default=30, help="Days to analyze (default: 30)")
    args = parser.parse_args()

    if args.seed_demo:
        seed_demo_logs(200)

    if args.report:
        MonitoringStats().print_report(days=args.days)

    if args.stats:
        stats = MonitoringStats().compute(days=args.days)
        print(json.dumps(stats, indent=2, default=str))

    if args.dashboard:
        generate_dashboard(days=args.days)

    if args.export:
        df = PredictionLogger().load(days=args.days)
        out = LOG_DIR / f"predictions_export_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(out, index=False)
        print(f"Exported {len(df):,} records → {out}")

    if not any(vars(args).values()):
        parser.print_help()
