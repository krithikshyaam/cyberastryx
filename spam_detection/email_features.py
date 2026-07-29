"""
email_features.py - Production email feature extractor.

Extracts rich signals beyond just the body text:
  - Subject line (weighted heavily)
  - Sender domain analysis (free vs suspicious vs known-spam)
  - Reply-To mismatch (phishing indicator)
  - URL count and suspicious domains
  - Attachment signals (.exe, .zip)
  - Recipient count (BCC bombs)
  - ALL-CAPS ratio, exclamation density
  - Urgency keywords
  - HTML-only emails (no plain text)

Usage in n8n "Edit Fields" node:
  Pass the full Gmail payload → this extractor builds a rich text
  representation that gets sent to your model.

Usage standalone:
    from email_features import EmailFeatureExtractor
    extractor = EmailFeatureExtractor()
    features = extractor.extract(raw_email_string)
    rich_text = extractor.to_model_input(features)
"""

import re
import email
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional
from urllib.parse import urlparse
from link_image_checker import LinkImageChecker
_link_checker = LinkImageChecker(use_model=False)


# ── Known domain lists ────────────────────────────────────────────────────────

TRUSTED_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "icloud.com", "protonmail.com", "microsoft.com", "google.com",
    "apple.com", "amazon.com", "linkedin.com", "twitter.com",
    "facebook.com", "github.com", "stripe.com", "paypal.com",
}

SPAM_TLD_PATTERNS = [
    r"\.xyz$", r"\.top$", r"\.click$", r"\.loan$", r"\.win$",
    r"\.gq$", r"\.ml$", r"\.tk$", r"\.cf$", r"\.ga$",
    r"\.work$", r"\.date$", r"\.racing$",
]

URGENCY_KEYWORDS = [
    "urgent", "immediately", "act now", "limited time", "expires today",
    "last chance", "final notice", "account suspended", "verify now",
    "confirm immediately", "click here now", "respond immediately",
    "your account will be", "24 hours", "48 hours",
]

SPAM_KEYWORDS = [
    "free", "winner", "won", "prize", "claim", "congratulations",
    "selected", "lucky", "cash", "earn money", "make money",
    "work from home", "no obligation", "risk free", "guaranteed",
    "100% free", "click below", "unsubscribe", "opt-out",
    "dear friend", "dear customer", "dear user",
    "nigerian", "inheritance", "wire transfer", "western union",
    "viagra", "cialis", "pharmacy", "medication",
    "lose weight", "weight loss", "diet pill",
]

PHISHING_KEYWORDS = [
    "verify your account", "update your information", "confirm your identity",
    "suspended", "unauthorized access", "security alert", "login attempt",
    "your password", "reset your password", "account verification",
    "banking details", "credit card", "ssn", "social security",
]


# ── Feature dataclass ─────────────────────────────────────────────────────────

@dataclass
class EmailFeatures:
    # Raw fields
    subject        : str   = ""
    sender         : str   = ""
    sender_domain  : str   = ""
    reply_to       : str   = ""
    body_text      : str   = ""
    body_html      : str   = ""

    # Computed signals
    has_reply_to_mismatch : bool  = False
    is_trusted_sender     : bool  = False
    is_suspicious_tld     : bool  = False
    url_count             : int   = 0
    suspicious_url_count  : int   = 0
    recipient_count       : int   = 1
    attachment_count      : int   = 0
    has_dangerous_attach  : bool  = False   # .exe, .scr, .bat, .zip with exe

    caps_ratio            : float = 0.0     # ALL CAPS proportion
    exclamation_count     : int   = 0
    question_count        : int   = 0
    word_count            : int   = 0
    char_count            : int   = 0
    avg_word_length       : float = 0.0

    urgency_score         : int   = 0       # count of urgency phrases
    spam_keyword_score    : int   = 0       # count of spam keywords
    phishing_score        : int   = 0       # count of phishing keywords

    is_html_only          : bool  = False
    has_unsubscribe       : bool  = False
    has_tracking_pixel    : bool  = False

    # Composite spam score (0-100)
    rule_based_score      : float = 0.0
    link_spam_verdicts: list  = field(default_factory=list)  # verdicts for each URL found


# ── Extractor ─────────────────────────────────────────────────────────────────

class EmailFeatureExtractor:
    """
    Extract rich spam-detection features from a raw email.

    Supports:
      - Raw RFC 2822 email strings (from Gmail API)
      - Plain text / subject only (fallback)
    """

    def extract(self, raw: str, subject: str = "", sender: str = "") -> EmailFeatures:
        """
        Extract all features from a raw email string.

        Args:
            raw     : Full raw email (headers + body) OR just the body text
            subject : Optional subject override (if raw is just body)
            sender  : Optional sender override

        Returns:
            EmailFeatures dataclass with all signals computed
        """
        f = EmailFeatures()

        # Try parsing as RFC 2822 email
        try:
            msg = email.message_from_string(raw)
            f.subject       = subject or self._decode_header(msg.get("Subject", ""))
            f.sender        = sender  or msg.get("From", "")
            f.reply_to      = msg.get("Reply-To", "")
            recipients      = (
                (msg.get("To", "") or "") + " " +
                (msg.get("CC", "") or "") + " " +
                (msg.get("BCC", "") or "")
            )
            f.recipient_count = max(1, recipients.count("@"))
            f.body_text, f.body_html, f.attachment_count = self._parse_body(msg)
            f.has_dangerous_attach = self._check_dangerous_attachments(msg)
        except Exception:
            # Fallback: treat entire input as body text
            f.subject   = subject
            f.sender    = sender
            f.body_text = raw[:5000]

        # Sender analysis
        f.sender_domain        = self._extract_domain(f.sender)
        f.is_trusted_sender    = f.sender_domain in TRUSTED_DOMAINS
        f.is_suspicious_tld    = self._is_suspicious_tld(f.sender_domain)
        f.has_reply_to_mismatch = self._check_reply_to_mismatch(f.sender, f.reply_to)

        # URL analysis
        full_text = f"{f.subject} {f.body_text} {f.body_html}"
        urls = self._extract_urls(full_text)
        f.url_count            = len(urls)
        f.suspicious_url_count = sum(1 for u in urls if self._is_suspicious_url(u))
        # Deep-scan each URL with link_image_checker
        for url in urls[:5]:   # cap at 5 to stay fast
            try:
                r = _link_checker.check_url(url)
                if r.verdict in ("SPAM", "PHISHING", "SUSPICIOUS"):
                    f.link_spam_verdicts.append({"url": url, "verdict": r.verdict,
                                                 "score": r.risk_score})
            except Exception:
                pass
        # Text statistics (on plain text body + subject)
        combined = f"{f.subject} {f.body_text}".strip()
        f.word_count        = len(combined.split())
        f.char_count        = len(combined)
        f.caps_ratio        = self._caps_ratio(combined)
        f.exclamation_count = combined.count("!")
        f.question_count    = combined.count("?")
        words = combined.split()
        f.avg_word_length   = sum(len(w) for w in words) / max(1, len(words))

        # Keyword scores
        combined_lower = combined.lower()
        f.urgency_score      = sum(1 for kw in URGENCY_KEYWORDS  if kw in combined_lower)
        f.spam_keyword_score = sum(1 for kw in SPAM_KEYWORDS     if kw in combined_lower)
        f.phishing_score     = sum(1 for kw in PHISHING_KEYWORDS if kw in combined_lower)

        # Other signals
        f.has_unsubscribe    = "unsubscribe" in combined_lower or "opt-out" in combined_lower
        f.has_tracking_pixel = bool(re.search(r'<img[^>]+width=["\']?1["\']?', f.body_html, re.I))
        f.is_html_only       = bool(f.body_html) and not bool(f.body_text.strip())

        # Composite rule-based score
        f.rule_based_score   = self._compute_rule_score(f)

        return f

    # ── Model input builder ───────────────────────────────────────────────────

    def to_model_input(self, f: EmailFeatures, include_metadata: bool = True) -> str:
        """
        Convert extracted features into a rich text string for the model.

        Combines subject + body + structured metadata tokens so the model
        sees all signals, not just the body.

        Example output:
            [SUBJECT] Win a prize now! [SENDER_RISK: suspicious] [URL_COUNT: 3]
            [SUSPICIOUS_URLS: 2] [CAPS_HIGH] [URGENCY: 3] [SPAM_KW: 5]
            Click here to claim your FREE prize...
        """
        parts = []

        # Subject (very high signal — prefix it)
        if f.subject:
            parts.append(f"[SUBJECT] {f.subject}")

        # Sender signals
        if f.is_suspicious_tld:
            parts.append("[SENDER_RISK: suspicious_tld]")
        elif not f.is_trusted_sender and f.sender_domain:
            parts.append(f"[SENDER_RISK: unknown_domain]")
        else:
            parts.append("[SENDER_RISK: trusted]")

        if f.has_reply_to_mismatch:
            parts.append("[REPLY_TO_MISMATCH]")

        # URL signals
        if f.url_count > 0:
            parts.append(f"[URL_COUNT: {f.url_count}]")
        if f.suspicious_url_count > 0:
            parts.append(f"[SUSPICIOUS_URLS: {f.suspicious_url_count}]")
        if f.link_spam_verdicts:
            worst = max(f.link_spam_verdicts, key=lambda x: x["score"])
            parts.append(f"[LINK_{worst['verdict']}: score={worst['score']}]")

        # Text signals
        if f.caps_ratio > 0.3:
            parts.append("[CAPS_HIGH]")
        if f.exclamation_count > 3:
            parts.append(f"[EXCLAMATIONS: {f.exclamation_count}]")
        if f.urgency_score > 0:
            parts.append(f"[URGENCY: {f.urgency_score}]")
        if f.spam_keyword_score > 0:
            parts.append(f"[SPAM_KW: {f.spam_keyword_score}]")
        if f.phishing_score > 0:
            parts.append(f"[PHISHING_KW: {f.phishing_score}]")

        # Structural signals
        if f.has_dangerous_attach:
            parts.append("[DANGEROUS_ATTACHMENT]")
        if f.recipient_count > 10:
            parts.append(f"[MASS_RECIPIENT: {f.recipient_count}]")
        if f.is_html_only:
            parts.append("[HTML_ONLY]")
        if f.has_unsubscribe:
            parts.append("[HAS_UNSUBSCRIBE]")

        # Rule score bucket
        score = f.rule_based_score
        if score >= 70:
            parts.append("[RULE_SCORE: very_high]")
        elif score >= 40:
            parts.append("[RULE_SCORE: high]")
        elif score >= 20:
            parts.append("[RULE_SCORE: medium]")

        # Body text
        body = (f.body_text or f.body_html)[:1500].strip()
        if body:
            parts.append(body)

        return " ".join(parts)

    def to_dict(self, f: EmailFeatures) -> dict:
        """Return features as a plain dict (for logging/storage)."""
        return asdict(f)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _decode_header(self, value: str) -> str:
        from email.header import decode_header
        parts = []
        for raw, enc in decode_header(value or ""):
            if isinstance(raw, bytes):
                parts.append(raw.decode(enc or "utf-8", errors="replace"))
            else:
                parts.append(str(raw))
        return " ".join(parts)

    def _parse_body(self, msg) -> tuple:
        """Returns (plain_text, html_text, attachment_count)."""
        plain = html = ""
        attachments = 0
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                disp = str(part.get("Content-Disposition", ""))
                if "attachment" in disp:
                    attachments += 1
                    continue
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                decoded = payload.decode("utf-8", errors="replace")
                if ct == "text/plain" and not plain:
                    plain = decoded
                elif ct == "text/html" and not html:
                    html = re.sub(r"<[^>]+>", " ", decoded)  # strip HTML tags
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                ct = msg.get_content_type()
                decoded = payload.decode("utf-8", errors="replace")
                if ct == "text/html":
                    html = re.sub(r"<[^>]+>", " ", decoded)
                else:
                    plain = decoded
        return plain[:3000], html[:3000], attachments

    def _check_dangerous_attachments(self, msg) -> bool:
        dangerous_exts = {".exe", ".scr", ".bat", ".cmd", ".pif", ".com", ".vbs", ".js"}
        for part in msg.walk():
            fname = part.get_filename() or ""
            ext = "." + fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            if ext in dangerous_exts:
                return True
        return False

    def _extract_domain(self, sender: str) -> str:
        match = re.search(r"@([\w.\-]+)", sender)
        return match.group(1).lower() if match else ""

    def _is_suspicious_tld(self, domain: str) -> bool:
        return any(re.search(pat, domain) for pat in SPAM_TLD_PATTERNS)

    def _check_reply_to_mismatch(self, sender: str, reply_to: str) -> bool:
        if not reply_to:
            return False
        sender_domain  = self._extract_domain(sender)
        reply_domain   = self._extract_domain(reply_to)
        return bool(sender_domain and reply_domain and sender_domain != reply_domain)

    def _extract_urls(self, text: str) -> list:
        return re.findall(r"https?://\S+|www\.\S+", text)

    def _is_suspicious_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url if url.startswith("http") else "http://" + url)
            domain = parsed.netloc.lower()
            # IP address URLs are suspicious
            if re.match(r"\d+\.\d+\.\d+\.\d+", domain):
                return True
            # Suspicious TLD
            if self._is_suspicious_tld(domain):
                return True
            # Very long URLs (obfuscation)
            if len(url) > 200:
                return True
            # URL shorteners
            shorteners = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd"}
            if any(s in domain for s in shorteners):
                return True
        except Exception:
            pass
        return False

    def _caps_ratio(self, text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        return sum(1 for c in letters if c.isupper()) / len(letters)

    def _compute_rule_score(self, f: EmailFeatures) -> float:
        """
        Heuristic rule-based spam score (0-100).
        Higher = more likely spam.
        Used as a feature signal — not the final decision.
        """
        score = 0.0

        # Sender signals (strong)
        if f.is_suspicious_tld:        score += 25
        if f.has_reply_to_mismatch:    score += 20
        if not f.is_trusted_sender:    score += 5

        # Content signals
        score += min(f.spam_keyword_score  * 4,  20)
        score += min(f.urgency_score       * 5,  15)
        score += min(f.phishing_score      * 6,  18)

        # URL signals
        score += min(f.suspicious_url_count * 8, 20)
        if f.url_count > 5:            score += 5

        # Text signals
        if f.caps_ratio > 0.5:         score += 10
        if f.caps_ratio > 0.3:         score += 5
        if f.exclamation_count > 5:    score += 5
        if f.exclamation_count > 10:   score += 5

        # Structural signals
        if f.has_dangerous_attach:     score += 30
        if f.recipient_count > 50:     score += 15
        if f.recipient_count > 10:     score += 8
        if f.is_html_only:             score += 5
        if f.has_tracking_pixel:       score += 5

        return min(score, 100.0)
        # Link checker results
        for v in f.link_spam_verdicts:
            if v["verdict"] == "PHISHING": score += 30
            elif v["verdict"] == "SPAM":   score += 20
            elif v["verdict"] == "SUSPICIOUS": score += 10


# ── n8n Integration helper ────────────────────────────────────────────────────

def process_gmail_payload(gmail_json: dict) -> str:
    """
    Process a Gmail API message payload into model input.

    Args:
        gmail_json: The JSON payload from the n8n "Get a message" node

    Returns:
        Rich text string ready to send to your spam detection model
    """
    extractor = EmailFeatureExtractor()

    # Extract from Gmail JSON structure
    headers = {h["name"]: h["value"] for h in gmail_json.get("payload", {}).get("headers", [])}

    subject = headers.get("Subject", "")
    sender  = headers.get("From", "")

    # Get body
    body = ""
    payload = gmail_json.get("payload", {})
    parts = payload.get("parts", [])
    if parts:
        for part in parts:
            if part.get("mimeType") == "text/plain":
                import base64
                data = part.get("body", {}).get("data", "")
                if data:
                    body = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
                    break
    else:
        import base64
        data = payload.get("body", {}).get("data", "")
        if data:
            body = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")

    # Build a synthetic RFC email for parsing
    raw = f"From: {sender}\nSubject: {subject}\n\n{body}"

    features = extractor.extract(raw)
    return extractor.to_model_input(features)


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    extractor = EmailFeatureExtractor()

    test_emails = [
        {
            "name": "Phishing email",
            "raw": """From: security@paypa1-alerts.xyz
Reply-To: collect@differentdomain.tk
Subject: URGENT: Your PayPal account has been SUSPENDED!!!
To: undisclosed-recipients

Dear Customer,

Your PayPal account has been SUSPENDED due to unauthorized access!!
You MUST verify your identity IMMEDIATELY or your account will be closed in 24 HOURS!

Click here NOW to verify: http://bit.ly/paypal-verify-acc-2024

ACT NOW - LIMITED TIME ONLY!!!

PayPal Security Team
"""
        },
        {
            "name": "Legitimate email",
            "raw": """From: alice@gmail.com
Subject: Re: Meeting tomorrow at 3pm
To: bob@gmail.com

Hi Bob,

Just confirming our meeting tomorrow at 3pm in conference room B.
Could you bring the Q3 report?

Thanks,
Alice
"""
        },
    ]

    for test in test_emails:
        print(f"\n{'='*60}")
        print(f"  {test['name']}")
        print(f"{'='*60}")
        features = extractor.extract(test["raw"])
        model_input = extractor.to_model_input(features)
        print(f"Rule-based score : {features.rule_based_score:.1f}/100")
        print(f"Spam keywords    : {features.spam_keyword_score}")
        print(f"Phishing keywords: {features.phishing_score}")
        print(f"Suspicious TLD   : {features.is_suspicious_tld}")
        print(f"Reply-To mismatch: {features.has_reply_to_mismatch}")
        print(f"\nModel input:\n{model_input[:300]}")
