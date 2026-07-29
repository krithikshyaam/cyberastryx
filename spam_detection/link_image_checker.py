"""
link_image_checker.py — URL and image spam/phishing analysis

Extends the spam detection pipeline with:
  [L1] URL structure analysis  (suspicious TLD, entropy, shorteners, redirects)
  [L2] Page content fetching   (extract visible text → run spam classifier)
  [L3] Phishing detection      (brand mentioned on page ≠ domain in URL)
  [I1] Image OCR               (extract text → run spam classifier)
  [I2] Visual spam signals     (color palette, layout urgency cues)

Usage (CLI):
    # Check a URL
    python link_image_checker.py --url "http://bit.ly/free-prize-claim"

    # Check an image file
    python link_image_checker.py --image path/to/banner.png

    # Check an image hosted at a URL
    python link_image_checker.py --image "https://example.com/ad.jpg"

    # Skip loading the ML model (structural analysis only)
    python link_image_checker.py --url "http://suspicious.xyz" --no-model

Usage (Python API):
    from link_image_checker import LinkImageChecker
    checker = LinkImageChecker()

    result = checker.check_url("http://bit.ly/win-iphone-now")
    result = checker.check_image("banner.png")

    print(result["verdict"])      # PHISHING / SPAM / SUSPICIOUS / CLEAN
    print(result["risk_score"])   # 0-100
    print(result["signals"])      # list of triggered signals with weights
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import socket
import time
import urllib.parse
import warnings
from dataclasses import dataclass, field
from typing import Optional

# Suppress urllib3/chardet version mismatch warning (harmless)
warnings.filterwarnings("ignore", category=Warning, module="requests")
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*chardet.*")
warnings.filterwarnings("ignore", message=".*charset_normalizer.*")

import requests
from bs4 import BeautifulSoup
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REQUEST_TIMEOUT  = 8          # seconds for HTTP requests
MAX_PAGE_CHARS   = 8_000      # chars of page text sent to spam classifier
MAX_IMAGE_SIDE   = 2400       # pixels — cap before visual analysis
OCR_TARGET_WIDTH = 2000       # upscale small images to this width for OCR
MIN_OCR_CONF     = 55         # minimum Tesseract confidence to count a word

# Known URL shorteners
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "tiny.cc", "rb.gy", "cutt.ly", "shorturl.at",
    "rebrand.ly", "bl.ink", "short.io", "clickmeter.com", "su.pr",
}

# Suspicious TLDs commonly abused in spam / phishing
SUSPICIOUS_TLDS = {
    ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".work", ".click",
    ".loan", ".win", ".racing", ".review", ".date", ".faith", ".science",
    ".cricket", ".accountant", ".party", ".stream", ".download", ".bid",
    ".trade", ".webcam", ".country", ".kim", ".men", ".rest",
}

# Brands often impersonated in phishing — maps brand name → legitimate domains
BRAND_DOMAINS: dict[str, set[str]] = {
    "paypal":     {"paypal.com", "paypal.co.uk"},
    "amazon":     {"amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr", "amazon.in"},
    "apple":      {"apple.com", "icloud.com"},
    "google":     {"google.com", "accounts.google.com", "gmail.com"},
    "microsoft":  {"microsoft.com", "live.com", "outlook.com", "office.com", "office365.com"},
    "netflix":    {"netflix.com"},
    "facebook":   {"facebook.com", "fb.com", "meta.com"},
    "instagram":  {"instagram.com"},
    "twitter":    {"twitter.com", "x.com"},
    "ebay":       {"ebay.com", "ebay.co.uk"},
    "dhl":        {"dhl.com", "dhl.de", "dhl.co.uk"},
    "fedex":      {"fedex.com"},
    "ups":        {"ups.com"},
    "chase":      {"chase.com"},
    "wellsfargo": {"wellsfargo.com"},
    "bankofamerica": {"bankofamerica.com"},
    "irs":        {"irs.gov"},
    "usps":       {"usps.com"},
}

# Phishing keyword patterns found on fake pages
PHISHING_PAGE_PATTERNS = [
    r"verify\s+your\s+account",
    r"confirm\s+your\s+(identity|details|information|password)",
    r"your\s+account\s+(has been|was|is)\s+(suspended|locked|limited|compromised)",
    r"unusual\s+(activity|sign.?in|login)",
    r"click\s+here\s+to\s+(verify|confirm|update|restore)",
    r"enter\s+your\s+(credit card|card number|ssn|social security|bank)",
    r"update\s+(payment|billing|account)\s+(information|details|method)",
    r"your\s+password\s+(has expired|will expire|needs to be updated)",
]

# Visual spam signals — keywords OCR'd from images
VISUAL_SPAM_KEYWORDS = [
    # Prize / lottery
    "winner", "won", "winning", "congratulations", "prize", "prizes",
    "lottery", "jackpot", "promotion", "selected", "chosen",
    # Action / urgency
    "claim", "act now", "limited time", "urgent", "expires", "today only",
    "last chance", "hurry", "immediately", "don't miss",
    # Money / reward
    "free", "reward", "cash", "gift card", "million", "thousand",
    "guaranteed", "paid", "earn", "bonus", "offer", "exclusive",
    # Phishing / credential
    "click here", "call now", "verify", "confirm", "update",
    "suspended", "locked", "access", "login", "password",
    # General scam markers
    "risk free", "no obligation", "instant", "100%", "participate",
]

# Regex patterns for prize amounts in OCR text (e.g. €574,147 or $1,000,000)
PRIZE_AMOUNT_RE = re.compile(
    r"(€|\$|£|USD|EUR)\s?[\d,\.]{4,}"   # currency + big number
    r"|[\d,\.]{5,}\s?(€|\$|£|USD|EUR)"  # big number + currency
)

# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    input: str
    input_type: str                      # "url" | "image"
    verdict: str = "UNKNOWN"             # CLEAN / SUSPICIOUS / SPAM / PHISHING
    risk_score: int = 0                  # 0-100
    signals: list[dict] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    classifier_result: Optional[dict] = None
    error: Optional[str] = None

    def add_signal(self, name: str, weight: int, description: str):
        self.signals.append({"signal": name, "weight": weight, "description": description})
        self.risk_score = min(100, self.risk_score + weight)

    def to_dict(self) -> dict:
        return {
            "input":       self.input,
            "type":        self.input_type,
            "verdict":     self.verdict,
            "risk_score":  self.risk_score,
            "signals":     self.signals,
            "details":     self.details,
            "classifier":  self.classifier_result,
            "error":       self.error,
        }

    def finalize_verdict(self):
        if self.error and not self.signals:
            self.verdict = "ERROR"
            return

        names = {s["signal"] for s in self.signals}

        # Hard escalation combos — these combinations are unambiguously malicious
        # regardless of total score
        hard_spam_combos = [
            # Dead site on a high-risk TLD with a scam word in the name
            {"suspicious_tld", "domain_unresolvable", "spam_word_in_domain"},
            # Brand impersonation
            {"phishing_brand_mismatch"},
            {"login_form_brand_mismatch"},
            # Brand in subdomain on a bad TLD
            {"brand_in_subdomain", "suspicious_tld"},
        ]
        for combo in hard_spam_combos:
            if combo.issubset(names):
                escalated = combo == {"phishing_brand_mismatch"} or \
                            combo == {"login_form_brand_mismatch"}
                self.verdict = "PHISHING" if escalated else "SPAM"
                return

        if self.risk_score >= 55:      # lowered from 70
            self.verdict = "SPAM"
        elif self.risk_score >= 25:
            self.verdict = "SUSPICIOUS"
        else:
            self.verdict = "CLEAN"


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model loader — avoids importing TF/Keras unless needed
# ─────────────────────────────────────────────────────────────────────────────

class _LazyPredictor:
    """Loads SpamPredictor only on first use."""

    def __init__(self, model_type: str = "baseline"):
        self._model_type = model_type
        self._predictor = None

    def _load(self):
        if self._predictor is None:
            from src.predict import SpamPredictor
            self._predictor = SpamPredictor(model_type=self._model_type)

    def predict(self, text: str) -> dict:
        self._load()
        return self._predictor.predict(text)

    def available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# URL Checker
# ─────────────────────────────────────────────────────────────────────────────

class URLChecker:
    """
    [L1] Structural URL analysis
    [L2] Page content → spam classifier
    [L3] Phishing brand-mismatch detection
    """

    def __init__(self, predictor: Optional[_LazyPredictor] = None):
        self.predictor = predictor
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

    # ── [L1] URL Structure ────────────────────────────────────────────────────

    def analyze_structure(self, url: str) -> tuple[list[dict], dict]:
        """
        Inspect the URL itself without fetching it.
        Returns (signals_list, details_dict).
        """
        signals = []
        details: dict = {}

        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower().lstrip("www.")
        path   = parsed.path
        query  = parsed.query

        # Store parsed info in details
        details["domain"]   = domain
        details["scheme"]   = parsed.scheme
        details["path"]     = path[:200]
        details["tld"]      = "." + domain.split(".")[-1] if "." in domain else ""

        # HTTP (no TLS)
        if parsed.scheme == "http":
            signals.append({"signal": "no_https", "weight": 10,
                            "description": "URL uses HTTP instead of HTTPS"})

        # Suspicious TLD — tiered weights based on abuse rate
        tld = details["tld"]
        HIGH_RISK_TLDS   = {".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".click",
                            ".win", ".loan", ".racing", ".bid", ".download", ".webcam"}
        MEDIUM_RISK_TLDS = {".work", ".review", ".date", ".faith", ".science", ".cricket",
                            ".accountant", ".party", ".stream", ".trade", ".country",
                            ".kim", ".men", ".rest"}
        if tld in HIGH_RISK_TLDS:
            signals.append({"signal": "suspicious_tld", "weight": 25,
                            "description": f"TLD '{tld}' is high-risk — heavily abused in spam/phishing"})
        elif tld in MEDIUM_RISK_TLDS:
            signals.append({"signal": "suspicious_tld", "weight": 15,
                            "description": f"TLD '{tld}' is commonly used in spam/phishing"})

        # Known URL shortener
        base_domain = ".".join(domain.split(".")[-2:])
        if base_domain in URL_SHORTENERS or domain in URL_SHORTENERS:
            signals.append({"signal": "url_shortener", "weight": 15,
                            "description": f"'{domain}' is a URL shortener — hides true destination"})
            details["is_shortener"] = True

        # Excessive subdomains (e.g. paypal.com.verify.evil.xyz)
        parts = domain.split(".")
        if len(parts) > 4:
            signals.append({"signal": "excessive_subdomains", "weight": 15,
                            "description": f"Domain has {len(parts)} labels — common in phishing URLs"})

        # Brand name in subdomain (paypal.com.evildomain.xyz)
        for brand in BRAND_DOMAINS:
            if brand in domain and not any(domain.endswith(d) for d in BRAND_DOMAINS[brand]):
                signals.append({"signal": "brand_in_subdomain", "weight": 30,
                                "description": f"Brand '{brand}' appears in URL but domain is not '{brand}.com'"})
                details["impersonated_brand"] = brand
                break

        # Very long URL
        url_len = len(url)
        details["url_length"] = url_len
        if url_len > 200:
            signals.append({"signal": "long_url", "weight": 8,
                            "description": f"URL is {url_len} chars long — legitimate URLs are usually shorter"})

        # High character entropy in domain (randomised-looking)
        entropy = _shannon_entropy(domain.split(".")[0])
        details["domain_entropy"] = round(entropy, 2)
        if entropy > 3.8:
            signals.append({"signal": "high_entropy_domain", "weight": 12,
                            "description": f"Domain has high randomness (entropy={entropy:.2f}) — may be auto-generated"})

        # Lots of query parameters (tracking/redirect chains)
        params = urllib.parse.parse_qs(query)
        details["query_param_count"] = len(params)
        if len(params) > 8:
            signals.append({"signal": "many_query_params", "weight": 5,
                            "description": f"URL has {len(params)} query params — may indicate redirect chain"})

        # IP address instead of domain
        if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain.split(":")[0]):
            signals.append({"signal": "ip_address_url", "weight": 25,
                            "description": "URL uses raw IP address instead of a domain name"})

        # Suspicious keywords in path
        suspicious_path_words = ["login", "signin", "verify", "secure", "account", "update",
                                 "confirm", "password", "credential", "banking", "paypal",
                                 "amazon", "microsoft", "apple", "google"]
        found_path_words = [w for w in suspicious_path_words if w in path.lower()]
        if found_path_words:
            signals.append({"signal": "suspicious_path_keywords", "weight": 10,
                            "description": f"Path contains: {', '.join(found_path_words)}"})

        # Spam/scam words embedded in the domain name itself (e.g. priwin, freegift, claimprize)
        DOMAIN_SPAM_WORDS = [
            "win", "prize", "promo", "free", "gift", "claim", "reward",
            "cash", "lucky", "bonus", "offer", "deal", "earn", "money",
            "lottery", "winner", "jackpot", "congrat", "selected",
        ]
        domain_name = domain.split(".")[0].lower()   # just the SLD e.g. "priwin"
        found_domain_words = [w for w in DOMAIN_SPAM_WORDS if w in domain_name]
        if found_domain_words:
            signals.append({
                "signal":      "spam_word_in_domain",
                "weight":      15,
                "description": (
                    f"Domain '{domain_name}' contains spam word(s): "
                    f"{', '.join(found_domain_words)}"
                ),
            })

        return signals, details

    # ── Redirect resolution ────────────────────────────────────────────────────

    def resolve_redirects(self, url: str) -> tuple[str, list[str], dict]:
        """
        Follow redirects and return (final_url, redirect_chain, details).
        """
        chain = [url]
        details: dict = {}
        try:
            resp = self._session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True,
                                     stream=True)
            resp.close()
            final_url = resp.url
            # requests stores redirect history
            for r in resp.history:
                chain.append(r.headers.get("Location", r.url))
            chain.append(final_url)
            chain = list(dict.fromkeys(chain))   # deduplicate while preserving order
            details["status_code"]  = resp.status_code
            details["redirect_count"] = len(resp.history)
            details["final_url"]    = final_url
        except requests.exceptions.Timeout:
            details["error"] = "timeout"
            final_url = url
        except Exception as e:
            details["error"] = str(e)[:120]
            final_url = url
        return final_url, chain, details

    # ── [L2] Page content + spam classifier ───────────────────────────────────

    def fetch_and_classify(self, url: str) -> tuple[list[dict], dict]:
        """
        Fetch page, extract visible text, run spam classifier.
        Returns (signals, details).
        """
        signals = []
        details: dict = {}

        try:
            resp = self._session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            err = str(e)
            details["fetch_error"] = err[:120]
            # DNS failure on a suspicious TLD = extra signal (scam sites often go dark)
            if "NameResolutionError" in err or "getaddrinfo" in err.lower():
                details["fetch_note"] = "Domain did not resolve — site may be down or already taken offline"
                signals.append({
                    "signal":      "domain_unresolvable",
                    "weight":      10,
                    "description": "Domain failed DNS resolution — common for scam sites taken offline",
                })
            return signals, details
        except Exception as e:
            details["fetch_error"] = str(e)[:120]
            return signals, details

        ct = resp.headers.get("content-type", "")
        if "text/html" not in ct and "text/plain" not in ct:
            details["content_type"] = ct
            details["note"] = "Non-HTML content — skipping text extraction"
            return signals, details

        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script / style / nav / footer noise
        for tag in soup(["script", "style", "nav", "footer", "noscript"]):
            tag.decompose()

        visible_text = " ".join(soup.get_text(separator=" ").split())
        details["page_text_chars"] = len(visible_text)
        details["page_title"]      = (soup.title.string or "").strip()[:200] if soup.title else ""

        # Count forms + password fields (phishing indicator)
        forms     = soup.find_all("form")
        pw_inputs = soup.find_all("input", {"type": "password"})
        details["form_count"]     = len(forms)
        details["password_inputs"] = len(pw_inputs)

        # Run spam classifier on page text
        if self.predictor and self.predictor.available():
            snippet = visible_text[:MAX_PAGE_CHARS]
            clf = self.predictor.predict(snippet)
            details["classifier"] = clf
            if clf["label"] == "SPAM":
                signals.append({
                    "signal": "page_content_spam",
                    "weight": 25,
                    "description": (
                        f"Spam classifier flagged page text as SPAM "
                        f"(confidence {clf['confidence']*100:.0f}%)"
                    ),
                })
        else:
            text_lower = visible_text.lower()

            # Phishing patterns (credential-harvesting pages)
            phish_hits = [p for p in PHISHING_PAGE_PATTERNS if re.search(p, text_lower, re.I)]
            details["phishing_pattern_hits"] = phish_hits
            if phish_hits:
                signals.append({
                    "signal":      "phishing_page_patterns",
                    "weight":      20 + min(20, len(phish_hits) * 5),
                    "description": (
                        f"Page contains {len(phish_hits)} phishing pattern(s): "
                        f"{phish_hits[0][:60]}…"
                    ),
                })

            # Prize / scam keywords (lottery, prize, free offer pages)
            spam_hits = [kw for kw in VISUAL_SPAM_KEYWORDS if kw in text_lower]
            details["page_spam_keyword_hits"] = spam_hits
            if len(spam_hits) >= 3:   # require ≥3 hits — avoids false-positives on retail sites
                signals.append({
                    "signal":      "page_content_spam_keywords",
                    "weight":      min(30, len(spam_hits) * 4),
                    "description": (
                        f"Page text contains {len(spam_hits)} spam keyword(s): "
                        f"{', '.join(spam_hits[:6])}"
                    ),
                })

            # Prize amount patterns (€574,147 / $1,000,000)
            prize_hits = PRIZE_AMOUNT_RE.findall(visible_text)
            if prize_hits:
                flat = ["".join(m) for m in prize_hits]
                details["page_prize_amounts"] = flat[:3]
                signals.append({
                    "signal":      "page_prize_amount",
                    "weight":      20,
                    "description": (
                        f"Large monetary amounts found in page: {', '.join(flat[:3])}"
                    ),
                })

        return signals, details

    # ── [L3] Phishing brand-mismatch ─────────────────────────────────────────

    def detect_phishing(self, final_url: str, page_details: dict) -> list[dict]:
        """
        Compare brand mentions on page against the actual domain.
        Returns signal list.
        """
        signals = []
        parsed = urllib.parse.urlparse(final_url)
        actual_domain = parsed.netloc.lower().lstrip("www.")

        title = (page_details.get("page_title") or "").lower()
        text_sample = str(page_details.get("classifier", {}).get("label", "")).lower()

        for brand, legit_domains in BRAND_DOMAINS.items():
            # Is this brand mentioned in the page title?
            if brand in title:
                is_legit = any(actual_domain.endswith(d) for d in legit_domains)
                if not is_legit:
                    signals.append({
                        "signal": "phishing_brand_mismatch",
                        "weight": 45,
                        "description": (
                            f"Page title mentions '{brand}' but domain is '{actual_domain}' "
                            f"(expected: {', '.join(legit_domains)})"
                        ),
                    })

        # Login form on non-brand domain
        if page_details.get("password_inputs", 0) > 0:
            # Check if any brand keyword appears in the URL path
            url_lower = final_url.lower()
            for brand, legit_domains in BRAND_DOMAINS.items():
                if brand in url_lower and not any(actual_domain.endswith(d) for d in legit_domains):
                    signals.append({
                        "signal": "login_form_brand_mismatch",
                        "weight": 40,
                        "description": (
                            f"Login form found on '{actual_domain}' but URL path references '{brand}'"
                        ),
                    })
                    break

        return signals

    # ── Full URL check ────────────────────────────────────────────────────────

    def check(self, url: str) -> CheckResult:
        result = CheckResult(input=url, input_type="url")

        # [L1] Structure
        struct_signals, struct_details = self.analyze_structure(url)
        for s in struct_signals:
            result.add_signal(s["signal"], s["weight"], s["description"])
        result.details.update(struct_details)

        # Resolve redirects
        final_url, chain, redirect_details = self.resolve_redirects(url)
        result.details["redirect_chain"] = chain
        result.details.update(redirect_details)

        if len(chain) > 2:
            result.add_signal("redirect_chain", min(15, (len(chain) - 1) * 5),
                              f"URL redirects {len(chain)-1} time(s) → {chain[-1][:80]}")

        # If the final URL differs from input, re-analyse its structure
        if final_url != url:
            final_signals, final_details = self.analyze_structure(final_url)
            for s in final_signals:
                label = s["signal"] + "_final"
                result.add_signal(label, s["weight"],
                                  "[After redirect] " + s["description"])
            result.details["final_url_details"] = final_details

        # [L2] Page content
        page_signals, page_details = self.fetch_and_classify(final_url)
        for s in page_signals:
            result.add_signal(s["signal"], s["weight"], s["description"])
        result.details["page"] = page_details
        if page_details.get("classifier"):
            result.classifier_result = page_details["classifier"]

        # [L3] Phishing
        phishing_signals = self.detect_phishing(final_url, page_details)
        for s in phishing_signals:
            result.add_signal(s["signal"], s["weight"], s["description"])

        result.finalize_verdict()
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Image Checker
# ─────────────────────────────────────────────────────────────────────────────

class ImageChecker:
    """
    [I1] OCR extracted text → spam classifier
    [I2] Visual spam signals (color palette, layout urgency cues)
    """

    def __init__(self, predictor: Optional[_LazyPredictor] = None):
        self.predictor = predictor

    # ── Load image from path or URL ───────────────────────────────────────────

    def _load_image(self, source: str) -> Image.Image:
        if source.startswith("http://") or source.startswith("https://"):
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                "Referer": "https://www.google.com/",
            }
            resp = requests.get(source, timeout=REQUEST_TIMEOUT, headers=headers)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        else:
            img = Image.open(source).convert("RGB")
        # Only cap extremely large images to avoid memory issues
        w, h = img.size
        if max(w, h) > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img

    # ── OCR preprocessing pipeline ────────────────────────────────────────────

    def _preprocess_for_ocr(self, img: Image.Image) -> list[Image.Image]:
        """
        Return a list of differently-preprocessed versions of the image.
        Tesseract is run on all of them; results are merged for best coverage.

        Pipeline per variant:
          v1 — upscale + sharpen + contrast boost          (good for coloured banners)
          v2 — grayscale + Otsu threshold                  (good for clean text on solid bg)
          v3 — grayscale + adaptive threshold (small zones) (good for varied backgrounds)
          v4 — inverted grayscale + Otsu                   (catches light text on dark bg)
        """
        from PIL import ImageFilter, ImageEnhance, ImageOps
        import numpy as np

        # ── auto-crop browser chrome ──────────────────────────────────────────
        # If the image is a browser screenshot, the actual content starts after
        # the toolbar (~10% from top). Crop the top 12% to remove URL bars etc.
        w, h = img.size
        ar   = w / h
        if 0.9 < ar < 2.5 and h > 300:   # looks like a browser window
            top_crop = int(h * 0.12)
            img = img.crop((0, top_crop, w, h))

        # ── upscale small / medium images for better OCR resolution ──────────
        w, h = img.size
        if w < OCR_TARGET_WIDTH:
            scale = OCR_TARGET_WIDTH / w
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        variants: list[Image.Image] = []

        # v1: colour → sharpen + contrast
        v1 = img.copy()
        v1 = ImageEnhance.Sharpness(v1).enhance(2.5)
        v1 = ImageEnhance.Contrast(v1).enhance(1.8)
        variants.append(v1)

        # v2: grayscale → Otsu global threshold
        gray = img.convert("L")
        arr  = np.array(gray)
        # Otsu threshold
        hist, bins = np.histogram(arr.flatten(), 256, [0, 256])
        total = arr.size
        sum_all = np.dot(np.arange(256), hist)
        sum_b = wb = 0.0
        best_var = best_t = 0
        for t in range(256):
            wb += hist[t]
            if wb == 0:
                continue
            wf = total - wb
            if wf == 0:
                break
            sum_b += t * hist[t]
            mb = sum_b / wb
            mf = (sum_all - sum_b) / wf
            var = wb * wf * (mb - mf) ** 2
            if var > best_var:
                best_var, best_t = var, t
        v2 = gray.point(lambda p: 255 if p > best_t else 0)
        variants.append(v2)

        # v3: grayscale → adaptive local threshold (16×16 tiles)
        from PIL import ImageFilter
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=15))
        arr_local = np.array(gray).astype(int) - np.array(blurred).astype(int)
        arr_local = np.clip(arr_local + 128, 0, 255).astype(np.uint8)
        v3 = Image.fromarray(arr_local).point(lambda p: 255 if p > 128 else 0)
        variants.append(v3)

        # v4: inverted grayscale + Otsu (light text on dark background)
        v4 = ImageOps.invert(gray)
        arr4 = np.array(v4)
        hist4, _ = np.histogram(arr4.flatten(), 256, [0, 256])
        sum_all4 = np.dot(np.arange(256), hist4)
        sum_b4 = wb4 = best_var4 = best_t4 = 0
        for t in range(256):
            wb4 += hist4[t]
            if wb4 == 0:
                continue
            wf4 = total - wb4
            if wf4 == 0:
                break
            sum_b4 += t * hist4[t]
            mb4 = sum_b4 / wb4
            mf4 = (sum_all4 - sum_b4) / wf4
            var4 = wb4 * wf4 * (mb4 - mf4) ** 2
            if var4 > best_var4:
                best_var4, best_t4 = var4, t
        v4 = v4.point(lambda p: 255 if p > best_t4 else 0)
        variants.append(v4)

        return variants

    def _merge_ocr_texts(self, texts: list[str]) -> str:
        """
        Merge OCR results from multiple preprocessing variants.
        Pick the variant with most high-confidence words, then append
        any unique words found exclusively by other variants.
        """
        # Normalise: lowercase, collapse whitespace, strip junk chars
        cleaned = []
        for t in texts:
            t = re.sub(r"[^\w\s€$£%.,!?@/-]", " ", t)
            t = re.sub(r"\s+", " ", t).strip().lower()
            cleaned.append(t)

        # Longest clean text wins as the base
        base = max(cleaned, key=len)

        # Add any tokens from other variants not already in base
        base_tokens = set(base.split())
        extras = []
        for t in cleaned:
            for tok in t.split():
                if len(tok) > 3 and tok not in base_tokens:
                    extras.append(tok)
                    base_tokens.add(tok)

        return (base + " " + " ".join(extras)).strip()

    def ocr_and_classify(self, img: Image.Image) -> tuple[list[dict], dict]:
        """
        Multi-strategy OCR:
          1. Build 4 preprocessed variants of the image
          2. Run Tesseract on each with PSM 6 (uniform block) and PSM 11 (sparse)
          3. Merge all results into one de-duplicated text corpus
          4. Score against expanded spam keyword list + prize-amount regex
          5. Run ML classifier on merged text (if model is loaded)
        """
        try:
            import pytesseract
        except ImportError:
            return [], {"ocr_error": "pytesseract not installed — run: pip install pytesseract"}

        signals: list[dict] = []
        details: dict       = {}

        # ── Build preprocessed variants ───────────────────────────────────────
        try:
            variants = self._preprocess_for_ocr(img)
        except Exception as e:
            details["preprocess_error"] = str(e)[:120]
            variants = [img]   # fall back to raw image

        # ── Run OCR on every variant × every PSM mode ─────────────────────────
        psm_configs = [
            "--oem 3 --psm 6",    # uniform block of text
            "--oem 3 --psm 11",   # sparse text (no block assumptions)
            "--oem 3 --psm 3",    # fully automatic page segmentation
        ]

        raw_texts: list[str] = []
        per_variant_chars: list[int] = []

        for v_idx, variant in enumerate(variants):
            best_for_variant = ""
            for cfg in psm_configs:
                try:
                    t = pytesseract.image_to_string(variant, config=cfg).strip()
                    if len(t) > len(best_for_variant):
                        best_for_variant = t
                except Exception:
                    pass
            raw_texts.append(best_for_variant)
            per_variant_chars.append(len(best_for_variant))

        details["ocr_chars_per_variant"] = per_variant_chars

        # ── Merge all variants into one text corpus ───────────────────────────
        merged_text = self._merge_ocr_texts(raw_texts)
        details["ocr_text"]       = merged_text[:1500]
        details["ocr_char_count"] = len(merged_text)
        details["ocr_best_variant"] = int(
            per_variant_chars.index(max(per_variant_chars))
        )

        if len(merged_text) < 10:
            details["ocr_note"] = "Very little text extracted — image may be a photo or icon"
            return signals, details

        # ── Spam keyword matching ─────────────────────────────────────────────
        text_lower = merged_text.lower()
        hits = [kw for kw in VISUAL_SPAM_KEYWORDS if kw in text_lower]
        details["spam_keyword_hits"] = hits
        if hits:
            signals.append({
                "signal":      "ocr_spam_keywords",
                "weight":      min(40, len(hits) * 5),
                "description": (
                    f"Image text contains {len(hits)} spam keyword(s): "
                    f"{', '.join(hits[:8])}"
                ),
            })

        # ── Prize / money amount detection ────────────────────────────────────
        # Also check raw OCR text (before lowercasing) for currency symbols
        raw_combined = " ".join(raw_texts)
        prize_amounts = PRIZE_AMOUNT_RE.findall(raw_combined)
        if prize_amounts:
            flat = [f"{''.join(m)}" for m in prize_amounts]
            details["prize_amounts_detected"] = flat[:5]
            signals.append({
                "signal":      "prize_amount_detected",
                "weight":      20,
                "description": f"Large monetary amounts detected in image: {', '.join(flat[:3])}",
            })

        # ── Suspicious domain in image text (e.g. address bar in screenshot) ──
        domain_re = re.compile(r"\b[\w-]+\.(xyz|tk|ml|ga|cf|top|click|win|loan)\b", re.I)
        domains_found = domain_re.findall(raw_combined)
        if domains_found:
            details["suspicious_domains_in_image"] = list(set(domains_found))
            signals.append({
                "signal":      "suspicious_domain_in_image",
                "weight":      20,
                "description": (
                    f"Suspicious domain TLD found in image text: "
                    f"{', '.join(set(domains_found))}"
                ),
            })

        # ── ML classifier on merged OCR text ─────────────────────────────────
        if self.predictor and self.predictor.available():
            clf = self.predictor.predict(merged_text[:MAX_PAGE_CHARS])
            details["classifier"] = clf
            if clf["label"] == "SPAM":
                signals.append({
                    "signal":      "ocr_text_spam_classifier",
                    "weight":      35,
                    "description": (
                        f"Spam classifier flagged image text as SPAM "
                        f"(confidence {clf['confidence']*100:.0f}%)"
                    ),
                })

        return signals, details

    def detect_visual_spam(self, img: Image.Image) -> tuple[list[dict], dict]:
        """
        Visual property analysis:
        - Warm/urgency color palette using HSV saturation + hue (more accurate than RGB thresholds)
        - Auto-strip browser chrome before measuring colors
        - Banner aspect ratio
        - High text-area density
        - High contrast (typical of spam banners)
        """
        import numpy as np

        signals: list[dict] = []
        details: dict       = {}

        w, h = img.size
        details["width"]        = w
        details["height"]       = h
        details["aspect_ratio"] = round(w / h, 2)

        # Banner-like aspect ratio
        ar = w / h
        if ar > 4.0 or ar < 0.25:
            signals.append({
                "signal":      "banner_aspect_ratio",
                "weight":      8,
                "description": f"Image has a banner-like aspect ratio ({ar:.1f}:1)",
            })

        # ── HSV-based warm colour analysis ────────────────────────────────────
        # Convert to HSV; warm colours = hue 0-30° (red) or 40-65° (yellow),
        # with high saturation (>0.4) and medium-high value (>0.3)
        thumb = img.resize((120, 120))
        arr_rgb = np.array(thumb, dtype=np.float32) / 255.0

        r, g, b = arr_rgb[..., 0], arr_rgb[..., 1], arr_rgb[..., 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        # Hue calculation (degrees 0-360)
        hue = np.zeros_like(cmax)
        mask = delta > 0
        # R-segment
        m = mask & (cmax == r)
        hue[m] = 60 * (((g[m] - b[m]) / delta[m]) % 6)
        # G-segment
        m = mask & (cmax == g)
        hue[m] = 60 * ((b[m] - r[m]) / delta[m] + 2)
        # B-segment
        m = mask & (cmax == b)
        hue[m] = 60 * ((r[m] - g[m]) / delta[m] + 4)

        saturation = np.where(cmax > 0, delta / cmax, 0)
        value      = cmax

        # Warm pixels: (red hue OR yellow hue) AND saturated AND bright
        red_hue    = (hue <= 30) | (hue >= 330)
        yellow_hue = (hue > 30) & (hue <= 75)
        saturated  = saturation > 0.40
        bright     = value > 0.30

        red_pixels    = int(np.sum(red_hue    & saturated & bright))
        yellow_pixels = int(np.sum(yellow_hue & saturated & bright))
        total_pixels  = hue.size
        warm_ratio    = (red_pixels + yellow_pixels) / total_pixels

        details["warm_color_ratio"] = round(warm_ratio, 3)
        details["red_pixel_pct"]    = round(red_pixels    / total_pixels * 100, 1)
        details["yellow_pixel_pct"] = round(yellow_pixels / total_pixels * 100, 1)

        if warm_ratio > 0.18:          # lower threshold than RGB version
            signals.append({
                "signal":      "urgency_color_palette",
                "weight":      15,
                "description": (
                    f"Image is {warm_ratio*100:.0f}% red/yellow (HSV) — "
                    "urgency colour palette common in spam banners"
                ),
            })

        # ── Brightness contrast ───────────────────────────────────────────────
        gray_arr = np.array(thumb.convert("L"), dtype=float)
        brightness_std = float(gray_arr.std())
        details["brightness_std"] = round(brightness_std, 1)
        if brightness_std > 80:
            signals.append({
                "signal":      "high_contrast_image",
                "weight":      8,
                "description": (
                    f"High-contrast image (std={brightness_std:.0f}) — "
                    "typical of spam banners with bold coloured text"
                ),
            })

        # ── Text-area density ─────────────────────────────────────────────────
        try:
            import pytesseract
            data = pytesseract.image_to_data(
                img, output_type=pytesseract.Output.DICT,
                config="--oem 3 --psm 6"
            )
            text_area = sum(
                w_ * h_
                for w_, h_, conf in zip(data["width"], data["height"], data["conf"])
                if int(conf) >= MIN_OCR_CONF and w_ > 5 and h_ > 5
            )
            image_area = img.width * img.height
            text_ratio = text_area / image_area if image_area else 0
            details["text_area_ratio"] = round(text_ratio, 3)

            if text_ratio > 0.30:
                signals.append({
                    "signal":      "high_text_density",
                    "weight":      10,
                    "description": (
                        f"Text covers {text_ratio*100:.0f}% of the image — "
                        "high-density text is common in image-based spam"
                    ),
                })
        except Exception:
            pass

        return signals, details

    # ── Full image check ──────────────────────────────────────────────────────

    def check(self, source: str) -> CheckResult:
        result = CheckResult(input=source, input_type="image")

        try:
            img = self._load_image(source)
            result.details["image_size"] = list(img.size)
        except Exception as e:
            result.error = f"Could not load image: {e}"
            result.verdict = "ERROR"
            return result

        # [I1] OCR + classifier
        ocr_signals, ocr_details = self.ocr_and_classify(img)
        for s in ocr_signals:
            result.add_signal(s["signal"], s["weight"], s["description"])
        result.details["ocr"] = ocr_details
        if ocr_details.get("classifier"):
            result.classifier_result = ocr_details["classifier"]

        # [I2] Visual signals
        vis_signals, vis_details = self.detect_visual_spam(img)
        for s in vis_signals:
            result.add_signal(s["signal"], s["weight"], s["description"])
        result.details["visual"] = vis_details

        result.finalize_verdict()
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Combined facade
# ─────────────────────────────────────────────────────────────────────────────

class LinkImageChecker:
    """
    High-level entry point. Auto-detects whether input is a URL or image.

    Usage:
        checker = LinkImageChecker(use_model=True)
        result  = checker.check("http://suspicious.xyz/login")
        result  = checker.check("banner.png")
        print(result.to_dict())
    """

    def __init__(self, model_type: str = "baseline", use_model: bool = True):
        predictor = _LazyPredictor(model_type) if use_model else None
        self.url_checker   = URLChecker(predictor)
        self.image_checker = ImageChecker(predictor)

    def check_url(self, url: str) -> CheckResult:
        return self.url_checker.check(url)

    def check_image(self, source: str) -> CheckResult:
        return self.image_checker.check(source)

    def check(self, source: str) -> CheckResult:
        """Auto-detect type and dispatch."""
        if _looks_like_image(source):
            return self.check_image(source)
        return self.check_url(source)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy of a string."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"}

def _looks_like_image(source: str) -> bool:
    path = urllib.parse.urlparse(source).path.lower()
    return any(path.endswith(ext) for ext in IMAGE_EXTENSIONS)


def _verdict_color(verdict: str) -> str:
    return {
        "CLEAN":      "\033[92m",   # green
        "SUSPICIOUS": "\033[93m",   # yellow
        "SPAM":       "\033[91m",   # red
        "PHISHING":   "\033[91m",   # red
        "ERROR":      "\033[90m",   # grey
    }.get(verdict, "\033[0m")


def _print_result(result: CheckResult):
    RESET = "\033[0m"
    BOLD  = "\033[1m"
    col   = _verdict_color(result.verdict)

    print()
    print("=" * 60)
    print(f"  Input      : {result.input[:70]}")
    print(f"  Type       : {result.input_type.upper()}")
    print(f"  Verdict    : {BOLD}{col}{result.verdict}{RESET}")
    print(f"  Risk score : {col}{result.risk_score}/100{RESET}")

    if result.classifier_result:
        clf = result.classifier_result
        print(f"  Classifier : {clf['label']}  "
              f"(spam {clf['spam_prob']*100:.1f}% / ham {clf['ham_prob']*100:.1f}%)")

    if result.signals:
        print()
        print("  Signals triggered:")
        for s in sorted(result.signals, key=lambda x: -x["weight"]):
            print(f"    [{s['weight']:+3d}]  {s['signal']}")
            print(f"           {s['description'][:90]}")

    if result.error:
        print(f"\n  Error: {result.error}")

    # Useful details
    page = result.details.get("page", {})
    if page.get("page_title"):
        print(f"\n  Page title : {page['page_title'][:80]}")
    chain = result.details.get("redirect_chain", [])
    if len(chain) > 1:
        print(f"\n  Redirect chain ({len(chain)-1} hop(s)):")
        for step in chain:
            print(f"    → {step[:80]}")

    ocr = result.details.get("ocr", {})
    if ocr.get("ocr_text"):
        print(f"\n  OCR text preview:")
        print(f"    {ocr['ocr_text'][:200]}")

    print("=" * 60)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check a URL or image for spam / phishing signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python link_image_checker.py --url "http://bit.ly/win-free-iphone"
  python link_image_checker.py --url "https://paypal.com.verify.evil.xyz/login"
  python link_image_checker.py --image banner.png
  python link_image_checker.py --image "https://example.com/promo.jpg"
  python link_image_checker.py --url "http://suspicious.tk" --no-model
  python link_image_checker.py --url "http://example.com" --json
        """
    )
    parser.add_argument("--url",      type=str, help="URL to check")
    parser.add_argument("--image",    type=str, help="Image path or URL to check")
    parser.add_argument("--model",    choices=["baseline", "transformer"], default="baseline")
    parser.add_argument("--no-model", action="store_true",
                        help="Skip ML classifier (structural/visual analysis only)")
    parser.add_argument("--json",     action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if not args.url and not args.image:
        parser.print_help()
        raise SystemExit(1)

    checker = LinkImageChecker(model_type=args.model, use_model=not args.no_model)

    if args.url:
        result = checker.check_url(args.url)
    else:
        result = checker.check_image(args.image)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        _print_result(result)