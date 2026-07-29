"""
api_server.py - OpenAI-compatible REST API for your spam detection model.

Exposes:
  POST /v1/chat/completions   ← n8n "OpenAI Chat Model" node points here
  POST /v1/classify           ← direct classification endpoint
  GET  /v1/models             ← model listing (required by OpenAI-compatible clients)
  GET  /health                ← health check

n8n Configuration:
  - Credential type : "OpenAI API"
  - Base URL        : http://YOUR_SERVER_IP:8000
  - API Key         : (the key from api_keys.json)
  - Model           : spam-detector-v1

Run:
  python api_server.py
  python api_server.py --host 0.0.0.0 --port 8000 --model baseline
"""

import os
import sys
import json
import uuid
import time
import argparse
import logging
from datetime import datetime
from typing import Optional

# ── FastAPI ──────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config
from src.predict import SpamPredictor
from link_image_checker import LinkImageChecker

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("spam-api")

# ── API Key store ─────────────────────────────────────────────────────────────
API_KEYS_FILE = "api_keys.json"

def load_api_keys() -> dict:
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE) as f:
            return json.load(f)
    return {}

def save_api_keys(keys: dict):
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)

def generate_api_key(name: str = "default") -> str:
    """Generate and persist a new API key."""
    keys = load_api_keys()
    new_key = "sk-spam-" + uuid.uuid4().hex[:32]
    keys[new_key] = {
        "name": name,
        "created": datetime.utcnow().isoformat(),
        "requests": 0,
    }
    save_api_keys(keys)
    log.info(f"Generated API key for '{name}': {new_key}")
    return new_key

# ── Auth dependency ───────────────────────────────────────────────────────────
bearer_scheme = HTTPBearer(auto_error=False)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    keys = load_api_keys()
    if not keys:
        # No keys configured → open access (dev mode)
        return "dev"
    if credentials is None or credentials.credentials not in keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    # Track usage
    keys[credentials.credentials]["requests"] += 1
    save_api_keys(keys)
    return credentials.credentials

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str          # "user" | "assistant" | "system"
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "spam-detector-v1"
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False

class ClassifyRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.5

class CheckLinkRequest(BaseModel):
    url: str

class CheckImageRequest(BaseModel):
    image_url: str          # public image URL

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Spam Detection API",
    description="OpenAI-compatible API for your trained spam classifier.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor (loaded once at startup)
predictor: Optional[SpamPredictor] = None
link_checker: Optional[LinkImageChecker] = None
MODEL_TYPE = "baseline"   # overridden by --model arg


@app.on_event("startup")
async def startup():
    global predictor, link_checker
    log.info(f"Loading {MODEL_TYPE} spam detector...")
    try:
        predictor = SpamPredictor(model_type=MODEL_TYPE)
        log.info("Model loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        log.warning("API will start but /v1/chat/completions will return 503 until model is ready.")

    # Link + image checker (always available, model-independent)
    link_checker = LinkImageChecker(use_model=False)
    log.info("Link/image checker ready.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status"      : "ok" if predictor else "model_not_loaded",
        "model_type"  : MODEL_TYPE,
        "timestamp"   : datetime.utcnow().isoformat(),
    }


@app.get("/v1/models")
def list_models(api_key: str = Depends(verify_api_key)):
    """Required by OpenAI-compatible clients like n8n."""
    return {
        "object": "list",
        "data": [
            {
                "id"      : "spam-detector-v1",
                "object"  : "model",
                "created" : 1700000000,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    OpenAI-compatible chat completions endpoint.

    n8n sends the email text as the last user message.
    We classify it and return a structured JSON response that your
    n8n 'If' node can read via the 'message.content' field.

    Response JSON (inside content):
        {
            "label": "SPAM" | "HAM",
            "spam_prob": 0.97,
            "ham_prob": 0.03,
            "confidence": 0.97,
            "action": "move_to_spam" | "keep_in_inbox"
        }
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Extract the email text from the last user message
    user_messages = [m for m in body.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found.")

    email_text = user_messages[-1].content
    log.info(f"Classifying [{len(email_text)} chars]: {email_text[:80]}...")

    # Classify
    result = predictor.predict(email_text)
    result["action"] = "move_to_spam" if result["label"] == "SPAM" else "keep_in_inbox"
    result["model"]  = MODEL_TYPE

    # Format as OpenAI response
    response_text = json.dumps(result)

    return {
        "id"     : f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object" : "chat.completion",
        "created": int(time.time()),
        "model"  : body.model,
        "choices": [
            {
                "index"        : 0,
                "message"      : {
                    "role"   : "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens"    : len(email_text.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens"     : len(email_text.split()) + len(response_text.split()),
        },
    }


@app.post("/v1/classify")
def classify(
    body: ClassifyRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Direct classification endpoint (simpler than chat/completions).
    Use this if you switch to an HTTP Request node in n8n.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    result = predictor.predict(body.text)
    result["action"] = "move_to_spam" if result["label"] == "SPAM" else "keep_in_inbox"
    return result


@app.post("/v1/check-link")
def check_link(
    body: CheckLinkRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Scan a URL for spam/phishing signals.

    n8n: HTTP Request node → POST /v1/check-link
    Body: { "url": "http://suspicious.xyz/claim" }

    Returns:
        verdict      : SPAM | PHISHING | SUSPICIOUS | CLEAN
        risk_score   : 0-100
        signals      : list of triggered signals with weights
    """
    result = link_checker.check_url(body.url)
    d = result.to_dict()
    d["is_spam"] = result.verdict in ("SPAM", "PHISHING")
    d["action"]  = "block" if d["is_spam"] else "allow"
    return d


@app.post("/v1/check-image")
def check_image(
    body: CheckImageRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Scan an image URL for visual spam signals + OCR text classification.

    n8n: HTTP Request node → POST /v1/check-image
    Body: { "image_url": "https://cdn.example.com/promo.jpg" }

    Returns:
        verdict      : SPAM | SUSPICIOUS | CLEAN
        risk_score   : 0-100
        signals      : list of triggered signals with weights
        ocr_text     : text extracted from the image
    """
    result = link_checker.check_image(body.image_url)
    d = result.to_dict()
    d["is_spam"] = result.verdict in ("SPAM", "PHISHING")
    d["action"]  = "block" if d["is_spam"] else "allow"
    d["ocr_text"] = (result.details.get("ocr") or {}).get("ocr_text", "")
    return d


@app.get("/v1/keys")
def list_keys(request: Request):
    """List all API keys (admin only — protect this in production)."""
    keys = load_api_keys()
    return {k: {**v, "key_preview": k[:16] + "..."} for k, v in keys.items()}


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spam Detection API Server")
    parser.add_argument("--host",  default="0.0.0.0",  help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port",  default=8000, type=int, help="Port (default: 8000)")
    parser.add_argument("--model", default="baseline",
                        choices=["baseline", "transformer"],
                        help="Which trained model to serve (default: baseline)")
    parser.add_argument("--generate-key", metavar="NAME",
                        help="Generate a new API key with this name and exit")
    args = parser.parse_args()

    # Key generation mode
    if args.generate_key:
        key = generate_api_key(args.generate_key)
        print(f"\n{'='*55}")
        print(f"  New API Key Generated")
        print(f"{'='*55}")
        print(f"  Name : {args.generate_key}")
        print(f"  Key  : {key}")
        print(f"{'='*55}")
        print(f"\nAdd this to n8n:")
        print(f"  Credential type : OpenAI API")
        print(f"  Base URL        : http://YOUR_SERVER_IP:{args.port}")
        print(f"  API Key         : {key}")
        print(f"  Model           : spam-detector-v1\n")
        sys.exit(0)

    MODEL_TYPE = args.model
    log.info(f"Starting Spam Detection API — model={MODEL_TYPE}, port={args.port}")

    # Auto-generate a key if none exist
    if not load_api_keys():
        key = generate_api_key("default")
        print(f"\n{'='*55}")
        print(f"  Auto-generated API Key (no keys found)")
        print(f"{'='*55}")
        print(f"  Key  : {key}")
        print(f"\nCopy this into n8n OpenAI credential > API Key field.")
        print(f"Base URL: http://YOUR_SERVER_IP:{args.port}\n{'='*55}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")