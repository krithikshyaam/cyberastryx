# n8n Integration Guide — Spam Detection API

## Overview

Your n8n workflow currently uses **OpenAI (ChatGPT)** as the AI model.
This guide replaces it with **your locally trained spam detection model**.

---

## Step 1 — Start Your API Server

After training your model, start the API:

```bash
# Baseline BiLSTM model (fast, ~97% accuracy)
python api_server.py --model baseline

# OR fine-tuned BERT (slower, ~99% accuracy)
python api_server.py --model transformer
```

The server starts on `http://0.0.0.0:8000`

---

## Step 2 — Generate Your API Key

```bash
python generate_key.py --name n8n-workflow
```

Output:
```
Key : sk-spam-a1b2c3d4e5f6...
```

Save this — you'll paste it into n8n.

---

## Step 3 — Add Credential in n8n

1. Go to **Settings → Credentials → Add Credential**
2. Search for **"OpenAI API"**
3. Fill in:

| Field    | Value                              |
|----------|------------------------------------|
| API Key  | `sk-spam-xxxxxxxx` (your key)      |
| Base URL | `http://YOUR_SERVER_IP:8000`       |

> If n8n and your API are on the same machine, use `http://localhost:8000`
> If n8n is in Docker: use `http://host.docker.internal:8000`

---

## Step 4 — Update the "OpenAI Chat Model" Node

In your n8n workflow, click the **OpenAI Chat Model** node and set:

| Setting    | Value                        |
|------------|------------------------------|
| Credential | *(the one you just created)* |
| Model      | `spam-detector-v1`           |

---

## Step 5 — Update the "Edit Fields" Node

Your **Edit Fields** node should pass the email body to the AI Agent.
Make sure it outputs a field like:

```
emailText = {{ $json.body || $json.snippet || $json.payload.body }}
```

The AI Agent will forward this as the user message to your API.

---

## Step 6 — Update the "If" Node

Your **If** node needs to read the spam label from the API response.

The API returns this JSON inside `choices[0].message.content`:
```json
{
  "label": "SPAM",
  "confidence": 0.97,
  "spam_prob": 0.97,
  "ham_prob": 0.03,
  "action": "move_to_spam"
}
```

Set your **If** condition to:

```
{{ $json.choices[0].message.content | fromJson | .label }}  equals  SPAM
```

Or using the action field:
```
{{ $json.choices[0].message.content | fromJson | .action }}  equals  move_to_spam
```

---

## Step 7 — Verify with the Direct Classify Endpoint

Before testing in n8n, verify your API works:

```bash
# Test health
curl http://localhost:8000/health

# Test classification (replace YOUR_KEY)
curl -X POST http://localhost:8000/v1/classify \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "FREE ENTRY! Win a prize, call now!"}'

# Expected response:
# {"label":"SPAM","confidence":0.97,"spam_prob":0.97,"ham_prob":0.03,"action":"move_to_spam"}

curl -X POST http://localhost:8000/v1/classify \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hi, can we reschedule our meeting to 3pm?"}'

# Expected response:
# {"label":"HAM","confidence":0.99,"spam_prob":0.01,"ham_prob":0.99,"action":"keep_in_inbox"}
```

---

## Full Workflow Summary

```
Gmail Trigger
    │
    ▼
Get a message         ← fetches full email
    │
    ▼
Edit Fields           ← extracts email body text
    │
    ▼
AI Agent              ← sends text to your spam API
    │  (OpenAI Chat Model → http://localhost:8000)
    ▼
Merge                 ← combines AI output with email data
    │
    ▼
If (label == SPAM?)
    │
    ├── TRUE  → Add label "SPAM" → Remove label "INBOX"
    │
    └── FALSE → Add label "REVIEWED" (keep in inbox)
```

---

## Running in Production (with ngrok for remote n8n)

If your n8n is cloud-hosted but your model is local:

```bash
# Install ngrok, then expose your API
ngrok http 8000

# Use the ngrok URL in n8n:
# Base URL: https://xxxx.ngrok.io
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| 401 Unauthorized | Check API key matches exactly |
| 503 Model not loaded | Model not trained yet — run `trainer.py` first |
| Connection refused | API server not running, or wrong IP/port |
| Wrong label in If node | Check the fromJson expression path |
| Docker n8n can't reach localhost | Use `http://host.docker.internal:8000` |
