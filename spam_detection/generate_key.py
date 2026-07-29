"""
generate_key.py - Standalone script to generate API keys for your spam detection API.

Usage:
    python generate_key.py                      # generate key named "default"
    python generate_key.py --name n8n-prod      # generate key with custom name
    python generate_key.py --list               # list all existing keys
    python generate_key.py --revoke sk-spam-... # revoke a key
"""

import json
import uuid
import argparse
from datetime import datetime

API_KEYS_FILE = "api_keys.json"


def load_keys():
    try:
        with open(API_KEYS_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_keys(keys):
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)


def generate_key(name: str) -> str:
    keys = load_keys()
    key = "sk-spam-" + uuid.uuid4().hex[:32]
    keys[key] = {
        "name"    : name,
        "created" : datetime.utcnow().isoformat() + "Z",
        "requests": 0,
        "active"  : True,
    }
    save_keys(keys)
    return key


def list_keys():
    keys = load_keys()
    if not keys:
        print("No API keys found.")
        return
    print(f"\n{'─'*70}")
    print(f"{'Name':<20} {'Key (preview)':<30} {'Created':<25} {'Requests'}")
    print(f"{'─'*70}")
    for k, v in keys.items():
        preview = k[:20] + "..."
        print(f"{v['name']:<20} {preview:<30} {v['created'][:19]:<25} {v['requests']}")
    print(f"{'─'*70}\n")


def revoke_key(key: str):
    keys = load_keys()
    if key in keys:
        name = keys[key]["name"]
        del keys[key]
        save_keys(keys)
        print(f"Revoked key for '{name}': {key[:20]}...")
    else:
        print(f"Key not found: {key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage API keys for Spam Detection API")
    parser.add_argument("--name",   default="default", help="Name for the new key")
    parser.add_argument("--list",   action="store_true", help="List all keys")
    parser.add_argument("--revoke", metavar="KEY", help="Revoke a key")
    args = parser.parse_args()

    if args.list:
        list_keys()

    elif args.revoke:
        revoke_key(args.revoke)

    else:
        key = generate_key(args.name)
        port = 8000

        print(f"""
{'='*60}
  API Key Generated Successfully
{'='*60}
  Name     : {args.name}
  Key      : {key}
  Saved to : {API_KEYS_FILE}
{'='*60}

HOW TO ADD THIS IN N8N:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Step 1 — Start your API server:
    python api_server.py --model baseline

  Step 2 — In n8n, go to:
    Settings > Credentials > Add Credential > OpenAI API

  Step 3 — Fill in:
    API Key   : {key}
    Base URL  : http://YOUR_SERVER_IP:{port}

  Step 4 — In your "OpenAI Chat Model" node:
    Model     : spam-detector-v1

  Step 5 — In your "If" node, set condition:
    {{ $json.choices[0].message.content | fromJson | .label }} = "SPAM"

{'='*60}
""")
