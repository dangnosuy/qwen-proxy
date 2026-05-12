#!/usr/bin/env python3
import json
import uuid
import time
import requests

TOKEN = "T2gAH8uZFmjn6G8g2loe7HN60M71BLd6yyl448Ktcra-jqqRxR-lOss9xPwek_KL-gM="

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Origin": "https://chat.qwen.ai",
    "Referer": "https://chat.qwen.ai/c/guest",
    "Source": "web",
    "Version": "0.2.45",
    "Bx-V": "2.5.36",
    "Bx-Umidtoken": TOKEN,
}

# Create chat
resp = requests.post("https://chat.qwen.ai/api/v2/chats/new", headers=HEADERS, json={
    "title": "Debug",
    "models": ["qwen3.6-plus"],
    "chat_mode": "guest",
    "chat_type": "t2t",
    "timestamp": int(time.time() * 1000),
    "project_id": ""
})
print(f"Create chat: {resp.json()}")
cid = resp.json()["data"]["id"]

mid = str(uuid.uuid4())
payload = {
    "stream": True,
    "version": "2.1",
    "incremental_output": True,
    "chat_id": cid,
    "chat_mode": "guest",
    "model": "qwen3.6-plus",
    "parent_id": None,
    "messages": [{
        "fid": mid,
        "parentId": None,
        "childrenIds": [],
        "role": "user",
        "content": "Say hello",
        "chat_type": "t2t",
        "feature_config": {
            "thinking_enabled": False,
            "auto_search": False,
        },
        "timestamp": int(time.time()),
        "models": ["qwen3.6-plus"],
        "sub_chat_type": "t2t",
        "parent_id": None,
    }],
    "timestamp": int(time.time()),
}

resp = requests.post(
    f"https://chat.qwen.ai/api/v2/chat/completions?chat_id={cid}",
    headers=HEADERS,
    json=payload,
    stream=True,
)

print(f"\nStatus: {resp.status_code}")
print(f"Content-Type: {resp.headers.get('Content-Type')}")
print(f"Transfer-Encoding: {resp.headers.get('Transfer-Encoding')}")
print(f"Content-Encoding: {resp.headers.get('Content-Encoding')}")
print()

raw = resp.content
print(f"Raw bytes length: {len(raw)}")
print(f"Raw first 500 bytes: {raw[:500]}")
print()

text = resp.text if hasattr(resp, '_content') else raw.decode('utf-8', errors='replace')
print(f"Decoded text (first 1000 chars):")
print(text[:1000])
