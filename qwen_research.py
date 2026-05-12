#!/usr/bin/env python3
"""Research script for Qwen API guest mode"""

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

def create_chat(model="qwen3.6-plus"):
    resp = requests.post("https://chat.qwen.ai/api/v2/chats/new", headers=HEADERS, json={
        "title": "Test",
        "models": [model],
        "chat_mode": "guest",
        "chat_type": "t2t",
        "timestamp": int(time.time() * 1000),
        "project_id": ""
    })
    data = resp.json()
    if not data.get("success"):
        raise Exception(f"Failed to create chat: {data}")
    return data["data"]["id"]

def send_message(chat_id, content, model="qwen3.6-plus", parent_id=None, thinking=False, search=False):
    mid = str(uuid.uuid4())
    payload = {
        "stream": True,
        "version": "2.1",
        "incremental_output": True,
        "chat_id": chat_id,
        "chat_mode": "guest",
        "model": model,
        "parent_id": parent_id,
        "messages": [{
            "fid": mid,
            "parentId": parent_id,
            "childrenIds": [],
            "role": "user",
            "content": content,
            "chat_type": "t2t",
            "feature_config": {
                "thinking_enabled": thinking,
                "output_schema": "phase" if thinking else None,
                "auto_thinking": thinking,
                "thinking_mode": "Auto" if thinking else None,
                "thinking_format": "summary" if thinking else None,
                "auto_search": search,
            },
            "timestamp": int(time.time()),
            "models": [model],
            "sub_chat_type": "t2t",
            "parent_id": parent_id,
        }],
        "timestamp": int(time.time()),
    }

    resp = requests.post(
        f"https://chat.qwen.ai/api/v2/chat/completions?chat_id={chat_id}",
        headers=HEADERS,
        json=payload,
        stream=True
    )

    full_content = ""
    thinking_content = ""
    response_id = None
    usage = None

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            data = json.loads(line[6:])
        except json.JSONDecodeError:
            continue

        if "response.created" in data:
            response_id = data["response.created"]["response_id"]
            continue

        if "choices" in data:
            delta = data["choices"][0].get("delta", {})
            content_piece = delta.get("content", "")
            thinking_piece = delta.get("thinking_content", "")
            full_content += content_piece
            thinking_content += thinking_piece
            usage = data.get("usage")

    return {
        "content": full_content,
        "thinking": thinking_content,
        "response_id": response_id,
        "usage": usage,
    }

# ============================================================
# TEST 1: All models
# ============================================================
print("=" * 60)
print("TEST 1: Available models")
print("=" * 60)

for model in ["qwen3.6-plus", "qwen3.6-max-preview", "qwen3.6-27b"]:
    try:
        cid = create_chat(model)
        result = send_message(cid, "Say only your exact model name. Nothing else.", model=model)
        print(f"  {model:25s} -> {result['content'][:100]}")
        print(f"    Usage: {result['usage']}")
    except Exception as e:
        print(f"  {model:25s} -> ERROR: {e}")

# ============================================================
# TEST 2: Multi-turn conversation (parent_id chain)
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Multi-turn conversation via parent_id")
print("=" * 60)

cid = create_chat()
print(f"Chat ID: {cid}")

r1 = send_message(cid, "My name is Bob and my favorite number is 42. Remember this.")
print(f"\nTurn 1 -> {r1['content'][:200]}")
print(f"  response_id: {r1['response_id']}")

r2 = send_message(cid, "What is my name?", parent_id=r1["response_id"])
print(f"\nTurn 2 -> {r2['content'][:200]}")
print(f"  response_id: {r2['response_id']}")

r3 = send_message(cid, "What is my favorite number?", parent_id=r2["response_id"])
print(f"\nTurn 3 -> {r3['content'][:200]}")

# ============================================================
# TEST 3: Thinking mode
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Thinking mode")
print("=" * 60)

cid = create_chat()
r = send_message(cid, "What is 15 * 23?", thinking=True)
print(f"Thinking: {r['thinking'][:300]}")
print(f"Answer: {r['content'][:200]}")

# ============================================================
# TEST 4: What happens with wrong parent_id (broken chain)?
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: Broken parent_id chain")
print("=" * 60)

cid = create_chat()
r1 = send_message(cid, "My secret word is BANANA.")
print(f"Turn 1: {r1['content'][:150]}")

# Skip r1's response_id, use a fake one
r2 = send_message(cid, "What is my secret word?", parent_id="00000000-0000-0000-0000-000000000000")
print(f"Turn 2 (fake parent): {r2['content'][:150]}")

# Now use correct parent
r3 = send_message(cid, "What is my secret word?", parent_id=r1["response_id"])
print(f"Turn 3 (correct parent): {r3['content'][:150]}")

# ============================================================
# TEST 5: Feature configs
# ============================================================
print("\n" + "=" * 60)
print("TEST 5: Feature config variations")
print("=" * 60)

cid = create_chat()
r_no_think = send_message(cid, "Hi", thinking=False)
print(f"No thinking: content={r_no_think['content'][:80]}... thinking={bool(r_no_think['thinking'])}")

cid2 = create_chat()
r_think = send_message(cid2, "What is 2+2?", thinking=True)
print(f"With thinking: content={r_think['content'][:80]}... thinking_len={len(r_think['thinking'])}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
