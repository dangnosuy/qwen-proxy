# Qwen Chat API - Nghiên cứu đầy đủ

> Ngày: 2026-04-30
> Endpoint: `https://chat.qwen.ai`

---

## 1. Tổng quan kiến trúc

```
                    ┌─────────────────────────────────────────┐
                    │              Qwen API Server             │
                    │                                         │
  Guest Mode ──────>│  Bx-Umidtoken (device fingerprint)      │
  (no login)        │  + chat_mode: "guest"                   │
                    │                                         │
  Auth Mode ───────>│  Cookie: token=<JWT>                    │
  (logged in)       │  OR Authorization: Bearer <JWT>         │
                    │  + chat_mode: "normal"                  │
                    │                                         │
                    │  ┌─── Server-side history ────┐         │
                    │  │ parent_id chain per chat   │         │
                    │  └────────────────────────────┘         │
                    └─────────────────────────────────────────┘
```

---

## 2. Hai chế độ xác thực

### 2.1 Guest Mode
- **Không cần login/cookie**
- Header bắt buộc: `Bx-Umidtoken`
- `chat_mode: "guest"`
- Rate limit thấp (~20-30 requests/ngày per token)

### 2.2 Authenticated Mode
- **JWT token** qua Cookie hoặc Bearer
- `chat_mode: "normal"`
- Rate limit cao hơn
- Truy cập thêm: chat list, chat history, notifications

### So sánh auth methods:

| Method | Tạo chat | Gửi message | Ghi chú |
|---|---|---|---|
| `Bx-Umidtoken` only | OK (guest) | OK (guest) | Rate limit thấp |
| `Cookie: token=<JWT>` only | OK | OK | **Không cần Bx-Umidtoken!** |
| `Authorization: Bearer <JWT>` only | OK | OK | **Không cần Bx-Umidtoken!** |
| Cookie + Bx-Umidtoken | OK | OK | Browser mặc định gửi cả hai |
| Không gì cả | FAIL | FAIL | `Unauthorized` |

**Phát hiện quan trọng**: Với JWT token, **không cần Bx-Umidtoken** gì cả. Chỉ cần `Content-Type: application/json` + JWT.

---

## 3. Models có sẵn

| Model ID | Mô tả | Guest | Auth |
|---|---|---|---|
| `qwen3.6-plus` | Model mặc định, text + multimodal | Yes | Yes |
| `qwen3.6-max-preview` | Flagship preview, SOTA. Không hỗ trợ Search/Code Interpreter | Yes | Yes |
| `qwen3.6-27b` | 27B dense, tối ưu local deployment | Yes | Yes |

Các model cũ (`qwen-plus`, `qwen-max`, `qwen-turbo`, `qwen2.5-*`, `qvq-max`) đều trả `Model not found` — ngay cả khi login.

---

## 4. JWT Token

```
Header:  {"alg": "HS256", "typ": "JWT"}
Payload: {
  "id": "<user-uuid>",
  "last_password_change": <unix_timestamp>,
  "exp": <unix_timestamp>           ← hết hạn sau ~30 ngày
}
```

- Token hết hạn: ~30 ngày kể từ lúc login
- Gửi qua: `Cookie: token=<jwt>` HOẶC `Authorization: Bearer <jwt>`
- API `/api/v1/auths/` trả token mới mỗi lần gọi (refresh tự động)

---

## 5. API Endpoints

### 5.1 Configs (không cần auth)
```
GET /api/v2/configs/              → App config, features, limits
GET /api/v2/configs/setting-config → Tools config
GET /api/v2/tts/config            → TTS voices config
```

### 5.2 Auth
```
GET  /api/v1/auths/               → User info + token refresh
POST /api/v2/users/status          → Analytics/tracking (optional)
```

### 5.3 Chat Management (cần auth hoặc bx-token)
```
POST   /api/v2/chats/new                    → Tạo chat mới
GET    /api/v2/chats/?page=1&limit=20       → Danh sách chat (auth only)
DELETE /api/v2/chats/<chat_id>               → Xóa chat (auth only)
```

### 5.4 Chat Completions
```
POST /api/v2/chat/completions?chat_id=<uuid>  → Gửi message
```

---

## 6. Luồng hoạt động chi tiết

### 6.1 Tạo chat mới

```http
POST /api/v2/chats/new

Body:
{
  "title": "New Chat",
  "models": ["qwen3.6-plus"],
  "chat_mode": "guest" | "normal",
  "chat_type": "t2t",
  "timestamp": 1777528203574,        ← milliseconds
  "project_id": ""
}

Response: { "success": true, "data": { "id": "<chat-uuid>" } }
```

### 6.2 Gửi message

```http
POST /api/v2/chat/completions?chat_id=<chat_id>

Body:
{
  "stream": true,
  "version": "2.1",
  "incremental_output": true,
  "chat_id": "<uuid>",
  "chat_mode": "guest" | "normal",
  "model": "qwen3.6-plus",
  "parent_id": null | "<response_id của turn trước>",
  "messages": [
    {
      "fid": "<uuid>",
      "parentId": null | "<response_id>",
      "childrenIds": [],
      "role": "user",
      "content": "Hello",
      "chat_type": "t2t",
      "feature_config": {
        "thinking_enabled": true|false,
        "output_schema": "phase",
        "research_mode": "normal",
        "auto_thinking": true|false,
        "thinking_mode": "Auto",
        "thinking_format": "summary",
        "auto_search": true|false
      },
      "timestamp": 1777528203,       ← seconds
      "models": ["qwen3.6-plus"],
      "sub_chat_type": "t2t",
      "parent_id": null | "<response_id>"
    }
  ],
  "timestamp": 1777528203            ← seconds
}
```

---

## 7. SSE Response Format

### 7.1 Event types

**Event 1: response.created**
```json
{"response.created": {
  "chat_id": "...",
  "parent_id": "...",
  "response_id": "...",        ← QUAN TRỌNG: dùng làm parent_id cho turn sau
  "response_index": "0"
}}
```

**Event 2: response.info (keep_alive)**
```json
{"response.info": {"action": "keep_alive", ...}}
```

**Event 3: choices (content chunks)**
```json
{"choices": [{"delta": {
  "role": "assistant",
  "content": "text...",
  "phase": "answer",           ← "thinking_summary" hoặc "answer"
  "status": "typing",          ← "typing" hoặc "finished"
  "extra": {...}               ← chỉ có trong thinking phase
}}], "usage": {...}}
```

### 7.2 Thinking mode response phases

Khi `thinking_enabled: true`, response có 2 phases theo thứ tự:

**Phase 1: `thinking_summary`**
```json
{
  "delta": {
    "role": "assistant",
    "content": "",
    "phase": "thinking_summary",
    "status": "typing",
    "extra": {
      "summary_title": {"content": ["Calculating the sum"]},
      "summary_thought": {"content": ["Step by step reasoning..."]}
    }
  }
}
```
→ Kết thúc bằng `status: "finished"`

**Phase 2: `answer`**
```json
{
  "delta": {
    "role": "assistant",
    "content": "The answer is 5",
    "phase": "answer",
    "status": "typing"
  }
}
```
→ Kết thúc bằng `status: "finished"`

### 7.3 Usage tracking
```json
{
  "usage": {
    "input_tokens": 1386,
    "output_tokens": 562,
    "total_tokens": 1948,
    "input_tokens_details": {"text_tokens": 1386},
    "output_tokens_details": {
      "reasoning_tokens": 256,      ← thinking tokens
      "text_tokens": 562
    }
  }
}
```

---

## 8. Cơ chế lịch sử hội thoại

### Quy tắc:
- **Server quản lý toàn bộ lịch sử**
- Client chỉ gửi **1 message mới** mỗi request (nhiều hơn → `"too many messages"`)
- Dùng **`parent_id`** = `response_id` từ turn trước để nối chain

### Chain:
```
Turn 1: parent_id = null     → response_id = "aaa"
Turn 2: parent_id = "aaa"   → response_id = "bbb"
Turn 3: parent_id = "bbb"   → response_id = "ccc"
```

### Khi parent_id sai:
- Server trả response bình thường nhưng **không có context** trước đó
- Không báo lỗi, chỉ mất history

### Khi parent_id đúng:
- Server tự inject toàn bộ lịch sử vào prompt
- Model nhớ tất cả các turn trước

---

## 9. Các tham số có thể thay đổi

### `model`
```
"qwen3.6-plus"          ← mặc định
"qwen3.6-max-preview"   ← flagship
"qwen3.6-27b"           ← nhỏ hơn
```

### `feature_config`

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `thinking_enabled` | `true/false` | Bật thinking (CoT) |
| `thinking_mode` | `"Auto"` | Auto-detect khi nào cần thinking |
| `thinking_format` | `"summary"` | Output summary thay vì full thought |
| `output_schema` | `"phase"` | Response chia theo phases |
| `auto_search` | `true/false` | Tự động search web khi cần |
| `research_mode` | `"normal"` | Deep research mode |

### `chat_mode`
```
"guest"    ← không cần login
"normal"   ← cần JWT token
```

### `chat_type` / `sub_chat_type`
```
"t2t"    ← text to text (mặc định)
```

---

## 10. Headers tối thiểu

### Guest mode:
```
Content-Type: application/json
Bx-Umidtoken: <token>
```

### Auth mode (SIÊU TỐI GIẢN):
```
Content-Type: application/json
Authorization: Bearer <jwt>
```
Hoặc:
```
Content-Type: application/json
Cookie: token=<jwt>
```

**Không cần**: `Bx-Umidtoken`, `Bx-Ua`, `Source`, `Version`, `Bx-V`, `Origin`, `Referer`, `User-Agent`

---

## 11. Rate Limiting

### Guest mode:
- ~20-30 requests/ngày per `Bx-Umidtoken`
- Error: `{"code": "RateLimited", "num": 18}` (giờ phải chờ)

### Auth mode:
- Limit cao hơn đáng kể (chưa đo được giới hạn cụ thể)
- Rate limit theo user account thay vì device token

---

## 12. Baxia SDK

- SDK: `https://g.alicdn.com/sd/baxia-entry/index.js`
- Anti-bot system của Alibaba Cloud
- Sinh `Bx-Umidtoken` (persistent) + `Bx-Ua` (per-request)
- **Bx-Ua có thể bỏ qua hoàn toàn** — server không enforce
- **Với JWT auth, Bx-Umidtoken cũng không cần**

---

## 13. Tools có sẵn (server-side)

| Tool | Mô tả |
|---|---|
| `web_search` | Tìm kiếm web |
| `web_search_image` | Tìm kiếm hình ảnh |
| `web_extractor` | Trích xuất nội dung URL |
| `code_interpreter` | Chạy code (Python) |
| `image_gen_tool` | Tạo hình ảnh |
| `image_edit_tool` | Chỉnh sửa hình ảnh |
| `image_zoom_in_tool` | Phóng to vùng hình ảnh |
| `history_retriever` | Truy xuất lịch sử ngoài session |
| `bio` | Lưu/cập nhật user preferences |

---

## 14. Workflow hoàn chỉnh cho Proxy Script

### Auth mode (khuyên dùng):

```python
import requests, json, uuid, time

JWT = "<jwt-token>"
BASE = "https://chat.qwen.ai"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {JWT}",
}

# Tạo chat
def create_chat(model="qwen3.6-plus"):
    r = requests.post(f"{BASE}/api/v2/chats/new", headers=HEADERS, json={
        "title": "Chat",
        "models": [model],
        "chat_mode": "normal",
        "chat_type": "t2t",
        "timestamp": int(time.time() * 1000),
        "project_id": ""
    })
    return r.json()["data"]["id"]

# Gửi message
def send(chat_id, content, parent_id=None, model="qwen3.6-plus", thinking=False):
    r = requests.post(
        f"{BASE}/api/v2/chat/completions?chat_id={chat_id}",
        headers=HEADERS,
        json={
            "stream": True,
            "version": "2.1",
            "incremental_output": True,
            "chat_id": chat_id,
            "chat_mode": "normal",
            "model": model,
            "parent_id": parent_id,
            "messages": [{
                "fid": str(uuid.uuid4()),
                "parentId": parent_id,
                "role": "user",
                "content": content,
                "chat_type": "t2t",
                "feature_config": {
                    "thinking_enabled": thinking,
                    "output_schema": "phase" if thinking else None,
                    "auto_thinking": thinking,
                    "thinking_mode": "Auto" if thinking else None,
                    "thinking_format": "summary" if thinking else None,
                    "auto_search": False,
                },
                "timestamp": int(time.time()),
                "models": [model],
                "sub_chat_type": "t2t",
                "parent_id": parent_id,
            }],
            "timestamp": int(time.time()),
        },
        stream=True
    )

    text, thinking_text, response_id = "", "", None
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "): continue
        d = json.loads(line[6:])
        if "response.created" in d:
            response_id = d["response.created"]["response_id"]
        elif "choices" in d:
            delta = d["choices"][0].get("delta", {})
            phase = delta.get("phase", "")
            if phase == "thinking_summary":
                extra = delta.get("extra", {})
                thinking_text = "\n".join(extra.get("summary_thought", {}).get("content", []))
            elif phase == "answer":
                text += delta.get("content", "")

    return {"content": text, "thinking": thinking_text, "response_id": response_id}

# Multi-turn
cid = create_chat()
r1 = send(cid, "Hello, my name is Alice")
r2 = send(cid, "What's my name?", parent_id=r1["response_id"])
r3 = send(cid, "Tell me a joke", parent_id=r2["response_id"], thinking=True)
```

---

## 15. Tóm tắt phát hiện chính

| # | Phát hiện | Ghi chú |
|---|---|---|
| 1 | **Auth mode chỉ cần JWT** | `Authorization: Bearer <jwt>` — không cần Bx-Umidtoken, cookie, hay header nào khác |
| 2 | **Guest mode cần Bx-Umidtoken** | Từ Baxia SDK, nhưng Bx-Ua bỏ qua được |
| 3 | **3 models** | qwen3.6-plus, qwen3.6-max-preview, qwen3.6-27b |
| 4 | **Server giữ history** | Client gửi 1 msg + parent_id, server nối chain |
| 5 | **JWT hết hạn 30 ngày** | Refresh qua `/api/v1/auths/` |
| 6 | **Thinking có 2 phases** | `thinking_summary` (extra.summary_thought) → `answer` (content) |
| 7 | **Rate limit guest thấp** | ~20-30/ngày; auth cao hơn nhiều |
| 8 | **CORS đúng** | Chỉ block browser cross-origin, curl/script bypass tự nhiên |
| 9 | **Headers tối thiểu cho auth** | Chỉ cần `Content-Type` + `Authorization` |
