# Qwen Proxy — Hướng dẫn sử dụng

## Khởi động

```bash
QWEN_JWT="<jwt-token-của-bạn>" python3 qwen_proxy.py --port 8080
```

Hoặc chạy background:

```bash
QWEN_JWT="<jwt>" python3 qwen_proxy.py --port 8080 &
```

---

## Models

Proxy mặc định chạy raw mode: client yêu cầu model nào thì gửi đúng model đó lên Qwen, không tự đổi sang tool model khác. Khi upstream trả response rỗng hoặc trả text kiểu từ chối gọi tool, proxy sẽ retry theo `QWEN_RAW_MAX_RETRIES` (mặc định `2`).

Mỗi model có 3 chế độ thinking:

| Model | Thinking | Mô tả |
|---|---|---|
| `qwen3.6-plus` | Auto | Model tự quyết có cần suy luận hay không |
| `qwen3.6-plus-thinking` | Luôn bật | Luôn suy luận trước khi trả lời |
| `qwen3.6-plus-fast` | Tắt | Trả lời ngay, nhanh nhất |
| `qwen3.7-max` | Auto | Flagship Qwen3.7, text-only, mạnh cho reasoning/coding |
| `qwen3.7-max-thinking` | Luôn bật | Qwen3.7 Max + luôn suy luận |
| `qwen3.7-max-fast` | Tắt | Qwen3.7 Max + nhanh |
| `qwen3.7-plus` | Auto | Qwen3.7 Plus khi tài khoản/region được rollout |
| `qwen3.7-plus-thinking` | Luôn bật | Qwen3.7 Plus + luôn suy luận |
| `qwen3.7-plus-fast` | Tắt | Qwen3.7 Plus + nhanh |
| `qwen3.6-max-preview` | Auto | Flagship, mạnh nhất |
| `qwen3.6-max-preview-thinking` | Luôn bật | Flagship + luôn suy luận |
| `qwen3.6-max-preview-fast` | Tắt | Flagship + nhanh |
| `qwen3.6-27b` | Auto | Model nhỏ 27B |
| `qwen3.6-27b-thinking` | Luôn bật | 27B + luôn suy luận |
| `qwen3.6-27b-fast` | Tắt | 27B + nhanh |

Khi thinking bật, response sẽ chứa block `<thinking>...</thinking>` trước câu trả lời.

---

## Test với curl

### Câu hỏi đơn giản (auto mode)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-plus","messages":[{"role":"user","content":"Xin chào, bạn là ai?"}]}'
```

### Thinking mode — luôn suy luận

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-plus-thinking","messages":[{"role":"user","content":"What is 15*23?"}]}'
```

Response sẽ có dạng:
```
<thinking>
I break down 15 × 23 into simpler parts...
15 × 20 = 300, 15 × 3 = 45...
</thinking>

15 × 23 = **345**
```

### Fast mode — trả lời nhanh, không thinking

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-plus-fast","messages":[{"role":"user","content":"What is 15*23?"}]}'
```

Response: `15 * 23 = 345` (không có `<thinking>` block)

### Streaming (real-time)

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-plus","messages":[{"role":"user","content":"Đếm từ 1 đến 10"}],"stream":true}'
```

### Streaming + thinking

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-plus-thinking","messages":[{"role":"user","content":"Solve: x^2 - 5x + 6 = 0"}],"stream":true}'
```

### System prompt + lịch sử hội thoại

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-plus",
    "messages": [
      {"role": "system", "content": "Bạn là một lập trình viên senior. Trả lời ngắn gọn."},
      {"role": "user", "content": "Viết quicksort bằng Python"}
    ]
  }'
```

### Đổi model

```bash
# Flagship model
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-max-preview","messages":[{"role":"user","content":"Explain quantum computing"}]}'

# Flagship + thinking
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-max-preview-thinking","messages":[{"role":"user","content":"Prove that sqrt(2) is irrational"}]}'

# Model nhỏ 27B, fast
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-fast","messages":[{"role":"user","content":"Hello"}]}'
```

### Xem models / health

```bash
curl http://localhost:8080/v1/models
curl http://localhost:8080/health
```

---

## Dùng với OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

# Auto mode
r = client.chat.completions.create(
    model="qwen3.6-plus",
    messages=[{"role": "user", "content": "Hello"}]
)
print(r.choices[0].message.content)

# Thinking mode
r = client.chat.completions.create(
    model="qwen3.6-plus-thinking",
    messages=[{"role": "user", "content": "What is 15*23?"}]
)
print(r.choices[0].message.content)

# Fast mode
r = client.chat.completions.create(
    model="qwen3.6-plus-fast",
    messages=[{"role": "user", "content": "What is 15*23?"}]
)
print(r.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="qwen3.6-plus",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    stream=True
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## Gắn vào Agent Framework

Proxy trả về tool calls theo OpenAI-compatible shape:

- Non-stream: `choices[0].message.tool_calls` + `finish_reason: "tool_calls"`
- Stream: `choices[0].delta.tool_calls` + final chunk `finish_reason: "tool_calls"`
- Tool execution vẫn do client/agent runner thực hiện. Proxy chỉ chuyển output của Qwen web thành protocol đúng để client chạy tool.
- Vòng lặp nhiều lượt dùng chuẩn OpenAI: client gửi lại assistant message có `tool_calls`, sau đó gửi `role: "tool"` với `tool_call_id` tương ứng.

### Claude Code (settings.json)

Anthropic-compatible endpoint hiện có:

```text
POST /v1/messages
POST /anthropic/v1/messages
GET  /anthropic/v1/models
POST /v1/messages/count_tokens
```

```json
{
  "model": "qwen3.6-plus",
  "apiBaseUrl": "http://localhost:8080"
}
```

Nếu client cho chọn provider Anthropic/Claude, trỏ base URL về root proxy, ví dụ `http://localhost:8080`. Nếu client chỉ nhận OpenAI-compatible API thì dùng `http://localhost:8080/v1`.

### Cursor / Continue / aider

```bash
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=unused
```

### LiteLLM

```python
import litellm
r = litellm.completion(
    model="openai/qwen3.6-plus-thinking",
    messages=[{"role": "user", "content": "Hi"}],
    api_base="http://localhost:8080/v1",
    api_key="unused"
)
```

---

## Ghi chú

- Mỗi request tạo chat session mới trên Qwen (stateless)
- JWT token hết hạn sau ~30 ngày, lấy mới từ browser `chat.qwen.ai`
- Proxy chỉ listen localhost mặc định, dùng `--host 0.0.0.0` để expose ra ngoài
- Suffix `-thinking` và `-fast` hoạt động với tất cả models
- Không có suffix = Auto mode (model tự quyết định)
