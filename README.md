# Qwen Proxy

Reverse proxy biến **Qwen Chat website** (`chat.qwen.ai`) thành **OpenAI-compatible API** — zero dependencies, chỉ dùng Python stdlib.

Dùng Qwen models (qwen3.7-max, qwen3.7-plus, qwen3.6-plus, qwen3.6-max, qwen3.6-27b) với bất kỳ công cụ nào hỗ trợ OpenAI API: Claude Code, Cursor, Continue, aider, LiteLLM, Open WebUI, v.v.

## Tính năng

- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`)
- Anthropic-compatible API (`/v1/messages`, `/anthropic/v1/messages`)
- 3 chế độ thinking: `auto`, `thinking` (luôn suy luận), `fast` (không suy luận)
- Streaming support (SSE)
- Tool calling (function calling) — OpenAI format
- Raw mode mặc định: dùng đúng model client yêu cầu, không tự đổi model khi có tools
- Auto-retry khi Qwen upstream trả response rỗng hoặc text từ chối gọi tool
- Zero dependencies — chỉ cần Python 3.11+
- Stateless — mỗi request tạo chat session mới

## Cài đặt

```bash
git clone https://github.com/dangnosuy/qwen-proxy.git
cd qwen-proxy
```

Không cần `pip install` gì cả — project chỉ dùng Python stdlib.

## Lấy JWT Token

1. Mở trình duyệt, đăng nhập tại https://chat.qwen.ai
2. Mở DevTools (F12) → Network tab
3. Gửi một tin nhắn bất kỳ
4. Tìm request đến `chat.qwen.ai/api/v2/...`
5. Copy giá trị header `Authorization: Bearer <token>`
6. Token hết hạn sau ~30 ngày

## Chạy Proxy

```bash
QWEN_JWT="your-jwt-token-here" python3 qwen_proxy.py --port 8080
```

Hoặc chạy background:

```bash
QWEN_JWT="your-jwt-token" python3 qwen_proxy.py --port 8080 &
```

Kiểm tra:

```bash
curl http://localhost:8080/health
# {"status": "ok"}

curl http://localhost:8080/v1/models
# Danh sách models
```

## Models

Mỗi model có 3 chế độ:

| Model | Mode | Mô tả |
|---|---|---|
| `qwen3.7-max` | Auto | Flagship Qwen3.7, text-only, mạnh nhất cho reasoning/coding |
| `qwen3.7-max-thinking` | Thinking | Qwen3.7 Max + luôn suy luận |
| `qwen3.7-max-fast` | Fast | Qwen3.7 Max + tắt thinking |
| `qwen3.7-plus` | Auto | Qwen3.7 Plus khi tài khoản/region được rollout |
| `qwen3.7-plus-thinking` | Thinking | Qwen3.7 Plus + luôn suy luận |
| `qwen3.7-plus-fast` | Fast | Qwen3.7 Plus + tắt thinking |
| `qwen3.6-plus` | Auto | Model tự quyết có thinking hay không |
| `qwen3.6-plus-thinking` | Thinking | Luôn suy luận trước khi trả lời |
| `qwen3.6-plus-fast` | Fast | Trả lời ngay, nhanh nhất |
| `qwen3.6-max-preview` | Auto | Flagship, mạnh nhất |
| `qwen3.6-max-preview-thinking` | Thinking | Flagship + luôn suy luận |
| `qwen3.6-max-preview-fast` | Fast | Flagship + nhanh |
| `qwen3.6-27b` | Auto | Model nhỏ 27B |
| `qwen3.6-27b-thinking` | Thinking | 27B + luôn suy luận |
| `qwen3.6-27b-fast` | Fast | 27B + nhanh |

## Sử dụng

### curl

```bash
# Cơ bản
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-plus","messages":[{"role":"user","content":"Hello"}]}'

# Streaming
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-plus","messages":[{"role":"user","content":"Đếm 1-10"}],"stream":true}'

# Thinking mode
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-plus-thinking","messages":[{"role":"user","content":"Solve x^2-5x+6=0"}]}'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

r = client.chat.completions.create(
    model="qwen3.6-plus",
    messages=[{"role": "user", "content": "Hello"}]
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

### Claude Code

```json
{
  "model": "qwen3.6-plus",
  "apiBaseUrl": "http://localhost:8080"
}
```

### Cursor / Continue / aider

```bash
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=unused
```

### LiteLLM

```python
import litellm
r = litellm.completion(
    model="openai/qwen3.6-plus",
    messages=[{"role": "user", "content": "Hi"}],
    api_base="http://localhost:8080/v1",
    api_key="unused"
)
```

## Cấu trúc project

```
qwen_proxy/
├── src/qwen_proxy/
│   ├── __init__.py
│   ├── server.py        # HTTP server chính
│   ├── anthropic.py     # Anthropic API compatibility
│   ├── toolcall.py      # Tool calling (non-stream)
│   └── toolstream.py    # Tool calling (stream)
├── tests/               # Test suite
├── pyproject.toml
└── README.md
```

## Ghi chú

- Proxy listen `127.0.0.1` mặc định. Dùng `--host 0.0.0.0` để expose ra ngoài.
- Mỗi request tạo chat session mới trên Qwen (stateless).
- JWT token hết hạn sau ~30 ngày — cần lấy lại từ browser.
- Entry point chính chạy raw mode. Nếu cần bản server cũ, chạy module `qwen_proxy.server` trực tiếp.
- Suffix `-thinking` và `-fast` hoạt động với tất cả models.
- Không có suffix = Auto mode.

## License

MIT
