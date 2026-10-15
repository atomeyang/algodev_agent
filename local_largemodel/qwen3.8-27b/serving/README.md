# serving — qwen3.8-27b

```bash
# 启动（幂等：已在运行时不要重复启动）
bash local_largemodel/qwen3.8-27b/serving/start_llama_server.sh

# 健康检查
curl -s http://127.0.0.1:8084/v1/models | head

# 冒烟测试（文本）
curl -s http://127.0.0.1:8084/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","max_tokens":64,"messages":[{"role":"user","content":"ping"}]}'
```

端点：`http://127.0.0.1:8084/v1`（OpenAI 兼容，支持 `image_url` 的 base64 data URI）。
