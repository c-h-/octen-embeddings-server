# octen-embeddings-server

Local MLX embeddings server with OpenAI-compatible API. Runs [Octen-Embedding-8B](https://huggingface.co/Octen/Octen-Embedding-8B) on Apple Silicon for fast, private embeddings.

## Why

- **Private**: Embeddings never leave your machine. No API keys, no network calls, no data sharing.
- **Fast**: MLX uses Metal for native Apple Silicon acceleration. Typical latency is ~50-200ms per text depending on length.
- **High quality**: Octen-Embedding-8B ranks #1 on MTEB/RTEB with a score of 0.8045, outperforming commercial APIs.
- **Drop-in compatible**: OpenAI `/v1/embeddings` API format. Works with any client that speaks OpenAI embeddings.

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| CPU | Apple Silicon (M1/M2/M3/M4) |
| RAM | 20 GB+ (model uses ~16 GB, server ~1 GB overhead) |
| Disk | ~17 GB for converted model weights |
| OS | macOS 13+ (Ventura or later) |

MLX requires Metal GPU access. This server does **not** run on Intel Macs or Linux.

## Installation

### 1. Clone and install dependencies

```bash
git clone https://github.com/c-h-/octen-embeddings-server.git
cd octen-embeddings-server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

### 2. Download the model

**Option A: Pre-converted weights (recommended)**

Download the ready-to-use MLX weights from HuggingFace (~16 GB):

```bash
pip install huggingface-hub
huggingface-cli download chulcher/Octen-Embedding-8B-mlx --local-dir models/Octen-Embedding-8B-mlx
```

**Option B: Convert from scratch**

If you prefer to convert the original model yourself (~32 GB temp disk required):

```bash
pip install huggingface-hub  # or: pip install -e ".[convert]"
python3 convert_model.py
```

Both options create `models/Octen-Embedding-8B-mlx/` with the MLX weights.

### 3. Start the server

```bash
python3 server.py
```

The server starts on `http://127.0.0.1:8100` by default.

### macOS Auto-Start (launchd)

To run the server at login, create `~/Library/LaunchAgents/com.octen-embeddings.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.octen-embeddings</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/python3</string>
        <string>/path/to/octen-embeddings-server/server.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/octen-embeddings-server</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>MLX_MODEL_PATH</key>
        <string>/path/to/octen-embeddings-server/models/Octen-Embedding-8B-mlx</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>/tmp/octen-embeddings.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/octen-embeddings.log</string>
</dict>
</plist>
```

Then load it:

```bash
launchctl load ~/Library/LaunchAgents/com.octen-embeddings.plist
```

## API Reference

### POST /v1/embeddings

OpenAI-compatible embeddings endpoint.

**Request:**

```bash
curl http://localhost:8100/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "Octen/Octen-Embedding-8B"}'
```

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0123, -0.0456, ...],
      "index": 0
    }
  ],
  "model": "Octen/Octen-Embedding-8B",
  "usage": {
    "prompt_tokens": 3,
    "total_tokens": 3
  }
}
```

**Batch request** — pass an array of strings:

```bash
curl http://localhost:8100/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["text one", "text two", "text three"]}'
```

### GET /health

Returns server status, model info, memory usage, and uptime.

### GET /v1/models

OpenAI-compatible model listing.

### GET /metrics

Prometheus metrics endpoint. Exposes request counts, latency histograms, batch sizes, and token throughput.

### Legacy Endpoints

For backward compatibility with `qwen3-embeddings-mlx` consumers:

- `POST /embed` — single text embedding (`{"text": "..."}`)
- `POST /embed_batch` — batch embedding (`{"texts": ["...", "..."]}`)
- `GET /models` — model info

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Octen/Octen-Embedding-8B` | Model identifier returned in API responses |
| `MLX_MODEL_PATH` | `./models/Octen-Embedding-8B-mlx` | Path to converted MLX model weights |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8100` | Bind port |
| `MAX_TEXT_LENGTH` | `8192` | Maximum tokens per text (truncated if exceeded) |
| `MAX_BATCH_SIZE` | `256` | Maximum texts per batch request |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `API_KEY` | *(empty)* | API key for Bearer auth; leave empty to disable |
| `DEV_MODE` | `false` | Enable uvicorn auto-reload for development |

### Authentication

By default, no authentication is required (suitable for localhost). For non-localhost deployments, set the `API_KEY` environment variable:

```bash
API_KEY=sk-my-secret-key python3 server.py
```

Clients must then include a Bearer token:

```bash
curl http://host:8100/v1/embeddings \
  -H "Authorization: Bearer sk-my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world"}'
```

Health and metrics endpoints (`/health`, `/metrics`) do not require authentication.

## Performance

Embedding dimensions: **4096** (full precision, no quantization).

Typical latency on M-series Apple Silicon:
- Single short text (~10 tokens): ~50ms
- Single long text (~2000 tokens): ~500ms
- Batch of 10 short texts: ~400ms

First request after startup includes a warmup phase (~2-5s) for Metal kernel compilation.

Memory: the model loads ~16 GB into unified memory. The server process uses ~17 GB total at steady state.

## Benchmarking

A throughput benchmark script is included:

```bash
python3 benchmark.py                           # default (5 rounds)
python3 benchmark.py --rounds 10               # more rounds for accuracy
python3 benchmark.py --url http://host:8100    # custom server URL
python3 benchmark.py --api-key sk-xxx          # if auth is enabled
```

This measures latency and texts/sec across various text lengths and batch sizes.

## Development

```bash
pip install -e ".[dev]"
ruff check .                  # lint
ruff format --check .         # format check
pytest                        # unit tests
python3 validate.py           # integration tests (requires running server)
```

## Model Licensing

This server is MIT-licensed. However, the **Octen-Embedding-8B model weights** have their own license terms:

- **Octen-Embedding-8B** is published by [Octen AI](https://huggingface.co/Octen) on HuggingFace. Check the [model card](https://huggingface.co/Octen/Octen-Embedding-8B) for current license terms before use in production.
- The model is a fine-tune of **Qwen3**, which is released under the [Apache 2.0 license](https://huggingface.co/Qwen). Qwen3's license permits commercial use but requires attribution.
- The `convert_model.py` script downloads weights from HuggingFace; you are responsible for complying with the model's license when downloading and deploying.

**Summary**: This server code (MIT) and the model weights (separate license) have independent terms. Always verify the upstream model license for your use case.

## License

MIT — see [LICENSE](LICENSE).
