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

### 2. Download and convert the model

The model must be converted from HuggingFace format to MLX format. This downloads ~16 GB and requires ~32 GB free disk temporarily.

```bash
pip install huggingface-hub  # or: pip install -e ".[convert]"
python3 convert_model.py
```

This creates `models/Octen-Embedding-8B-mlx/` with the converted weights.

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
| `DEV_MODE` | `false` | Enable uvicorn auto-reload for development |

## Performance

Embedding dimensions: **4096** (full precision, no quantization).

Typical latency on M-series Apple Silicon:
- Single short text (~10 tokens): ~50ms
- Single long text (~2000 tokens): ~500ms
- Batch of 10 short texts: ~400ms

First request after startup includes a warmup phase (~2-5s) for Metal kernel compilation.

Memory: the model loads ~16 GB into unified memory. The server process uses ~17 GB total at steady state.

## Development

```bash
pip install -e ".[dev]"
ruff check .                  # lint
ruff format --check .         # format check
pytest                        # unit tests
python3 validate.py           # integration tests (requires running server)
```

## License

MIT
