# AGENTS.md

Developer reference for AI coding agents and contributors.

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python3 server.py              # start server (needs model weights)
python3 validate.py            # integration tests against running server
```

## Project Structure

```
server.py           # FastAPI embedding server — the main application
convert_model.py    # One-time model conversion from HuggingFace to MLX format
validate.py         # Integration test suite (runs against a live server)
tests/              # Unit tests (run without model or server)
pyproject.toml      # Package config, dependencies, tool settings
requirements.txt    # Pinned dependencies (mirrors pyproject.toml)
Dockerfile          # Container image (note: MLX needs Apple Silicon)
```

## How It Works

### MLX Model Loading

The server uses `mlx_lm.load()` to load a converted Octen-Embedding-8B model. The model is a Qwen3 fine-tune with an important twist:

1. **Architecture mismatch**: Octen publishes weights as `Qwen3Model` (encoder), but `mlx-lm` expects `Qwen3ForCausalLM`. The `convert_model.py` script bridges this by prefixing all weight keys with `model.` and adding a dummy `lm_head`.

2. **Causal attention mask**: Despite being used for embeddings, Octen was trained as a decoder with causal attention. The mask is **critical** — without it, cosine similarity drops from 0.82 to 0.23 for similar texts. See `ModelManager._forward()`.

3. **Last-token pooling**: Octen uses the last token's hidden state (not mean pooling). The embedding is L2-normalized after extraction.

### Embedding Pipeline

```
Input text → Tokenize → Token IDs → Embed tokens → Causal attention layers
→ Layer norm → Last token → L2 normalize → 4096-dim float32 vector
```

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/embeddings` | OpenAI-compatible embeddings (primary) |
| POST | `/embed` | Legacy single-text endpoint |
| POST | `/embed_batch` | Legacy batch endpoint |
| GET | `/health` | Server status and memory usage |
| GET | `/v1/models` | OpenAI-compatible model listing |
| GET | `/models` | Legacy model info |
| GET | `/metrics` | Prometheus metrics |

## Adding New Models

To add support for a different embedding model:

1. **Convert weights**: Modify `convert_model.py` to handle the new model's architecture. Key concerns:
   - Weight key prefixing (model-specific)
   - Architecture class expected by `mlx-lm`
   - Whether a dummy `lm_head` is needed

2. **Update pooling**: Different models use different pooling strategies. Modify `ModelManager._forward()`:
   - Last-token pooling (Octen, decoder-only models)
   - Mean pooling (most encoder models like BERT, Snowflake)
   - CLS token pooling (some BERT variants)

3. **Update attention**: Check whether the model needs causal masking or bidirectional attention in `_forward()`.

4. **Update dimensions**: Change `EMBEDDING_DIM` constant and update validation tests.

5. **Validate**: Run `python3 validate.py` and check cosine similarity scores. Similar texts should score > 0.7, dissimilar < 0.3.

## Testing

```bash
pytest                         # unit tests (no model needed)
ruff check .                   # lint
python3 validate.py            # integration tests (needs running server)
```

Unit tests use FastAPI's `TestClient` and mock the model manager. Integration tests in `validate.py` hit a live server and verify actual embedding quality.

## CI

GitHub Actions runs on push and PR to `main`:
1. Lint (`ruff check`)
2. Format check (`ruff format --check`)
3. Unit tests (`pytest`)

Integration tests and type checking are not in CI because they require the MLX model (Apple Silicon + 16 GB model weights).

## Git

- `main` branch — requires PR
- Commit messages: conventional commits (`feat:`, `fix:`, `docs:`, etc.)
