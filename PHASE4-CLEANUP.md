# Phase 4: Cleanup — Files to Delete

**DO NOT DELETE YET** — review after Phases 2+3 are verified working.

## Files to Remove

| Path | Size | Reason |
|------|------|--------|
| `~/.openclaw/models/Qwen3-Embedding-8B-Q8_0.gguf` | 7.5 GB | Replaced by Octen-Embedding-8B MLX server |
| `~/personal/retrieval-skill/models/Snowflake/` | 1.2 GB | Replaced by Octen-Embedding-8B MLX server |
| `~/personal/skills/retrieve/models/` (BAAI/ + Xenova/) | 316 MB | Old prototype BGE-small, never deployed |

**Total recoverable:** ~9.0 GB

## Commands to Run (when ready)

```bash
# 1. Remove old GGUF model (after Phase 2 verified)
rm ~/.openclaw/models/Qwen3-Embedding-8B-Q8_0.gguf

# 2. Remove Snowflake ONNX model (after Phase 3 verified)
rm -rf ~/personal/retrieval-skill/models/Snowflake/

# 3. Remove old prototype models
rm -rf ~/personal/skills/retrieve/models/

# 4. Remove @huggingface/transformers from retrieval-skill
cd ~/personal/retrieval-skill && npm uninstall @huggingface/transformers
```

## Also Consider

- Delete existing retrieval-skill `.db` indexes (they contain 1024-dim Snowflake embeddings, incompatible with 4096-dim Octen)
- Update `retrieval-skill-implementation.md` spec with new model info
- Update `TOOLS.md` to document the embedding server
