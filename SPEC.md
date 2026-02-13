# Embedding Consolidation: Octen-Embedding-8B on MLX

**Project:** Consolidate all local embeddings on Octen-Embedding-8B via MLX  
**Repo:** `~/personal/octen-embeddings-server` (new, private)  
**Date:** 2026-02-13  
**Status:** 🟡 Planning  
**Owner:** Jarvis (personal agent)

---

## Problem

We currently run **three different embedding models** across two systems, none optimal:

| System | Model | Dims | RTEB Score | Runtime | Issue |
|--------|-------|------|------------|---------|-------|
| OpenClaw `memory_search` | Qwen3-Embedding-8B (GGUF) | 2048 | 0.7547 | node-llama-cpp | GGUF quantization degrades embedding quality ~50% |
| Retrieval skill (indexer) | Snowflake Arctic Embed L (ONNX) | 1024 | 0.6150 | @huggingface/transformers | Outdated model, poor retrieval scores |
| Retrieval skill (spec target) | BGE-M3 (ONNX) | 1024 | 0.5893 | Never deployed | Even worse than Snowflake |

## Solution

Consolidate on **Octen-Embedding-8B** — #1 on RTEB (0.8045), fine-tuned from Qwen3-Embedding-8B, Apache 2.0 — running via **MLX** for native Apple Silicon acceleration.

**Single embedding server → two consumers:**

```
┌──────────────────────┐
│  Octen-Embedding-8B  │
│   MLX Server :8100   │
│  (OpenAI-compatible) │
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
  OpenClaw    Retrieval
  memory_     Skill
  search      (indexer)
```

---

## Model Selection

| Model | Params | RTEB Score | Architecture | MLX Support |
|-------|--------|------------|--------------|-------------|
| **Octen-Embedding-8B** | 8B | **0.8045** (#1) | Qwen3 (fine-tune) | ✅ Qwen3 arch supported by mlx-embeddings |
| Qwen3-Embedding-8B | 8B | 0.7547 | Qwen3 | ✅ mlx-community conversions exist |
| voyage-3-large | — | 0.7812 | Proprietary | ❌ Cloud-only |
| Snowflake Arctic Embed L | 335M | 0.6150 | Custom | ONNX only |

**Decision:** Octen-Embedding-8B. +6.6% over Qwen3, +30.8% over Snowflake. Same Qwen3 architecture = MLX compatible.

---

## Phases

### Phase 1: MLX Embedding Server ← START HERE

**Goal:** Get Octen-Embedding-8B running locally via MLX, exposed as an OpenAI-compatible `/v1/embeddings` endpoint.

**Repo:** `~/personal/octen-embeddings-server`

**Approach:** Fork/adapt `jakedahn/qwen3-embeddings-mlx` (MIT license, Python, FastAPI). That project already runs Qwen3-Embedding on MLX with REST API. Since Octen is a Qwen3 fine-tune, it should work with minimal changes.

**Tasks:**
- [ ] **P1.1:** Clone qwen3-embeddings-mlx, adapt for Octen-Embedding-8B
- [ ] **P1.2:** Convert Octen-Embedding-8B to MLX format using `mlx-lm` (full precision, NO quantization — quality is king, RAM is free)
- [ ] **P1.3:** Verify embeddings match sentence-transformers reference output (cosine sim > 0.99)
- [ ] **P1.4:** Expose OpenAI-compatible `/v1/embeddings` endpoint (model name, dimensions in response)
- [ ] **P1.5:** Add health check, batch support, basic metrics
- [ ] **P1.6:** Test throughput on Mac Studio (target: >5K tokens/sec)
- [ ] **P1.7:** Create launchd plist for auto-start on boot (`com.openclaw.octen-embeddings.plist`)

**Port:** `8100` (avoid conflicts with existing services)

**Validation:** Compare cosine similarity of 100 test queries between:
- Octen via sentence-transformers (PyTorch, ground truth)
- Octen via MLX server (our build)
- Must achieve >0.99 cosine similarity on all pairs

**Deliverable:** `http://localhost:8100/v1/embeddings` accepting OpenAI-format requests, returning embeddings.

**Estimated effort:** 1 day

---

### Phase 2: OpenClaw Memory Search Migration

**Goal:** Point OpenClaw's `memory_search` at the MLX server instead of the GGUF model.

**Approach:** OpenClaw supports `provider: "openai"` with custom `remote.baseUrl`. We point it at our local MLX server.

**Config change:**
```json5
agents: {
  defaults: {
    memorySearch: {
      provider: "openai",
      model: "Octen/Octen-Embedding-8B",
      remote: {
        baseUrl: "http://localhost:8100/v1/",
        apiKey: "local"  // server doesn't need auth, but field is required
      },
      local: {
        // Remove or leave — won't be used with provider: "openai"
      }
    }
  }
}
```

**What happens automatically:**
- OpenClaw detects provider/model change → **auto-reindexes all memory** (documented behavior)
- No manual reindexing needed
- All agents get the new embeddings on next `memory_search` call

**Tasks:**
- [ ] **P2.1:** Verify MLX server returns OpenAI-compatible response format
- [ ] **P2.2:** Apply config change via `gateway config.patch`
- [ ] **P2.3:** Verify auto-reindex triggers for all agents
- [ ] **P2.4:** Test `memory_search` quality — run 10 known queries, compare relevance before/after
- [ ] **P2.5:** Delete old GGUF model (`~/.openclaw/models/Qwen3-Embedding-8B-Q8_0.gguf`, 8GB)

**Rollback:** Revert config to `provider: "local"` + GGUF path. Auto-reindex back.

**Estimated effort:** 2 hours

---

### Phase 3: Retrieval Skill Migration

**Goal:** Replace Snowflake Arctic Embed L in `~/personal/retrieval-skill` with calls to the MLX server.

**Current architecture:** `embedder.mjs` loads ONNX model via `@huggingface/transformers` pipeline, generates embeddings in-process.

**New architecture:** `embedder.mjs` calls `http://localhost:8100/v1/embeddings` via fetch. Much simpler code.

**Tasks:**
- [ ] **P3.1:** Rewrite `src/embedder.mjs`:
  - Replace ONNX pipeline with HTTP calls to MLX server
  - Update `EMBEDDING_DIM` from 1024 → 4096
  - Update query prefix handling (Octen may use instruction-aware prompts)
  - Keep batch support (server supports `/embed_batch`)
- [ ] **P3.2:** Update `src/schema.mjs` — vector column size changes (1024 → 4096)
- [ ] **P3.3:** Delete all existing indexes (they're Snowflake 1024-dim, incompatible)
  - `rm ~/personal/retrieval-skill/*.db` (or wherever SQLite DBs live)
- [ ] **P3.4:** Re-index test corpus with new embeddings
- [ ] **P3.5:** Run existing tests, fix any dimension-related breakage
- [ ] **P3.6:** Remove `@huggingface/transformers` dependency from package.json
- [ ] **P3.7:** Remove downloaded Snowflake model files from `models/Snowflake/`
- [ ] **P3.8:** Update SPEC.md with new model info

**Breaking change:** All existing indexes become invalid. This is fine — Charlie confirmed we nuke and regenerate.

**Estimated effort:** Half day

---

### Phase 4: Cleanup & Docs

- [ ] **P4.1:** Delete old models:
  - `~/.openclaw/models/Qwen3-Embedding-8B-Q8_0.gguf` (8GB)
  - `~/personal/retrieval-skill/models/Snowflake/` 
  - `~/personal/skills/retrieve/models/` (old prototype BGE-small)
- [ ] **P4.2:** Update `retrieval-skill-implementation.md` spec (mark model decision updated)
- [ ] **P4.3:** Update `TOOLS.md` — document the embedding server
- [ ] **P4.4:** Add Tailscale serve entry if we want remote access (probably not needed)
- [ ] **P4.5:** Update this spec with final status

**Estimated effort:** 1 hour

---

## Infrastructure

### Embedding Server (launchd)

```
Service: com.openclaw.octen-embeddings
Port: 8100
Binary: python3 ~/personal/octen-embeddings-server/server.py
Model: Octen/Octen-Embedding-8B (MLX format, full precision)
RAM: ~15-17GB (trivial on 192GB machine)
Auto-start: Yes (launchd plist)
```

### Dependencies

| Dependency | Used By | Notes |
|------------|---------|-------|
| `mlx` | Server | Apple's ML framework |
| `mlx-lm` | Conversion | One-time model conversion |
| `mlx-embeddings` | Server | Qwen3 arch support |
| `fastapi` + `uvicorn` | Server | HTTP API |

### Disk Space

| Item | Size | Action |
|------|------|--------|
| Octen-8B MLX weights (new) | ~15GB | Download + convert |
| Qwen3-8B GGUF (remove) | 8GB | Delete after Phase 2 |
| Snowflake ONNX (remove) | ~1GB | Delete after Phase 3 |
| BGE-small ONNX (remove) | ~130MB | Delete in Phase 4 |
| **Net change** | +6GB | Acceptable |

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| MLX conversion produces bad embeddings | Medium | Phase 1 validation: cosine sim >0.99 vs PyTorch reference |
| Octen Qwen3 fine-tune has unsupported ops in MLX | Low | Same base arch as Qwen3 which already works in MLX. Fallback: sentence-transformers + MPS |
| Embedding server crashes/restarts | Low | launchd auto-restart + OpenClaw fallback config |
| 4096-dim embeddings are too large for SQLite | Low | sqlite-vec handles arbitrary dims; just more disk |
| OpenClaw rejects custom embedding endpoint | Low | Docs confirm `provider: "openai"` + custom `baseUrl` is supported |

**Fallback plan:** If MLX doesn't work, run via `sentence-transformers` + PyTorch MPS. Slower but guaranteed to work (Octen's official usage example). Same REST API wrapper, just different backend.

---

## Success Criteria

1. ✅ Single embedding model (Octen-Embedding-8B) serving all use cases
2. ✅ Running on MLX for native Apple Silicon performance
3. ✅ OpenClaw `memory_search` using new embeddings (auto-reindexed)
4. ✅ Retrieval skill using new embeddings (re-indexed)
5. ✅ Old models deleted, no more model sprawl
6. ✅ Auto-start on boot, no manual intervention needed
7. ✅ Measurable improvement in search relevance

---

## Timeline

| Phase | Effort | Dependencies |
|-------|--------|-------------|
| Phase 1: MLX Server | 1 day | None |
| Phase 2: OpenClaw Migration | 2 hours | Phase 1 |
| Phase 3: Retrieval Skill Migration | Half day | Phase 1 |
| Phase 4: Cleanup | 1 hour | Phases 2+3 |
| **Total** | **~2 days** | |
