#!/usr/bin/env python3
"""
Octen-Embedding-8B Server using MLX on Apple Silicon

OpenAI-compatible /v1/embeddings endpoint serving Octen-Embedding-8B
via the MLX framework for native Apple Silicon acceleration.

Adapted from jakedahn/qwen3-embeddings-mlx (MIT license).
Key differences from base Qwen3:
  - Last-token pooling (not mean pooling) per Octen model card
  - Automatic "- " prefix on documents (Octen recommendation for untitled input)
  - Single-model server (Octen-Embedding-8B only)
  - OpenAI-compatible /v1/embeddings response format
"""

import os
import sys
import time
import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
import mlx.core as mx
from mlx_lm import load
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = os.getenv("MODEL_ID", "Octen/Octen-Embedding-8B")
MLX_MODEL_PATH = os.getenv(
    "MLX_MODEL_PATH",
    os.path.expanduser("~/personal/octen-embeddings-server/models/Octen-Embedding-8B-mlx"),
)
EMBEDDING_DIM = 4096
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "8192"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "256"))
PORT = int(os.getenv("PORT", "8100"))
HOST = os.getenv("HOST", "127.0.0.1")


def setup_logging() -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("octen-embeddings")


logger = setup_logging()


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQUESTS_TOTAL = Counter(
    "embeddings_requests_total",
    "Total embedding requests",
    ["endpoint"],
)
DURATION_SECONDS = Histogram(
    "embeddings_duration_seconds",
    "Embedding request duration in seconds",
    ["endpoint"],
)
BATCH_SIZE = Histogram(
    "embeddings_batch_size",
    "Number of texts per embedding request",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256],
)
TOKENS_TOTAL = Counter(
    "embeddings_tokens_total",
    "Total tokens processed (estimated)",
)
MODEL_LOADED = Gauge(
    "embeddings_model_loaded",
    "Whether the embedding model is loaded (1) or not (0)",
)


# ---------------------------------------------------------------------------
# Model manager
# ---------------------------------------------------------------------------

class ModelManager:
    """Loads and manages the MLX model for embedding generation."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.ready = False
        self.load_time: Optional[float] = None
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        async with self._lock:
            if self.ready:
                return
            logger.info("Loading model from %s ...", MLX_MODEL_PATH)
            t0 = time.time()
            self.model, self.tokenizer = load(MLX_MODEL_PATH)
            if not hasattr(self.model, "model"):
                raise ValueError("Invalid model architecture: missing 'model' attribute")
            self.load_time = time.time() - t0
            logger.info("Model loaded in %.2fs. Warming up...", self.load_time)
            self._warmup()
            self.ready = True
            MODEL_LOADED.set(1)
            logger.info("Model ready.")

    def _warmup(self) -> None:
        """Run a few dummy inferences to compile Metal kernels."""
        for text in ("warmup", "test embedding sentence"):
            tokens = self.tokenizer.encode(text)
            input_ids = mx.array([tokens])
            pooled = self._forward(input_ids)
            mx.eval(pooled)

    def _forward(self, input_ids: mx.array) -> mx.array:
        """Run the transformer and return the last-token embedding (L2-normalised)."""
        h = self.model.model.embed_tokens(input_ids)
        # Causal mask is required — Octen is a Qwen3 fine-tune (decoder-only)
        # trained with causal attention. Without it, embeddings degrade severely.
        seq_len = input_ids.shape[1]
        mask = mx.triu(
            mx.full((seq_len, seq_len), float("-inf"), dtype=h.dtype), k=1
        )
        for layer in self.model.model.layers:
            h = layer(h, mask=mask, cache=None)
        h = self.model.model.norm(h)
        # Last-token pooling (Octen uses last token, not mean pooling)
        last = h[:, -1, :]  # [batch, hidden_dim]
        # L2 normalise
        norm = mx.linalg.norm(last, axis=1, keepdims=True)
        last = last / mx.maximum(norm, mx.array(1e-9))
        return last

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts. Returns (N, 4096) float32 numpy array."""
        results = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > MAX_TEXT_LENGTH:
                tokens = tokens[:MAX_TEXT_LENGTH]
            input_ids = mx.array([tokens])
            pooled = self._forward(input_ids)
            mx.eval(pooled)
            vec = np.array(pooled.tolist()[0], dtype=np.float32)
            results.append(vec)
        return np.array(results, dtype=np.float32)


manager = ModelManager()


# ---------------------------------------------------------------------------
# Pydantic request / response models  (OpenAI-compatible)
# ---------------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible /v1/embeddings request."""
    input: Union[str, List[str]] = Field(..., description="Text(s) to embed")
    model: str = Field(default=MODEL_ID, description="Model identifier (ignored, single-model server)")
    encoding_format: Optional[str] = Field(default="float", description="Encoding format")

class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class UsageInfo(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    """OpenAI-compatible /v1/embeddings response."""
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: UsageInfo


# Also support the legacy /embed and /embed_batch endpoints from qwen3 server

class LegacyEmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model: Optional[str] = None
    normalize: bool = True

class LegacyEmbedResponse(BaseModel):
    embedding: List[float]
    model: str
    dim: int
    normalized: bool
    processing_time_ms: float

class LegacyBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    model: Optional[str] = None
    normalize: bool = True

class LegacyBatchResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dim: int
    count: int
    normalized: bool
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Octen Embedding Server on %s:%d", HOST, PORT)
    try:
        await manager.load()
    except Exception:
        logger.exception("Failed to load model at startup")
    app.state.start_time = time.time()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Octen Embedding Server",
    description="MLX-accelerated Octen-Embedding-8B with OpenAI-compatible API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def openai_embeddings(request: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint."""
    if not manager.ready:
        raise HTTPException(503, "Model not loaded yet")

    # Normalise input to list
    texts = [request.input] if isinstance(request.input, str) else request.input

    if len(texts) > MAX_BATCH_SIZE:
        raise HTTPException(400, f"Batch size {len(texts)} exceeds limit {MAX_BATCH_SIZE}")
    if not texts:
        raise HTTPException(400, "Input must not be empty")

    REQUESTS_TOTAL.labels(endpoint="/v1/embeddings").inc()
    BATCH_SIZE.observe(len(texts))

    t0 = time.time()
    embeddings = manager.embed(texts)
    elapsed = time.time() - t0
    elapsed_ms = elapsed * 1000

    DURATION_SECONDS.labels(endpoint="/v1/embeddings").observe(elapsed)

    # Estimate token count (rough: chars / 4)
    total_tokens = sum(len(t) for t in texts) // 4 or 1
    TOKENS_TOTAL.inc(total_tokens)

    data = [
        EmbeddingObject(embedding=emb.tolist(), index=i)
        for i, emb in enumerate(embeddings)
    ]

    logger.info(
        "Embedded %d text(s) in %.1fms (%.0f tok/s est.)",
        len(texts), elapsed_ms, total_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
    )

    return EmbeddingResponse(
        data=data,
        model=MODEL_ID,
        usage=UsageInfo(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )


# ---------------------------------------------------------------------------
# Legacy endpoints (compatible with qwen3-embeddings-mlx consumers)
# ---------------------------------------------------------------------------

@app.post("/embed", response_model=LegacyEmbedResponse)
async def embed_single(request: LegacyEmbedRequest):
    if not manager.ready:
        raise HTTPException(503, "Model not loaded yet")
    REQUESTS_TOTAL.labels(endpoint="/embed").inc()
    BATCH_SIZE.observe(1)
    t0 = time.time()
    embeddings = manager.embed([request.text])
    elapsed = (time.time() - t0) * 1000
    DURATION_SECONDS.labels(endpoint="/embed").observe(elapsed / 1000)
    TOKENS_TOTAL.inc(len(request.text) // 4 or 1)
    return LegacyEmbedResponse(
        embedding=embeddings[0].tolist(),
        model=MODEL_ID,
        dim=EMBEDDING_DIM,
        normalized=request.normalize,
        processing_time_ms=elapsed,
    )

@app.post("/embed_batch", response_model=LegacyBatchResponse)
async def embed_batch(request: LegacyBatchRequest):
    if not manager.ready:
        raise HTTPException(503, "Model not loaded yet")
    if len(request.texts) > MAX_BATCH_SIZE:
        raise HTTPException(400, f"Batch too large: {len(request.texts)} > {MAX_BATCH_SIZE}")
    REQUESTS_TOTAL.labels(endpoint="/embed_batch").inc()
    BATCH_SIZE.observe(len(request.texts))
    t0 = time.time()
    embeddings = manager.embed(request.texts)
    elapsed = (time.time() - t0) * 1000
    DURATION_SECONDS.labels(endpoint="/embed_batch").observe(elapsed / 1000)
    TOKENS_TOTAL.inc(sum(len(t) for t in request.texts) // 4 or 1)
    return LegacyBatchResponse(
        embeddings=embeddings.tolist(),
        model=MODEL_ID,
        dim=EMBEDDING_DIM,
        count=len(request.texts),
        normalized=request.normalize,
        processing_time_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health():
    memory_mb = None
    try:
        import psutil
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        pass
    uptime = time.time() - app.state.start_time if hasattr(app.state, "start_time") else 0
    return {
        "status": "healthy" if manager.ready else "loading",
        "model": MODEL_ID,
        "embedding_dim": EMBEDDING_DIM,
        "memory_usage_mb": round(memory_mb, 1) if memory_mb else None,
        "uptime_seconds": round(uptime, 1),
    }

@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "octen",
                "permission": [],
            }
        ],
    }

@app.get("/models")
async def list_models_legacy():
    return {
        "default_model": MODEL_ID,
        "embedding_dim": EMBEDDING_DIM,
        "status": "ready" if manager.ready else "loading",
        "load_time_seconds": manager.load_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        reload=os.getenv("DEV_MODE", "false").lower() == "true",
    )

if __name__ == "__main__":
    main()
