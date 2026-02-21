"""Unit tests for the embedding server.

These tests mock the model manager so they run without MLX model weights.
They verify API contracts, request validation, and response formats.
"""

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture(autouse=True)
def mock_model_manager():
    """Mock the model manager to return fake embeddings."""
    fake_embedding = np.random.randn(4096).astype(np.float32)
    fake_embedding /= np.linalg.norm(fake_embedding)

    def fake_embed(texts):
        return np.array([fake_embedding] * len(texts), dtype=np.float32)

    with (
        patch.object(server.manager, "ready", True),
        patch.object(server.manager, "embed", side_effect=fake_embed),
        patch.object(server.manager, "load_time", 1.0),
    ):
        yield


@pytest.fixture
def client():
    return TestClient(server.app)


class TestOpenAIEmbeddings:
    def test_single_text(self, client):
        resp = client.post("/v1/embeddings", json={"input": "hello", "model": "test"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert body["data"][0]["object"] == "embedding"
        assert len(body["data"][0]["embedding"]) == 4096
        assert body["data"][0]["index"] == 0
        assert "usage" in body
        assert body["usage"]["prompt_tokens"] > 0

    def test_batch_text(self, client):
        resp = client.post("/v1/embeddings", json={"input": ["a", "b", "c"]})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 3
        for i, item in enumerate(body["data"]):
            assert item["index"] == i
            assert len(item["embedding"]) == 4096

    def test_empty_input_rejected(self, client):
        resp = client.post("/v1/embeddings", json={"input": []})
        assert resp.status_code == 400

    def test_batch_too_large(self, client):
        texts = [f"text {i}" for i in range(257)]
        resp = client.post("/v1/embeddings", json={"input": texts})
        assert resp.status_code == 400

    def test_model_field_in_response(self, client):
        resp = client.post("/v1/embeddings", json={"input": "test"})
        body = resp.json()
        assert "model" in body
        assert "Octen" in body["model"]


class TestLegacyEndpoints:
    def test_embed_single(self, client):
        resp = client.post("/embed", json={"text": "hello"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["embedding"]) == 4096
        assert body["dim"] == 4096
        assert body["model"] == "Octen/Octen-Embedding-8B"

    def test_embed_batch(self, client):
        resp = client.post("/embed_batch", json={"texts": ["a", "b"]})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["embeddings"]) == 2
        assert body["count"] == 2
        assert body["dim"] == 4096


class TestMonitoring:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["embedding_dim"] == 4096

    def test_v1_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert "Octen" in body["data"][0]["id"]

    def test_legacy_models(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["embedding_dim"] == 4096

    def test_metrics(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert b"embeddings_requests_total" in resp.content


class TestModelNotReady:
    """Test behavior when model is not loaded."""

    def test_503_when_not_ready(self, client):
        with patch.object(server.manager, "ready", False):
            resp = client.post("/v1/embeddings", json={"input": "test"})
            assert resp.status_code == 503
