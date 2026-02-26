"""Unit tests for the embedding server.

These tests mock the model manager so they run without MLX model weights.
They verify API contracts, request validation, and response formats.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture(autouse=True)
def mock_model_manager():
    """Mock the model manager to return fake embeddings."""
    fake_embedding = np.random.randn(4096).astype(np.float32)
    fake_embedding /= np.linalg.norm(fake_embedding)

    def fake_embed(texts, **kwargs):
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


class TestAuthentication:
    """Test optional API key authentication."""

    def test_no_auth_required_by_default(self, client):
        resp = client.post("/v1/embeddings", json={"input": "test"})
        assert resp.status_code == 200

    def test_auth_required_when_api_key_set(self, client):
        with patch.object(server, "API_KEY", "test-key"):
            resp = client.post("/v1/embeddings", json={"input": "test"})
            assert resp.status_code == 401

    def test_auth_accepted_with_valid_key(self, client):
        with patch.object(server, "API_KEY", "test-key"):
            resp = client.post(
                "/v1/embeddings",
                json={"input": "test"},
                headers={"Authorization": "Bearer test-key"},
            )
            assert resp.status_code == 200

    def test_auth_rejected_with_wrong_key(self, client):
        with patch.object(server, "API_KEY", "test-key"):
            resp = client.post(
                "/v1/embeddings",
                json={"input": "test"},
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert resp.status_code == 401

    def test_health_no_auth_required(self, client):
        with patch.object(server, "API_KEY", "test-key"):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_metrics_no_auth_required(self, client):
        with patch.object(server, "API_KEY", "test-key"):
            resp = client.get("/metrics")
            assert resp.status_code == 200

    def test_legacy_embed_requires_auth(self, client):
        with patch.object(server, "API_KEY", "test-key"):
            resp = client.post("/embed", json={"text": "hello"})
            assert resp.status_code == 401

    def test_legacy_embed_batch_requires_auth(self, client):
        with patch.object(server, "API_KEY", "test-key"):
            resp = client.post("/embed_batch", json={"texts": ["hello"]})
            assert resp.status_code == 401


class TestGracefulShutdown:
    """Test shutdown middleware behavior."""

    def test_503_during_shutdown(self, client):
        client.app.state.shutting_down = True
        resp = client.post("/v1/embeddings", json={"input": "test"})
        assert resp.status_code == 503
        client.app.state.shutting_down = False

    def test_health_available_during_shutdown(self, client):
        client.app.state.shutting_down = True
        resp = client.get("/health")
        assert resp.status_code == 200
        client.app.state.shutting_down = False

    def test_metrics_available_during_shutdown(self, client):
        client.app.state.shutting_down = True
        resp = client.get("/metrics")
        assert resp.status_code == 200
        client.app.state.shutting_down = False


class TestLongTextTruncation:
    """Test that long texts are truncated to max token limit."""

    def _make_manager(self):
        """Create a ModelManager with a mock tokenizer and forward pass."""
        mgr = server.ModelManager()
        mgr.tokenizer = MagicMock()
        mgr.tokenizer.pad_token_id = 0
        mgr.ready = True
        return mgr

    def test_truncation_to_max_tokens(self):
        mgr = self._make_manager()
        # Simulate tokenizer returning 10000 tokens
        mgr.tokenizer.encode.return_value = list(range(10000))

        fake_emb = np.random.randn(1, 4096).astype(np.float32)
        with patch.object(mgr, "_embed_batch", return_value=fake_emb) as mock_batch:
            mgr.embed(["very long text"])
            # _embed_batch should receive tokens truncated to MAX_TOKENS (8192)
            called_tokens = mock_batch.call_args[0][0]
            assert len(called_tokens[0]) == server.MAX_TOKENS

    def test_truncation_to_custom_max_tokens(self):
        mgr = self._make_manager()
        mgr.tokenizer.encode.return_value = list(range(500))

        fake_emb = np.random.randn(1, 4096).astype(np.float32)
        with patch.object(mgr, "_embed_batch", return_value=fake_emb) as mock_batch:
            mgr.embed(["medium text"], max_tokens=100)
            called_tokens = mock_batch.call_args[0][0]
            assert len(called_tokens[0]) == 100

    def test_short_text_not_truncated(self):
        mgr = self._make_manager()
        mgr.tokenizer.encode.return_value = list(range(50))

        fake_emb = np.random.randn(1, 4096).astype(np.float32)
        with patch.object(mgr, "_embed_batch", return_value=fake_emb) as mock_batch:
            mgr.embed(["short"])
            called_tokens = mock_batch.call_args[0][0]
            assert len(called_tokens[0]) == 50

    def test_request_level_max_tokens(self, client):
        """The /v1/embeddings endpoint accepts max_tokens in the request body."""
        resp = client.post(
            "/v1/embeddings",
            json={"input": "test", "max_tokens": 512},
        )
        assert resp.status_code == 200
        # Verify embed was called with max_tokens kwarg
        server.manager.embed.assert_called_once()
        _, kwargs = server.manager.embed.call_args
        assert kwargs["max_tokens"] == 512


class TestBatchErrorResilience:
    """Test that a single failing item does not break the whole batch."""

    def _make_manager(self):
        mgr = server.ModelManager()
        mgr.tokenizer = MagicMock()
        mgr.tokenizer.pad_token_id = 0
        mgr.ready = True
        return mgr

    def test_batch_failure_falls_back_to_individual(self):
        mgr = self._make_manager()
        mgr.tokenizer.encode.side_effect = [
            list(range(10)),
            list(range(20)),
            list(range(15)),
        ]

        good_emb = np.random.randn(1, 4096).astype(np.float32)
        call_count = 0

        def selective_embed(tokens):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call is the batch — make it fail
                raise RuntimeError("OOM")
            return good_emb

        with patch.object(mgr, "_embed_batch", side_effect=selective_embed):
            result = mgr.embed(["a", "b", "c"])
            assert result.shape == (3, 4096)
            # 1 batch attempt + 3 individual attempts = 4 calls
            assert call_count == 4

    def test_individual_item_failure_returns_zero_vector(self):
        mgr = self._make_manager()
        mgr.tokenizer.encode.side_effect = [
            list(range(10)),
            list(range(20)),
        ]

        good_emb = np.ones((1, 4096), dtype=np.float32)
        call_count = 0

        def selective_embed(tokens):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Batch fails
                raise RuntimeError("OOM")
            if call_count == 3:
                # Second individual item fails
                raise RuntimeError("bad text")
            return good_emb

        with patch.object(mgr, "_embed_batch", side_effect=selective_embed):
            result = mgr.embed(["ok text", "bad text"])

        assert result.shape == (2, 4096)
        # First item should have real embedding
        assert np.all(result[0] == 1.0)
        # Second item should be zero vector
        assert np.all(result[1] == 0.0)

    def test_batch_success_no_fallback(self):
        mgr = self._make_manager()
        mgr.tokenizer.encode.side_effect = [list(range(10)), list(range(20))]

        batch_emb = np.ones((2, 4096), dtype=np.float32)
        with patch.object(mgr, "_embed_batch", return_value=batch_emb) as mock_batch:
            result = mgr.embed(["a", "b"])
            assert result.shape == (2, 4096)
            # Only one call — the batch succeeded
            assert mock_batch.call_count == 1

    def test_batch_error_via_endpoint(self, client):
        """Batch request with mixed results still returns 200."""
        resp = client.post(
            "/v1/embeddings",
            json={"input": ["good text", "another good text"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 2


class TestMaxTokensConfig:
    """Test MAX_TOKENS configuration controls truncation."""

    def test_server_max_tokens_controls_truncation(self):
        """Patching server.MAX_TOKENS limits token length in embed()."""
        mgr = server.ModelManager()
        mgr.tokenizer = MagicMock()
        mgr.tokenizer.pad_token_id = 0
        mgr.tokenizer.encode.return_value = list(range(200))
        mgr.ready = True

        fake_emb = np.random.randn(1, 4096).astype(np.float32)
        with (
            patch.object(server, "MAX_TOKENS", 50),
            patch.object(mgr, "_embed_batch", return_value=fake_emb) as mock_batch,
        ):
            mgr.embed(["long text"])
            called_tokens = mock_batch.call_args[0][0]
            assert len(called_tokens[0]) == 50

    def test_request_max_tokens_overrides_server_default(self):
        """Per-request max_tokens overrides the server-wide MAX_TOKENS."""
        mgr = server.ModelManager()
        mgr.tokenizer = MagicMock()
        mgr.tokenizer.pad_token_id = 0
        mgr.tokenizer.encode.return_value = list(range(200))
        mgr.ready = True

        fake_emb = np.random.randn(1, 4096).astype(np.float32)
        with (
            patch.object(server, "MAX_TOKENS", 8192),
            patch.object(mgr, "_embed_batch", return_value=fake_emb) as mock_batch,
        ):
            mgr.embed(["long text"], max_tokens=30)
            called_tokens = mock_batch.call_args[0][0]
            assert len(called_tokens[0]) == 30
