#!/usr/bin/env python3
"""
Validation script for Octen-Embedding-8B MLX server.

Tests:
  1. Health endpoint responds
  2. Single embedding returns correct dimensions
  3. Batch embedding works
  4. OpenAI-compatible format is correct
  5. Cosine similarity sanity check (similar texts score high, dissimilar low)
  6. /v1/models endpoint works

Usage:
  python3 validate.py [--url http://localhost:8100]
"""

import argparse
import json
import sys
import urllib.request


def fetch(url, data=None):
    """Simple HTTP request helper."""
    req = urllib.request.Request(url)
    if data:
        req.add_header("Content-Type", "application/json")
        body = json.dumps(data).encode()
    else:
        body = None
    with urllib.request.urlopen(req, body, timeout=120) as resp:
        return json.loads(resp.read())


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8100")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}: {detail}")
            failed += 1

    print(f"\nValidating Octen Embedding Server at {base}\n")

    # 1. Health
    print("--- Health ---")
    try:
        h = fetch(f"{base}/health")
        check("health responds", h.get("status") in ("healthy", "loading"))
        check("model field present", h.get("model") is not None)
        check("embedding_dim is 4096", h.get("embedding_dim") == 4096)
    except Exception as e:
        check("health endpoint", False, str(e))

    # 2. Single embedding via OpenAI endpoint
    print("\n--- Single Embedding (/v1/embeddings) ---")
    try:
        r = fetch(f"{base}/v1/embeddings", {"input": "Hello world", "model": "Octen/Octen-Embedding-8B"})
        check("object is 'list'", r.get("object") == "list")
        check("data has 1 item", len(r.get("data", [])) == 1)
        emb = r["data"][0]["embedding"]
        check("dimension is 4096", len(emb) == 4096, f"got {len(emb)}")
        check("values are floats", isinstance(emb[0], float))
        check("usage present", "usage" in r)
    except Exception as e:
        check("single embedding", False, str(e))

    # 3. Batch embedding
    print("\n--- Batch Embedding (/v1/embeddings) ---")
    texts = ["The cat sat on the mat", "Dogs are loyal animals", "Quantum mechanics is complex"]
    try:
        r = fetch(f"{base}/v1/embeddings", {"input": texts})
        check("batch returns 3 items", len(r.get("data", [])) == 3)
        dims = [len(d["embedding"]) for d in r["data"]]
        check("all dims are 4096", all(d == 4096 for d in dims), f"got {dims}")
    except Exception as e:
        check("batch embedding", False, str(e))

    # 4. Cosine similarity sanity
    print("\n--- Cosine Similarity Sanity ---")
    try:
        all_texts = [
            "The weather is sunny today",  # 0
            "It's a beautiful sunny day",  # 1 - similar to 0
            "Quantum entanglement in photons",  # 2 - dissimilar to 0
        ]
        r = fetch(f"{base}/v1/embeddings", {"input": all_texts})
        vecs = [d["embedding"] for d in sorted(r["data"], key=lambda x: x["index"])]

        sim_high = cosine_sim(vecs[0], vecs[1])
        sim_low = cosine_sim(vecs[0], vecs[2])

        check(f"similar texts sim={sim_high:.4f} > 0.7", sim_high > 0.7, f"got {sim_high:.4f}")
        check(f"dissimilar texts sim={sim_low:.4f} < similar", sim_low < sim_high, f"sim_low={sim_low:.4f}")
    except Exception as e:
        check("cosine similarity", False, str(e))

    # 5. /v1/models
    print("\n--- Models Endpoint ---")
    try:
        r = fetch(f"{base}/v1/models")
        check("object is 'list'", r.get("object") == "list")
        check("has model data", len(r.get("data", [])) > 0)
        model_id = r["data"][0].get("id", "")
        check("model id contains 'Octen'", "Octen" in model_id, f"got '{model_id}'")
    except Exception as e:
        check("/v1/models", False, str(e))

    # 6. Legacy endpoints
    print("\n--- Legacy Endpoints ---")
    try:
        r = fetch(f"{base}/embed", {"text": "test legacy endpoint"})
        check("/embed works", "embedding" in r)
        check("/embed dim=4096", r.get("dim") == 4096)
    except Exception as e:
        check("/embed", False, str(e))

    try:
        r = fetch(f"{base}/embed_batch", {"texts": ["a", "b"]})
        check("/embed_batch works", "embeddings" in r)
        check("/embed_batch count=2", r.get("count") == 2)
    except Exception as e:
        check("/embed_batch", False, str(e))

    # Summary
    total = passed + failed
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
