#!/usr/bin/env python3
"""Throughput benchmark for the Octen Embedding Server.

Measures requests/sec, texts/sec, and latency at various batch sizes.
Requires a running server (default: http://127.0.0.1:8100).

Usage:
    python benchmark.py                           # default settings
    python benchmark.py --url http://host:8100    # custom server URL
    python benchmark.py --rounds 10               # more rounds for accuracy
    python benchmark.py --api-key sk-xxx          # if auth is enabled
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

SAMPLE_TEXTS = {
    "short": "Machine learning on Apple Silicon.",
    "medium": (
        "The MLX framework provides efficient array operations on Apple Silicon, "
        "enabling local inference of large language models and embedding models "
        "without requiring cloud GPU resources."
    ),
    "long": (
        "Retrieval-Augmented Generation (RAG) is a technique that enhances large "
        "language model outputs by first retrieving relevant documents from a "
        "knowledge base using embedding similarity search, then providing those "
        "documents as context to the language model. This approach reduces "
        "hallucinations and allows the model to reference up-to-date or "
        "domain-specific information that was not present in its training data. "
        "Effective RAG systems depend on high-quality embedding models that can "
        "capture semantic similarity between queries and documents."
    ),
}

BATCH_SIZES = [1, 2, 4, 8, 16, 32]


def embed(url: str, texts: list[str], api_key: str | None = None) -> float:
    """Send an embedding request, return elapsed seconds."""
    body = json.dumps({"input": texts, "model": "benchmark"}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"{url}/v1/embeddings", data=body, headers=headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        resp.read()
    return time.perf_counter() - t0


def run_benchmark(url: str, rounds: int, api_key: str | None = None) -> None:
    # Verify server is reachable
    try:
        with urllib.request.urlopen(f"{url}/health") as resp:
            health = json.loads(resp.read())
        if health.get("status") != "healthy":
            print(f"Server not healthy: {health}")
            sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Cannot reach server at {url}: {exc}")
        sys.exit(1)

    print(f"Benchmarking {url}  ({rounds} rounds per configuration)\n")
    print(f"{'Text':>8}  {'Batch':>5}  {'Avg ms':>8}  {'p50 ms':>8}  {'p99 ms':>8}  {'texts/s':>8}")
    print("-" * 62)

    for text_label, text in SAMPLE_TEXTS.items():
        for batch_size in BATCH_SIZES:
            texts = [text] * batch_size
            # Warm-up
            embed(url, texts, api_key)

            latencies = []
            for _ in range(rounds):
                elapsed = embed(url, texts, api_key)
                latencies.append(elapsed)

            avg_ms = statistics.mean(latencies) * 1000
            p50_ms = statistics.median(latencies) * 1000
            p99_ms = sorted(latencies)[int(len(latencies) * 0.99)] * 1000
            texts_per_sec = batch_size / statistics.mean(latencies)

            print(
                f"{text_label:>8}  {batch_size:>5}  {avg_ms:>8.1f}  {p50_ms:>8.1f}"
                f"  {p99_ms:>8.1f}  {texts_per_sec:>8.1f}"
            )

    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark the Octen Embedding Server")
    parser.add_argument("--url", default="http://127.0.0.1:8100", help="Server URL")
    parser.add_argument("--rounds", type=int, default=5, help="Rounds per configuration")
    parser.add_argument("--api-key", default=None, help="API key (if auth is enabled)")
    args = parser.parse_args()
    run_benchmark(args.url, args.rounds, args.api_key)


if __name__ == "__main__":
    main()
