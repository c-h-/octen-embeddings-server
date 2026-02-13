#!/usr/bin/env python3
"""
Convert Octen-Embedding-8B to MLX format for use with mlx-lm.load().

The Octen model uses Qwen3Model architecture (encoder-only, no lm_head),
but mlx-lm expects Qwen3ForCausalLM (model.model.* + model.lm_head).
We fix this by:
  1. Prefixing all weight keys with "model."
  2. Adding a dummy lm_head (won't be used — we only need hidden states)
  3. Setting architectures to Qwen3ForCausalLM in config.json

This allows server.py to use mlx_lm.load() and access the transformer via
model.model.embed_tokens / model.model.layers / model.model.norm.
"""

import json
import os
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

MODEL_ID = "Octen/Octen-Embedding-8B"
OUTPUT_DIR = Path("models/Octen-Embedding-8B-mlx")


def main():
    print(f"Step 1: Downloading {MODEL_ID}...")
    local_path = Path(snapshot_download(MODEL_ID))
    print(f"  Downloaded to: {local_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy tokenizer files
    print("Step 2: Copying tokenizer files...")
    for name in os.listdir(local_path):
        if name.startswith("tokenizer") or name in ("special_tokens_map.json", "vocab.json", "merges.txt"):
            src = local_path / name
            if src.is_file():
                shutil.copy2(src, OUTPUT_DIR / name)
                print(f"  Copied {name}")

    # Load and prefix weights
    print("Step 3: Loading safetensors and adding 'model.' prefix...")
    safetensor_files = sorted(local_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {local_path}")

    all_weights = {}
    for sf_path in safetensor_files:
        print(f"  Loading {sf_path.name}...")
        weights = mx.load(str(sf_path))
        for key, value in weights.items():
            new_key = f"model.{key}"
            all_weights[new_key] = value

    # Add dummy lm_head (required by mlx-lm Model wrapper, but we never call it)
    # Use a small random weight that won't consume much memory
    print("Step 4: Adding dummy lm_head weight...")
    hidden_size = 4096
    vocab_size = 151665
    # Create a tiny placeholder — mlx-lm lazy loading means this is only loaded if accessed
    all_weights["lm_head.weight"] = mx.zeros((vocab_size, hidden_size), dtype=mx.bfloat16)

    # Save as MLX safetensors (sharded)
    print("Step 5: Saving MLX weights...")
    # Save in chunks to avoid memory issues
    shard_size = 5_000_000_000  # 5GB per shard
    shard_idx = 0
    current_shard = {}
    current_size = 0
    weight_map = {}

    for key, value in sorted(all_weights.items()):
        nbytes = value.nbytes
        if current_size > 0 and current_size + nbytes > shard_size:
            shard_name = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
            mx.save_safetensors(str(OUTPUT_DIR / shard_name), current_shard)
            print(f"  Saved {shard_name} ({current_size / 1e9:.2f} GB)")
            shard_idx += 1
            current_shard = {}
            current_size = 0
        current_shard[key] = value
        current_size += nbytes

    # Save last shard
    if current_shard:
        total_shards = shard_idx + 1
        shard_name = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
        mx.save_safetensors(str(OUTPUT_DIR / shard_name), current_shard)
        print(f"  Saved {shard_name} ({current_size / 1e9:.2f} GB)")

    # Rename shards with correct total count
    for i in range(total_shards):
        old_name = f"model-{i:05d}-of-PLACEHOLDER.safetensors"
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        (OUTPUT_DIR / old_name).rename(OUTPUT_DIR / new_name)
        # Build weight map
        shard_weights = mx.load(str(OUTPUT_DIR / new_name))
        for key in shard_weights.keys():
            weight_map[key] = new_name

    # Write weight map index
    index = {
        "metadata": {"total_size": sum(v.nbytes for v in all_weights.values())},
        "weight_map": weight_map,
    }
    with open(OUTPUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    print("  Saved model.safetensors.index.json")

    # Write config.json with Qwen3ForCausalLM architecture
    print("Step 6: Writing config.json...")
    with open(local_path / "config.json") as f:
        config = json.load(f)

    # Change architecture so mlx-lm recognizes it
    config["architectures"] = ["Qwen3ForCausalLM"]
    # Keep model_type as qwen3 (mlx-lm dispatches on this)

    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("  Saved config.json")

    # Verify we can load it
    print("\nStep 7: Verifying model loads...")
    from mlx_lm import load
    model, tokenizer = load(str(OUTPUT_DIR))
    print(f"  Model loaded successfully!")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Layers: {len(model.model.layers)}")

    # Quick embedding test
    tokens = tokenizer.encode("Hello world")
    input_ids = mx.array([tokens])
    h = model.model.embed_tokens(input_ids)
    for layer in model.model.layers:
        h = layer(h, mask=None, cache=None)
    h = model.model.norm(h)
    last_token = h[:, -1, :]
    norm = mx.linalg.norm(last_token, axis=1, keepdims=True)
    embedding = last_token / mx.maximum(norm, mx.array(1e-9))
    mx.eval(embedding)
    emb_np = np.array(embedding.tolist()[0])
    print(f"  Test embedding dim: {len(emb_np)}")
    print(f"  Test embedding norm: {np.linalg.norm(emb_np):.6f}")
    print(f"  First 5 values: {emb_np[:5]}")

    print(f"\nConversion complete! Model saved to: {OUTPUT_DIR}")
    print(f"Total size: {sum(f.stat().st_size for f in OUTPUT_DIR.glob('*')) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
