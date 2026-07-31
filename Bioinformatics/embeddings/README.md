# Embeddings

| Directory | Purpose |
|-----------|---------|
| `esm2_t33_650M/` | Paper corpus ESM2 vectors (`ids.txt`, `vectors.npy`) |
| `smoke/` | Smoke test embeds |
| `ablation/` | Layer/pooling variants (`layer33_mean/`, `layer16_mean/`, …) |

Produced by `scripts/embed_esm2.py` or `scripts/extract_ablation_embeddings.py`. Fingerprint checked at index time (`data/processed/xqdrant/` manifests).
