#!/usr/bin/env python3
"""Embed corpus chains with local ESM2-650M (mean-pool last layer).

Reads chains.csv from build_corpus.py output.
Writes:
  embeddings/<tag>/vectors.npy      # float32 [N, D]
  embeddings/<tag>/ids.txt          # chain_id per row
  embeddings/<tag>/meta.json
  embeddings/<tag>/embed_manifest.json

Smoke (--smoke) uses corpus_smoke/ → embeddings/smoke/
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import (  # noqa: E402
    BIO_ROOT,
    BatchProgress,
    ensure_dir,
    load_yaml,
    resolve_path,
    write_json,
)


def pick_device(pref: str) -> str:
    import torch

    if pref and pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_chains(csv_path: Path, *, split: str, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open() as f:
        for rec in csv.DictReader(f):
            if split != "all" and rec.get("split") != split:
                continue
            rows.append(rec)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def mean_pool(last_hidden, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="Alias for --smoke")
    ap.add_argument(
        "--split",
        choices=("train", "holdout", "all"),
        default="all",
        help="Which split to embed (default: all)",
    )
    ap.add_argument("--limit", type=int, default=None, help="Max chains (smoke default 32)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Override corpus directory containing chains.csv",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Override embeddings output directory",
    )
    args = ap.parse_args()
    smoke = bool(args.smoke or args.pilot)

    corpus_cfg = load_yaml("corpus.yaml")
    esm_cfg = load_yaml("esm2.yaml")

    if args.corpus_dir:
        corpus_dir = resolve_path(args.corpus_dir)
    else:
        corpus_dir = resolve_path(
            corpus_cfg["smoke_outdir"] if smoke else corpus_cfg["paper_outdir"]
        )
    csv_path = corpus_dir / "chains.csv"
    if not csv_path.exists():
        print(
            f"ERROR: missing {csv_path}. Run build_corpus.py"
            f"{' --smoke' if smoke else ''} first.",
            file=sys.stderr,
        )
        return 1

    if args.outdir:
        outdir = resolve_path(args.outdir)
    else:
        outdir = resolve_path("embeddings/smoke" if smoke else "embeddings/esm2_t33_650M")
    ensure_dir(outdir)

    limit = args.limit
    if smoke and limit is None:
        limit = 32

    rows = load_chains(csv_path, split=args.split, limit=limit)
    if not rows:
        print("ERROR: no chains to embed.", file=sys.stderr)
        return 1

    local_dir = resolve_path(esm_cfg.get("local_dir", "models/esm2_t33_650M_UR50D"))
    if not local_dir.exists():
        print(f"ERROR: ESM2 model missing at {local_dir}", file=sys.stderr)
        return 1

    device = pick_device(args.device or esm_cfg.get("device", "auto"))
    batch_size = int(args.batch_size or esm_cfg.get("batch_size", 8))
    max_len = int(esm_cfg.get("max_seq_len", 1024))
    pooling = esm_cfg.get("pooling", "mean")

    print(
        f"Embedding {len(rows)} chains | device={device} batch={batch_size} "
        f"max_len={max_len} pooling={pooling} → {outdir}"
    )

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(local_dir))
    model = AutoModel.from_pretrained(str(local_dir))
    model.eval()
    model.to(device)

    vectors: list[np.ndarray] = []
    ids: list[str] = []
    progress = BatchProgress(len(rows), label="embed-ESM2")
    t0 = time.time()

    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            seqs = [r["sequence"] for r in batch]
            enc = tokenizer(
                seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            hidden = out.last_hidden_state
            if pooling == "cls":
                emb = hidden[:, 0, :]
            else:
                emb = mean_pool(hidden, enc["attention_mask"])
            emb = emb.detach().float().cpu().numpy()
            for i, r in enumerate(batch):
                vectors.append(emb[i])
                ids.append(r["chain_id"])
                progress.tick(ok=True, force_print=(start + i + 1 >= len(rows)))

    mat = np.stack(vectors, axis=0).astype(np.float32)
    np.save(outdir / "vectors.npy", mat)
    (outdir / "ids.txt").write_text("\n".join(ids) + "\n")

    meta = {
        "model_id": esm_cfg.get("model_id"),
        "local_dir": str(local_dir.relative_to(BIO_ROOT)),
        "layer": esm_cfg.get("layer", "last"),
        "pooling": pooling,
        "max_seq_len": max_len,
        "device": device,
        "dim": int(mat.shape[1]),
        "n": int(mat.shape[0]),
        "split": args.split,
        "corpus_dir": str(corpus_dir.relative_to(BIO_ROOT)),
        "mode": "smoke" if smoke else "paper",
        "dtype": "float32",
    }
    write_json(outdir / "meta.json", meta)
    write_json(
        outdir / "embed_manifest.json",
        {
            **meta,
            "elapsed_s": round(time.time() - t0, 1),
            "vectors": str((outdir / "vectors.npy").relative_to(BIO_ROOT)),
            "ids": str((outdir / "ids.txt").relative_to(BIO_ROOT)),
        },
    )

    # Quick self-check: norms finite, no NaN
    if not np.isfinite(mat).all():
        print("ERROR: non-finite values in embeddings", file=sys.stderr)
        return 1
    norms = np.linalg.norm(mat, axis=1)
    print(
        f"Done: shape={mat.shape} | norm mean={norms.mean():.3f} "
        f"std={norms.std():.3f} | {outdir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
