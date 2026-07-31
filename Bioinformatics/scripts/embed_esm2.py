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


def cuda_usable() -> tuple[bool, str]:
    """Return (usable, reason). PyTorch may report CUDA available but fail on newer GPUs."""
    import torch

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    try:
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
    except Exception as e:
        return False, f"cannot query GPU: {e}"
    try:
        x = torch.zeros(1, device="cuda")
        _ = (x + 1).item()
        torch.cuda.synchronize()
        return True, f"{name} (sm_{cap[0]}{cap[1]})"
    except RuntimeError as e:
        return False, f"{name} (sm_{cap[0]}{cap[1]}): {e}"


def pick_device(pref: str) -> str:
    import torch

    if pref and pref != "auto":
        if pref == "cuda":
            ok, reason = cuda_usable()
            if not ok:
                print(
                    f"WARN: --device cuda requested but GPU not usable ({reason}); using cpu",
                    file=sys.stderr,
                )
                return "cpu"
        return pref
    ok, reason = cuda_usable()
    if ok:
        print(f"Using CUDA: {reason}", file=sys.stderr)
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available() and not ok:
        print(
            f"WARN: CUDA visible but not usable ({reason}); falling back to cpu",
            file=sys.stderr,
        )
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


def resolve_layer_index(layer_spec: str | int, n_hidden: int) -> int:
    """Map layer spec to index into model hidden_states tuple."""
    if str(layer_spec).lower() in ("last", "final"):
        return n_hidden - 1
    idx = int(layer_spec)
    if idx < 0:
        idx = n_hidden + idx
    return max(0, min(idx, n_hidden - 1))


def pool_hidden(hidden, attention_mask, pooling: str):
    import torch

    if pooling == "cls":
        return hidden[:, 0, :]
    if pooling == "mean":
        return mean_pool(hidden, attention_mask)
    raise ValueError(f"Unknown pooling: {pooling}")


def load_chain_ids_file(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def load_chains_by_ids(csv_path: Path, chain_ids: list[str]) -> list[dict]:
    want = set(chain_ids)
    rows: list[dict] = []
    with csv_path.open() as f:
        for rec in csv.DictReader(f):
            if rec["chain_id"] in want:
                rows.append(rec)
    missing = want - {r["chain_id"] for r in rows}
    if missing:
        print(f"WARN: {len(missing)} chain_ids missing from {csv_path}", file=sys.stderr)
    return rows


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
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max chains (default: all rows in the chosen corpus/split)",
    )
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
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append embeddings for corpus chains not already in outdir/ids.txt",
    )
    ap.add_argument("--layer", default=None, help="Hidden layer index or 'last' (default: esm2.yaml)")
    ap.add_argument("--pooling", default=None, choices=("mean", "cls"))
    ap.add_argument(
        "--chains-file",
        type=Path,
        default=None,
        help="Embed only these chain_ids (one per line)",
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

    rows = load_chains(csv_path, split=args.split, limit=args.limit)
    if args.chains_file:
        cf = resolve_path(args.chains_file)
        ids_want = load_chain_ids_file(cf)
        rows = load_chains_by_ids(csv_path, ids_want)
        print(f"Chains file: {len(ids_want)} requested → {len(rows)} found in corpus")

    existing_ids: list[str] = []
    existing_mat: np.ndarray | None = None
    if args.append:
        ids_path = outdir / "ids.txt"
        vec_path = outdir / "vectors.npy"
        if not ids_path.exists() or not vec_path.exists():
            print(
                f"ERROR: --append requires existing {ids_path} and {vec_path}",
                file=sys.stderr,
            )
            return 1
        existing_ids = [
            ln.strip() for ln in ids_path.read_text().splitlines() if ln.strip()
        ]
        existing_mat = np.load(vec_path)
        if existing_mat.shape[0] != len(existing_ids):
            print(
                f"ERROR: vectors rows {existing_mat.shape[0]} != ids {len(existing_ids)}",
                file=sys.stderr,
            )
            return 1
        have = set(existing_ids)
        rows = [r for r in rows if r["chain_id"] not in have]
        print(f"Append mode: {len(rows)} new chains (existing {len(existing_ids)})")

    if not rows:
        print("Nothing to embed.")
        return 0

    local_dir = resolve_path(esm_cfg.get("local_dir", "models/esm2_t33_650M_UR50D"))
    if not local_dir.exists():
        print(f"ERROR: ESM2 model missing at {local_dir}", file=sys.stderr)
        return 1

    device = pick_device(args.device or esm_cfg.get("device", "auto"))
    batch_size = int(args.batch_size or esm_cfg.get("batch_size", 8))
    max_len = int(esm_cfg.get("max_seq_len", 1024))
    pooling = args.pooling or esm_cfg.get("pooling", "mean")
    layer_spec = args.layer if args.layer is not None else esm_cfg.get("layer", "last")

    print(
        f"Embedding {len(rows)} chains | device={device} batch={batch_size} "
        f"max_len={max_len} layer={layer_spec} pooling={pooling} → {outdir}"
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
            out = model(**enc, output_hidden_states=True)
            layer_idx = resolve_layer_index(layer_spec, len(out.hidden_states))
            hidden = out.hidden_states[layer_idx]
            emb = pool_hidden(hidden, enc["attention_mask"], pooling)
            emb = emb.detach().float().cpu().numpy()
            for i, r in enumerate(batch):
                vectors.append(emb[i])
                ids.append(r["chain_id"])
                progress.tick(ok=True, force_print=(start + i + 1 >= len(rows)))

    mat = np.stack(vectors, axis=0).astype(np.float32)
    if args.append and existing_mat is not None:
        mat = np.concatenate([existing_mat, mat], axis=0)
        ids = existing_ids + ids
    np.save(outdir / "vectors.npy", mat)
    (outdir / "ids.txt").write_text("\n".join(ids) + "\n")

    meta = {
        "model_id": esm_cfg.get("model_id"),
        "local_dir": str(local_dir.relative_to(BIO_ROOT)),
        "layer": str(layer_spec),
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
