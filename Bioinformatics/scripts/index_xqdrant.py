#!/usr/bin/env python3
"""Upsert ESM2 embeddings into a local XQdrant collection (REST).

Requires a running XQdrant fork server (./start_xqdrant.sh).

Artifacts (under data/processed/xqdrant/):
  <collection>_manifest.json
  <collection>_id_map.tsv          # point_id \\t chain_id
  <collection>_checkpoint.json     # resume cursor (next_index)

Resume: continues from checkpoint unless --restart / --recreate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
from xqdrant_rest import XQdrantREST  # noqa: E402


def load_payload_map(chains_csv: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with chains_csv.open() as f:
        for rec in csv.DictReader(f):
            out[rec["chain_id"]] = {
                "chain_id": rec["chain_id"],
                "pdb_id": rec["pdb_id"],
                "chain": rec["chain"],
                "split": rec["split"],
                "scop_domain": rec.get("scop_domain", ""),
                "scop_sccs": rec.get("scop_sccs", ""),
                "scop_fold": rec.get("scop_fold", ""),
                "scop_class": rec.get("scop_class", ""),
                "uniprot": rec.get("uniprot", ""),
                "length": int(rec["length"]) if rec.get("length") else 0,
            }
    return out


def file_sha256(path: Path, *, max_bytes: int = 8_000_000) -> str:
    """Hash file (full if small; else size+head+tail fingerprint)."""
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(f"size={size}\n".encode())
    with path.open("rb") as f:
        if size <= max_bytes:
            h.update(f.read())
        else:
            h.update(f.read(1_000_000))
            f.seek(max(size - 1_000_000, 0))
            h.update(f.read(1_000_000))
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--recreate",
        "--restart",
        action="store_true",
        dest="restart",
        help="Drop collection + clear checkpoint and start over",
    )
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--url", default=None)
    ap.add_argument("--collection", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Upsert only first N (debug)")
    args = ap.parse_args()

    xcfg = load_yaml("xqdrant.yaml")
    corpus_cfg = load_yaml("corpus.yaml")
    smoke = bool(args.smoke)

    url = args.url or xcfg.get("url", "http://127.0.0.1:6333")
    coll = args.collection or (
        xcfg["smoke_collection"] if smoke else xcfg["collection"]
    )
    emb_dir = resolve_path("embeddings/smoke" if smoke else "embeddings/esm2_t33_650M")
    corpus_dir = resolve_path(
        corpus_cfg["smoke_outdir"] if smoke else corpus_cfg["paper_outdir"]
    )
    chains_csv = corpus_dir / "chains.csv"
    vec_path = emb_dir / "vectors.npy"
    ids_path = emb_dir / "ids.txt"
    meta_path = emb_dir / "meta.json"

    for p in (vec_path, ids_path, chains_csv):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    ids = [ln.strip() for ln in ids_path.read_text().splitlines() if ln.strip()]
    vectors = np.load(vec_path)
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        print(
            f"ERROR: vectors shape {vectors.shape} vs {len(ids)} ids",
            file=sys.stderr,
        )
        return 1

    dim = int(xcfg.get("dim", vectors.shape[1]))
    if vectors.shape[1] != dim:
        print(f"ERROR: expected dim={dim}, got {vectors.shape[1]}", file=sys.stderr)
        return 1

    if args.limit is not None:
        ids = ids[: args.limit]
        vectors = vectors[: args.limit]

    payload_map = load_payload_map(chains_csv)
    missing_payload = sum(1 for c in ids if c not in payload_map)
    if missing_payload:
        print(
            f"WARN: {missing_payload} embedding ids missing from chains.csv "
            "(payload will be minimal)",
            file=sys.stderr,
        )

    out_dir = ensure_dir(resolve_path("data/processed/xqdrant"))
    ckpt_path = out_dir / f"{coll}_checkpoint.json"
    id_map_path = out_dir / f"{coll}_id_map.tsv"
    manifest_path = out_dir / f"{coll}_manifest.json"

    client = XQdrantREST(url)
    try:
        client.wait_ready(retries=10, sleep_s=0.3)
    except RuntimeError as e:
        print(
            f"ERROR: {e}\nStart the server first: ./start_xqdrant.sh",
            file=sys.stderr,
        )
        return 1

    if args.restart and client.collection_exists(coll):
        print(f"Dropping collection {coll} (--restart)")
        client.delete_collection(coll)
        time.sleep(0.5)
        for p in (ckpt_path, id_map_path):
            p.unlink(missing_ok=True)

    emb_meta = {}
    if meta_path.exists():
        emb_meta = json.loads(meta_path.read_text())
    lock_path = corpus_dir / "corpus_lock.json"
    emb_rel = str(emb_dir.relative_to(BIO_ROOT))

    def write_id_map() -> None:
        with id_map_path.open("w") as f:
            f.write("point_id\tchain_id\n")
            for i, cid in enumerate(ids):
                f.write(f"{i}\t{cid}\n")

    def write_manifest(*, n_collection: int, elapsed_s: float, upserted_now: int) -> None:
        write_json(
            manifest_path,
            {
                "url": url,
                "collection": coll,
                "mode": "smoke" if smoke else "paper",
                "n_upserted": len(ids),
                "n_upserted_this_run": upserted_now,
                "n_in_collection": n_collection,
                "dim": dim,
                "distance": xcfg.get("distance", "Cosine"),
                "embeddings": emb_rel,
                "embeddings_meta": emb_meta,
                "embeddings_vectors_fingerprint": file_sha256(vec_path),
                "corpus": str(corpus_dir.relative_to(BIO_ROOT)),
                "corpus_lock": (
                    str(lock_path.relative_to(BIO_ROOT)) if lock_path.exists() else None
                ),
                "id_map": str(id_map_path.relative_to(BIO_ROOT)),
                "checkpoint": str(ckpt_path.relative_to(BIO_ROOT)),
                "elapsed_s": round(elapsed_s, 1),
                "point_id_scheme": "row index in embeddings/*/ids.txt (0-based)",
                "completed_unix": int(time.time()),
            },
        )

    if not client.collection_exists(coll):
        print(
            f"Creating collection {coll} size={dim} "
            f"distance={xcfg.get('distance', 'Cosine')}"
        )
        client.create_collection(
            coll,
            size=dim,
            distance=str(xcfg.get("distance", "Cosine")),
        )
        start_idx = 0
    else:
        n_existing = client.count(coll)
        start_idx = 0
        if not args.restart and n_existing == len(ids):
            write_id_map()
            write_json(
                ckpt_path,
                {
                    "collection": coll,
                    "embeddings": emb_rel,
                    "n_total": len(ids),
                    "next_index": len(ids),
                    "updated_unix": int(time.time()),
                },
            )
            write_manifest(n_collection=n_existing, elapsed_s=0.0, upserted_now=0)
            print(
                f"Already complete: collection_count={n_existing} == n_ids={len(ids)}"
            )
            print("Refreshed id_map + manifest. Use --restart to rebuild.")
            return 0
        if ckpt_path.exists() and not args.restart:
            ckpt = json.loads(ckpt_path.read_text())
            if ckpt.get("n_total") == len(ids) and ckpt.get("embeddings") == emb_rel:
                start_idx = int(ckpt.get("next_index", 0))
                print(
                    f"Resume checkpoint: next_index={start_idx}/{len(ids)} "
                    f"(collection count={n_existing})"
                )
            else:
                print(
                    "WARN: checkpoint mismatch (corpus/size changed); "
                    "re-upserting from 0 (idempotent ids)",
                    file=sys.stderr,
                )
        else:
            print(
                f"Collection {coll} exists ({n_existing} points) — "
                "upserting from 0 (idempotent)"
            )

    write_id_map()

    if start_idx >= len(ids) and not args.restart:
        n = client.count(coll)
        write_manifest(n_collection=n, elapsed_s=0.0, upserted_now=0)
        print(f"Already complete: next_index={start_idx} collection_count={n}")
        print("Use --restart to rebuild.")
        return 0

    batch_size = int(args.batch_size or xcfg.get("upsert_batch_size", 128))
    remaining = len(ids) - start_idx
    progress = BatchProgress(remaining, label=f"upsert-{coll}")
    t0 = time.time()

    for start in range(start_idx, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        points = []
        for i in range(start, end):
            cid = ids[i]
            pl = payload_map.get(
                cid,
                {"chain_id": cid, "split": "", "scop_fold": ""},
            )
            points.append(
                {
                    "id": i,
                    "vector": vectors[i].astype(np.float32).tolist(),
                    "payload": pl,
                }
            )
        client.upsert(coll, points, wait=True)
        write_json(
            ckpt_path,
            {
                "collection": coll,
                "embeddings": emb_rel,
                "n_total": len(ids),
                "next_index": end,
                "updated_unix": int(time.time()),
            },
        )
        for _ in range(end - start):
            progress.tick(ok=True, force_print=(end >= len(ids)))

    n = client.count(coll)
    write_manifest(
        n_collection=n,
        elapsed_s=time.time() - t0,
        upserted_now=len(ids) - start_idx,
    )
    print(f"Indexed {len(ids)} → collection={coll} count={n} ({time.time() - t0:.1f}s)")
    print(f"  id_map:     {id_map_path}")
    print(f"  checkpoint: {ckpt_path} (next_index={len(ids)})")
    print(f"  manifest:   {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
