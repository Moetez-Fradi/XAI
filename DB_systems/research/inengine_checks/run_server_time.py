"""Server-side (in-engine) latency for masked variants and attribution.

Uses Qdrant's own per-request ``time`` field, so client/transport effects (HTTP,
JSON, TCP delayed-ACK stalls) drop out. Fresh connection per request.
"""
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_backends import HttpBackend  # noqa: E402
from bench_common import SIFT_DIR, read_fvecs  # noqa: E402

URL = "http://127.0.0.1:6333"
Q, K, EF, REPS = 500, 10, 128, 3
H = {"Connection": "close"}


def server_ms(coll, bodies):
    """p50 / mean of server-side time (ms) over bodies, best-of-REPS per query."""
    per_q = []
    for b in bodies:
        ts = []
        for _ in range(REPS):
            r = requests.post(f"{URL}/collections/{coll}/points/query", json=b, headers=H, timeout=60)
            r.raise_for_status()
            ts.append(r.json()["time"] * 1e3)
        per_q.append(min(ts))
    a = np.asarray(per_q)
    return np.median(a), a.mean()


def row(name, coll, mk, qs, base=None):
    p50, mean = server_ms(coll, [mk(q) for q in qs])
    rel = f"  x{base / p50:.2f} vs plain" if base else ""
    print(f"  {name:36s} p50={p50:7.3f} ms  mean={mean:7.3f} ms{rel}", flush=True)
    return p50


# --- A. masked variants on SIFT1M ------------------------------------------------
sift = "bench_sift1m_n1000000"
qs = read_fvecs(SIFT_DIR / "sift_query.fvecs", limit=Q)
print(f"[A] SIFT1M server-side time, Q={Q}, ef={EF}, best of {REPS}")
base = row("plain full search", sift, lambda q: cr.focus_body(q, K, EF), qs)
for ratio in (0.25, 0.75):
    dims = cr.subspace_indices(128, ratio, seed=cr.trial_seed(0))
    print(f" ratio={ratio}")
    fb = lambda q, **kw: cr.focus_body(q, kw.pop("limit", K), EF, dims, **kw)  # noqa: E731
    row("rescore-only focus (unmasked)", sift, lambda q: fb(q), qs, base)
    row("masked L=0 (no layer masked)", sift, lambda q: fb(q, masked=True, mask_from_layer=0), qs, base)
    row("masked L=1 (M2)", sift, lambda q: fb(q, masked=True, mask_from_layer=1), qs, base)
    row("masked L=3 (M2)", sift, lambda q: fb(q, masked=True, mask_from_layer=3), qs, base)
    row("masked all layers (M1)", sift, lambda q: fb(q, masked=True), qs, base)
    row("L=1 verify, m=1", sift, lambda q: fb(q, masked=True, mask_from_layer=1, verify=True), qs, base)
    row("L=1 limit=80, no verify", sift, lambda q: fb(q, limit=80, masked=True, mask_from_layer=1), qs, base)
    row("L=1 verify, m=8 (limit=80)", sift,
        lambda q: fb(q, limit=80, masked=True, mask_from_layer=1, verify=True), qs, base)

# --- B. attribution on synthetic D=768, N=10k (attribution-suite corpus) ----------
ds = cr.build_dataset("synthetic", dimension=768, num_vectors=10_000, num_queries=Q)
be = HttpBackend(ds, qdrant_url=URL, xqdrant_url=URL, setup_collection=True)
be.wait_for_indexing(expected_points=ds.num_vectors)
coll = be.collection_name
aq = ds.queries[:Q]
print(f"[B] attribution server-side time, {coll}, Q={Q}, ef={EF}")
plain = row("plain search", coll, lambda q: cr.focus_body(q, K, EF), aq)
for m in (1, 10, 50):
    row(f"with_dims_explained top={m}", coll,
        lambda q, m=m: {**cr.focus_body(q, K, EF), "with_dims_explained": {"top": m}}, aq, plain)
