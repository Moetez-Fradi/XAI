"""Server-side (in-engine) latency for masked variants and attribution.

Uses Qdrant's own per-request ``time`` field, so client/transport effects (HTTP,
JSON, TCP delayed-ACK stalls) drop out. Fresh connection per request. All
configurations of a section are timed interleaved per query
(``cr.server_time_interleaved``), so drift over the run affects them alike.

Start the server with the kernel to report (``XQDRANT_MASKED_KERNEL``; engine default
``repack``) and pass the same value as ``--kernel`` so the output records it.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_backends import HttpBackend  # noqa: E402
from bench_common import SIFT_DIR, read_fvecs  # noqa: E402

URL = "http://127.0.0.1:6333"
Q, K, EF, REPS = 500, 10, 128, 3

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--kernel", required=True, choices=["repack", "gather"],
               help="kernel the running server was started with (for the record)")
p.add_argument("--skip-attribution", action="store_true", help="only run section A")
args = p.parse_args()


def report(results, base_name):
    base = results[base_name]["p50"]
    for name, r in results.items():
        rel = "" if name == base_name else f"  x{base / r['p50']:.2f} vs plain"
        print(f"  {name:40s} p50={r['p50']:7.3f} ms  mean={r['mean']:7.3f} ms{rel}", flush=True)


# --- A. masked variants on SIFT1M ------------------------------------------------
sift = "bench_sift1m_n1000000"
qs = read_fvecs(SIFT_DIR / "sift_query.fvecs", limit=Q)
print(f"[A] SIFT1M server-side time, kernel={args.kernel}, Q={Q}, ef={EF}, best of {REPS}, "
      "interleaved")
configs = [("plain full search", [cr.focus_body(q, K, EF) for q in qs])]
for ratio in (0.25, 0.75):
    dims = cr.subspace_indices(128, ratio, seed=cr.trial_seed(0))

    def fb(q, limit=K, **kw):
        return cr.focus_body(q, limit, EF, dims, **kw)

    variants = [
        ("rescore-only focus (unmasked)", {}),
        ("masked L=0 (no layer masked)", {"masked": True, "mask_from_layer": 0}),
        ("masked L=1 (M2)", {"masked": True, "mask_from_layer": 1}),
        ("masked L=3 (M2)", {"masked": True, "mask_from_layer": 3}),
        ("masked all layers (M1)", {"masked": True}),
        ("L=1 verify, m=1", {"masked": True, "mask_from_layer": 1, "verify": True}),
        ("L=1 limit=80, no verify", {"limit": 80, "masked": True, "mask_from_layer": 1}),
        ("L=1 verify, m=8 (limit=80)",
         {"limit": 80, "masked": True, "mask_from_layer": 1, "verify": True}),
    ]
    for name, kw in variants:
        configs.append((f"r={ratio} {name}", [fb(q, **kw) for q in qs]))
results = cr.server_time_interleaved(f"{URL}/collections/{sift}/points/query", configs, REPS)
report(results, "plain full search")

if args.skip_attribution:
    raise SystemExit(0)

# --- B. attribution on synthetic D=768, N=10k (attribution-suite corpus) ----------
ds = cr.build_dataset("synthetic", dimension=768, num_vectors=10_000, num_queries=Q)
be = HttpBackend(ds, qdrant_url=URL, xqdrant_url=URL, setup_collection=True)
be.wait_for_indexing(expected_points=ds.num_vectors)
coll = be.collection_name
aq = ds.queries[:Q]
print(f"[B] attribution server-side time, {coll}, Q={Q}, ef={EF}, interleaved")
configs = [("plain search", [cr.focus_body(q, K, EF) for q in aq])]
for m in (1, 10, 50):
    configs.append((f"with_dims_explained top={m}",
                    [{**cr.focus_body(q, K, EF), "with_dims_explained": {"top": m}} for q in aq]))
results = cr.server_time_interleaved(f"{URL}/collections/{coll}/points/query", configs, REPS)
report(results, "plain search")
