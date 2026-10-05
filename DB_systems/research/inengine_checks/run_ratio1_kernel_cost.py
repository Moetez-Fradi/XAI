"""Masked vs plain at ratio 1.0: identical metric, so identical traversal.

Any server-side time difference is per-distance kernel cost (masked scoring vs
contiguous SIMD). Also checks the result sets are identical, which confirms identical
traversal. Configurations are timed interleaved per query (``cr.server_time_interleaved``).

    python run_ratio1_kernel_cost.py --kernel repack
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_common import SIFT_DIR, read_fvecs  # noqa: E402

URL = "http://127.0.0.1:6333/collections/bench_sift1m_n1000000/points/query"
Q, K, EF, REPS = 500, 10, 128, 3

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--kernel", required=True, choices=["repack", "gather"],
               help="kernel the running server was started with (for the record)")
args = p.parse_args()

qs = read_fvecs(SIFT_DIR / "sift_query.fvecs", limit=Q)
all_dims = np.arange(128)
configs = [
    ("plain", [cr.focus_body(q, K, EF) for q in qs]),
    ("masked all layers (M1), all 128 dims",
     [cr.focus_body(q, K, EF, all_dims, masked=True) for q in qs]),
    ("masked L=1, all 128 dims",
     [cr.focus_body(q, K, EF, all_dims, masked=True, mask_from_layer=1) for q in qs]),
]
print(f"[ratio 1.0] kernel={args.kernel} SIFT1M Q={Q} ef={EF} best of {REPS}, interleaved")
res = cr.server_time_interleaved(URL, configs, REPS)
plain = res["plain"]
for name, r in res.items():
    if name == "plain":
        continue
    same = np.mean([a == b for a, b in zip(r["ids"], plain["ids"])])
    print(f"{name:40s} p50={r['p50']:.3f} ms  plain p50={plain['p50']:.3f} ms  "
          f"x{plain['p50'] / r['p50']:.2f}  identical result sets: {same:.1%}", flush=True)
