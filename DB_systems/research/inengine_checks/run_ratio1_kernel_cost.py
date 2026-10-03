"""Masked vs plain at ratio 1.0: identical metric, so identical traversal.

Any server-side time difference is per-distance kernel cost (gather vs contiguous SIMD).
Also checks the result sets are identical, which confirms identical traversal.
"""
import numpy as np
import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_common import SIFT_DIR, read_fvecs  # noqa: E402

URL = "http://127.0.0.1:6333/collections/bench_sift1m_n1000000/points/query"
H = {"Connection": "close"}
qs = read_fvecs(SIFT_DIR / "sift_query.fvecs", limit=500)
all_dims = np.arange(128)


def run(mk):
    ts, ids = [], []
    for q in qs:
        best = None
        for _ in range(3):
            r = requests.post(URL, json=mk(q), headers=H, timeout=60)
            r.raise_for_status()
            j = r.json()
            t = j["time"] * 1e3
            best = t if best is None else min(best, t)
        ts.append(best)
        ids.append([p["id"] for p in j["result"]["points"]])
    return np.median(ts), ids


plain_t, plain_ids = run(lambda q: cr.focus_body(q, 10, 128))
for name, mk in [
    ("masked all layers (M1), all 128 dims", lambda q: cr.focus_body(q, 10, 128, all_dims, masked=True)),
    ("masked L=1, all 128 dims", lambda q: cr.focus_body(q, 10, 128, all_dims, masked=True, mask_from_layer=1)),
]:
    t, ids = run(mk)
    same = np.mean([set(a) == set(b) for a, b in zip(ids, plain_ids)])
    print(f"{name:40s} p50={t:.3f} ms  plain p50={plain_t:.3f} ms  "
          f"x{plain_t / t:.2f}  identical result sets: {same:.1%}", flush=True)
