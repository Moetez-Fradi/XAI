"""Plain full-search Recall@10 and exact coherence on the synthetic D=768, N=50k corpus.

Answers: is the high-D corpus already hard for plain HNSW (before any masking)?
Requires the collection to be provisioned and indexed (run the M2 sweep first).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402

URL = "http://127.0.0.1:6333"
DIM, N, Q, K, EF, TRIALS = 768, 50_000, 500, 10, 128, 5
RATIOS = [0.1, 0.25, 0.5, 0.75]

ds = cr.build_dataset("synthetic", dimension=DIM, num_vectors=N, num_queries=Q)
client = cr.ResearchClient(ds, URL, URL, provision=False)

plain, coh = [], {r: [] for r in RATIOS}
for trial in range(TRIALS):
    seed = cr.trial_seed(trial)
    queries, _ = cr.select_queries(ds, Q, seed=seed)
    sub_gts = {r: cr.SubspaceGT(ds.vectors, cr.subspace_indices(DIM, r, seed=seed),
                                distance=ds.distance) for r in RATIOS}
    rec, c = [], {r: [] for r in RATIOS}
    for q in queries:
        full_ids, _ = cr.brute_force_top_k(q, ds.vectors, K, distance=ds.distance)
        out = client.query(cr.focus_body(q, K, EF))
        rec.append(cr.recall_at_k(out.result.ids, full_ids, K))
        for r in RATIOS:
            c[r].append(cr.recall_at_k(sub_gts[r].top_k(q, K)[0], full_ids, K))
    plain.append(np.mean(rec))
    for r in RATIOS:
        coh[r].append(np.mean(c[r]))
    print(f"trial {trial}: plain recall={plain[-1]:.3f} "
          + " ".join(f"coh@{r}={coh[r][-1]:.3f}" for r in RATIOS), flush=True)

print(f"PLAIN full-search Recall@10 (ef={EF}): {np.mean(plain):.3f} ± {np.std(plain, ddof=1):.3f}")
for r in RATIOS:
    print(f"COHERENCE ratio={r}: {np.mean(coh[r]):.3f} ± {np.std(coh[r], ddof=1):.3f}")
