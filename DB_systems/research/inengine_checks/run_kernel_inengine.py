"""In-engine masked latency per kernel (K1): plain vs. M1 at ratios 0.25 / 0.75 / 1.0.

Start XQdrant once per kernel (``XQDRANT_MASKED_KERNEL=repack`` or ``gather``; the engine
default is ``repack``) and run this script against it with the matching ``--kernel`` label.
Server-side ``time`` field, best of REPS per query, configurations interleaved per query
(``cr.server_time_interleaved``). Focus sets match ``run_server_time.py``
(``subspace_indices`` with ``trial_seed(0)``); ratio 1.0 uses all 128 dimensions and also
checks that masked and plain return identical result sets.

    python run_kernel_inengine.py --kernel gather  [--out results.csv]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402
from bench_common import SIFT_DIR, read_fvecs  # noqa: E402

URL = "http://127.0.0.1:6333/collections/bench_sift1m_n1000000/points/query"
Q, K, EF, REPS = 500, 10, 128, 3
RATIOS = (0.25, 0.75, 1.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", required=True, choices=["repack", "gather"],
                   help="label for the kernel the running server was started with")
    p.add_argument("--out", type=Path, default=None, help="append rows to this CSV")
    args = p.parse_args()

    qs = read_fvecs(SIFT_DIR / "sift_query.fvecs", limit=Q)
    configs = [("plain", [cr.focus_body(q, K, EF) for q in qs])]
    for ratio in RATIOS:
        dims = (np.arange(128) if ratio == 1.0
                else cr.subspace_indices(128, ratio, seed=cr.trial_seed(0)))
        configs.append((f"M1 r={ratio}", [cr.focus_body(q, K, EF, dims, masked=True) for q in qs]))

    print(f"[K1 in-engine] kernel={args.kernel} SIFT1M Q={Q} ef={EF} best of {REPS}, interleaved")
    res = cr.server_time_interleaved(URL, configs, REPS)
    plain = res["plain"]
    print(f"  plain full search   p50={plain['p50']:.3f} ms")
    rows = [{"kernel": args.kernel, "query": "plain", "ratio": "", "p50_ms": f"{plain['p50']:.4f}",
             "speed_vs_plain": "1.00", "identical_sets": ""}]
    for ratio in RATIOS:
        r = res[f"M1 r={ratio}"]
        same = (np.mean([a == b for a, b in zip(r["ids"], plain["ids"])])
                if ratio == 1.0 else None)
        extra = f"  identical sets {same:.1%}" if same is not None else ""
        print(f"  M1 ratio={ratio:<4}       p50={r['p50']:.3f} ms  "
              f"x{plain['p50'] / r['p50']:.2f} vs plain{extra}", flush=True)
        rows.append({"kernel": args.kernel, "query": "M1", "ratio": ratio,
                     "p50_ms": f"{r['p50']:.4f}", "speed_vs_plain": f"{plain['p50'] / r['p50']:.2f}",
                     "identical_sets": "" if same is None else f"{same:.4f}"})

    if args.out:
        new = not args.out.exists()
        with args.out.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
