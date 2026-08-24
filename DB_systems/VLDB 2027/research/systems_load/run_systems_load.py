#!/usr/bin/env python3
"""Fig 6 — concurrent systems load: QPS and vector egress.

Measures plain HNSW, in-DB attribution, and client-side (with_vector) extraction
at concurrent clients in {1,2,4,8,16}. Right panel estimates egress MB/s when
full vectors leave the engine.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_research as cr  # noqa: E402

STEP_ID = "systems_load"
METRIC = "qps_egress"
DEFAULT_CLIENTS = [1, 2, 4, 8, 16]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["http", "simulated"], default="http")
    p.add_argument("--qdrant-url", default="http://127.0.0.1:6335")
    p.add_argument("--xqdrant-url", default="http://127.0.0.1:6333")
    p.add_argument("--dimensions", type=int, nargs="+", default=[768])
    p.add_argument("--queries", type=int, default=40)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--ef-search", type=int, default=128)
    p.add_argument("--clients", type=int, nargs="+", default=DEFAULT_CLIENTS)
    p.add_argument("--no-provision", action="store_true")
    cr.add_dataset_args(p)
    cr.add_trials_arg(p)
    return p.parse_args()


def _sim_qps(path: str, clients: int, dim: int) -> tuple[float, float]:
    # Paper: plain ~0.8–1.5k QPS; attribution paths ~20–345 QPS; egress ~1.76 MB/s at 16.
    if path == "plain":
        qps = min(1500.0, 180.0 * clients ** 0.85)
        egress = 0.0
    elif path == "in_db":
        qps = min(345.0, 22.0 * clients ** 0.9)
        egress = 0.02 * clients
    else:
        qps = min(340.0, 21.0 * clients ** 0.9)
        bytes_per_hit = dim * 4 * 10
        egress = qps * bytes_per_hit / 1e6
    return qps, egress


def _http_qps(client, queries, path, k, ef, dim, n_clients) -> tuple[float, float]:
    def one(q):
        if path == "plain":
            return client.query(cr.focus_body(q, k, ef))
        if path == "in_db":
            body = cr.focus_body(q, k, ef)
            body["with_dims_explained"] = {"top": 10}
            return client.query(body)
        body = cr.focus_body(q, k, ef)
        body["with_vector"] = True
        return client.query(body)

    n = min(len(queries), 32)
    batch = list(queries[:n]) * max(1, n_clients)
    t0 = time.perf_counter()
    ok = 0
    with ThreadPoolExecutor(max_workers=n_clients) as pool:
        futs = [pool.submit(one, batch[i % len(batch)]) for i in range(len(batch))]
        for f in as_completed(futs):
            out = f.result()
            if out.status == "ok":
                ok += 1
    elapsed = max(1e-6, time.perf_counter() - t0)
    qps = ok / elapsed
    if path == "client":
        egress = qps * dim * 4 * k / 1e6
    elif path == "in_db":
        egress = 0.0
    else:
        egress = 0.0
    return qps, egress


def main() -> int:
    args = parse_args()
    xq = args.xqdrant_url or args.qdrant_url
    run = cr.init_research_run(
        STEP_ID, METRIC, mode=args.mode, dimensions=args.dimensions,
        queries=args.queries, qdrant_url=args.qdrant_url, xqdrant_url=xq,
        sweep={"clients": args.clients}, extra={"trials": args.trials},
    )
    trial_rows = []
    for dataset in cr.iter_datasets(args):
        dim = dataset.dimension
        client = None
        if args.mode == "http":
            client = cr.ResearchClient(
                dataset, args.qdrant_url, xq, provision=not args.no_provision
            )
            if not args.no_provision:
                client.wait_for_indexing(expected_points=dataset.num_vectors)
        for trial in range(args.trials):
            seed = cr.trial_seed(trial)
            queries, _ = cr.select_queries(dataset, args.queries, seed=seed)
            for n_cli in args.clients:
                for path in ("plain", "in_db", "client"):
                    if args.mode == "http":
                        qps, egress = _http_qps(
                            client, queries, path, args.k, args.ef_search, dim, n_cli
                        )
                    else:
                        qps, egress = _sim_qps(path, n_cli, dim)
                    trial_rows.append({
                        "trial": trial, "dimension": dim, "clients": n_cli,
                        "path": path, "qps": qps, "egress_MBps": egress,
                        "simulated": int(args.mode == "simulated"),
                    })
                    print(f"  trial={trial} clients={n_cli} {path}: qps={qps:.1f} egress={egress:.3f} MB/s")

    rows = cr.write_trial_and_summary_csv(
        cr.bench_common.RESULTS_DIR / "systems_load.csv",
        trial_rows, key_fields=["dimension", "clients", "path"],
    )
    plt = cr._mpl()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    dim = args.dimensions[0]
    for path, label in (("plain", "plain HNSW"), ("in_db", "in-DB attrib."), ("client", "client-side")):
        xs, ys, es = [], [], []
        for c in args.clients:
            match = [r for r in rows if int(r["clients"]) == c and r["path"] == path]
            if not match:
                continue
            xs.append(c)
            ys.append(float(match[0]["qps"]))
            es.append(float(match[0].get("qps_std", 0.0) or 0.0))
        axes[0].errorbar(xs, ys, yerr=es if any(e > 0 for e in es) else None,
                         marker="o", label=label)
        e_y, e_e = [], []
        for c in args.clients:
            match = [r for r in rows if int(r["clients"]) == c and r["path"] == path]
            if match:
                e_y.append(float(match[0]["egress_MBps"]))
                e_e.append(float(match[0].get("egress_MBps_std", 0.0) or 0.0))
        axes[1].errorbar(args.clients[:len(e_y)], e_y, yerr=e_e if any(e > 0 for e in e_e) else None,
                         marker="o", label=label)
    axes[0].set_xlabel("Concurrent clients")
    axes[0].set_ylabel("QPS")
    axes[0].set_title("Throughput")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Concurrent clients")
    axes[1].set_ylabel("Egress MB/s")
    axes[1].set_title("Vector egress")
    axes[1].legend(fontsize=8)
    prefix = "[SIMULATED] " if args.mode == "simulated" else ""
    fig.suptitle(f"{prefix}Systems load (D={dim})", y=1.02)
    fig.tight_layout()
    cr.save_fig(fig, "fig06_systems_load")
    print(f"[systems_load] done -> {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
