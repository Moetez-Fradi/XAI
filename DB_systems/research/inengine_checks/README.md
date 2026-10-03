# In-engine checks (`inengine_checks`)

**Paper:** §5.2 (attribution in-engine cost) · §5.4 (synthetic D=768 baseline) · §5.5 Table 3
(in-engine latency, ratio-1.0 kernel cost) · §5.6 (verify cost)

All three scripts talk to a running XQdrant on `http://127.0.0.1:6333` and read Qdrant's
server-side `time` field, so HTTP, JSON and TCP delayed-ACK stalls drop out. Each request
uses a fresh connection.

| Script | Needs | What it measures |
|--------|-------|------------------|
| `run_server_time.py` | SIFT1M collection (`bench_sift1m_n1000000`, e.g. from the M1 run); provisions `bench_synth_d768_n10000` itself | Server-side p50 for plain, rescoring, masked `L=0/1/3`, M1, verify `m=1/8` at ratios 0.25/0.75; attribution `top=1/10/50` vs plain at D=768 |
| `run_ratio1_kernel_cost.py` | SIFT1M collection | Masked vs plain with **all** 128 dims in focus: identical result sets (same traversal), so the time gap is per-distance kernel cost |
| `run_plain_baseline_highd.py` | `bench_synth_d768_n50000` (e.g. from the D=768 M2 run) | Plain full-search Recall@10 and exact coherence on the synthetic D=768 corpus |

## Results (host B: Ryzen 5 220, XQdrant `ed5ca79` in WSL2, 2026-10-03)

- SIFT1M plain search 1.25 ms; masked 0.70–0.89× of plain speed; masked `L=0` 1.00×;
  verify `m=8` adds 0.16–0.26 ms.
- Ratio 1.0: identical results for 500/500 queries; M1 1.94 ms, `L=1` 1.84 ms vs plain 1.23 ms.
- Attribution at D=768, N=10k: plain 2.45 ms; `top=1/10/50` 2.81/2.81/2.85 ms.
- Synthetic D=768, N=50k, ef=128: plain Recall@10 0.194; coherence 0.006/0.029/0.113/0.303
  at ratios 0.1/0.25/0.5/0.75.

Raw output: `experiments/2026-10-03_16-24-43__inengine_checks/`.

See also: [`../README.md`](../README.md).
