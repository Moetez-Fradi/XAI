# In-engine checks (`inengine_checks`)

**Paper:** §5.2 (attribution in-engine cost) · §5.4 (synthetic D=768 baseline) · §5.5 Table 3
(in-engine latency, ratio-1.0 cost, kernel comparison) · §5.6 (verify cost)

All scripts talk to a running XQdrant on `http://127.0.0.1:6333` and read Qdrant's
server-side `time` field, so HTTP, JSON and TCP delayed-ACK stalls drop out. Each request
uses a fresh connection. Within a script, all configurations are timed **interleaved per
query** (`common_research.server_time_interleaved`: for query *i*, every configuration in an
order rotated by *i*, best of 3), so drift over a long run (laptop thermals, frequency,
background load) affects every configuration alike.

The masked kernel is a server setting: start XQdrant with `XQDRANT_MASKED_KERNEL=repack`
(the engine default, used for every paper number unless the kernel is the subject) or
`gather`, and pass the same value as `--kernel` so the output records it.

| Script | Needs | What it measures |
|--------|-------|------------------|
| `run_server_time.py --kernel K` | SIFT1M collection (`bench_sift1m_n1000000`, e.g. from the M1 run); provisions `bench_synth_d768_n10000` itself | Server-side p50 for plain, rescoring, masked `L=0/1/3`, M1, verify `m=1/8` at ratios 0.25/0.75 (Table 3); attribution `top=1/10/50` vs plain at D=768 |
| `run_ratio1_kernel_cost.py --kernel K` | SIFT1M collection | Masked vs plain with **all** 128 dims in focus: identical result sets (same traversal), so the time gap is per-distance cost |
| `run_kernel_inengine.py --kernel K` | SIFT1M collection | Plain vs M1 at ratios 0.25/0.75/1.0, run once per kernel (K1 in-engine) |
| `run_plain_baseline_highd.py` | `bench_synth_d768_n50000` (e.g. from the D=768 M2 run) | Plain full-search Recall@10 and exact coherence on the synthetic D=768 corpus |

## Results (host B: Ryzen 5 220, XQdrant `ed5ca79` perf build in WSL2)

Canonical: [`results_2026-10-05_hostB/`](./results_2026-10-05_hostB/) — interleaved runs,
kernel recorded per file (`server_env_*.txt` shows the server's actual environment).

All rows within a script are interleaved; repack unless noted. 2026-10-05.

- **Table 3** (`server_time_repack.txt`, SIFT1M): plain 1.32 ms; rescoring 0.80×/0.81×;
  masked `L=0` 1.03×/1.02× (request path adds nothing); M2 `L=1` 0.84×/0.67×, `L=3`
  0.85×/0.67×, M1 0.85×/0.67× at ratios 0.25/0.75; V1 `m=8` 0.73×/0.59× (adds
  0.23–0.28 ms over `L=1`, of which the rescore is ≤ 0.09 ms).
- **Ratio 1.0** (`ratio1_*.txt`): identical result sets for 500/500 queries; M1 1.67 ms vs
  plain 1.04 ms (0.62×), `L=1` 0.64× with repack; 0.65×/0.65× with gather.
- **Kernel, in-engine** (`kernel_inengine_*.txt`, `kernel_inengine.csv`): M1 speed vs plain
  at ratios 0.25/0.75/1.0 is 0.82×/0.65×/0.63× with repack and 0.84×/0.67×/0.65× with
  gather — within 3%.
- **Attribution** (section B of `server_time_repack.txt`, synthetic D=768, N=10k): plain
  2.12 ms; `top=1/10/50` 2.42/2.41/2.44 ms (+0.29–0.32 ms, ~14%).

Synthetic D=768, N=50k, ef=128 (2026-10-03): plain Recall@10 0.194; coherence
0.006/0.029/0.113/0.303 at ratios 0.1/0.25/0.5/0.75.

The 2026-10-03 run (`experiments/2026-10-03_16-24-43__inengine_checks/`, sequential blocks,
kernel not recorded) is superseded. Sequential blocks are sensitive to drift on this laptop:
a sequential gather-vs-repack comparison on 2026-10-05 showed a spurious 0.73× vs 0.89× gap
that disappears when the configurations are interleaved.

See also: [`../README.md`](../README.md).
