# Fig 8 / Table 2 — MiniLM and GIST M1

Requires `../../fetch_minilm.sh` and/or `../../fetch_gist.sh`.

```bash
python run_minilm_gist_m1.py --mode simulated --corpora synthetic --queries 8 --trials 1
python run_minilm_gist_m1.py --mode http --corpora minilm gist1m --num-vectors 20000 --queries 40 --trials 5
```
