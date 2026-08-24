# Fig 6 systems load

```bash
python run_systems_load.py --mode simulated --queries 8 --trials 1
python run_systems_load.py --mode http --xqdrant-url http://127.0.0.1:6333 \
  --dataset synthetic --num-vectors 20000 --queries 40 --trials 5
```
