#!/usr/bin/env bash
#
# Build a cached MS MARCO / all-MiniLM-L6-v2 embedding matrix.
#
# Writes:
#   data/minilm/embeddings.npy     (N, 384) float32, L2-normalized
#   data/minilm/queries.npy        (Q, 384)
#   data/minilm/passages.jsonl     original texts (id, text)
#   data/minilm/query_texts.jsonl  query texts used for the case study
#
# Env:
#   MINILM_N       number of passages (default 100000; smoke/expansion: 20000)
#   MINILM_Q       number of queries  (default 500)
#   MINILM_MODEL   sentence-transformers id (default all-MiniLM-L6-v2)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$SCRIPT_DIR/data/minilm}"
MINILM_N="${MINILM_N:-100000}"
MINILM_Q="${MINILM_Q:-500}"
MINILM_MODEL="${MINILM_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"

mkdir -p "$DEST"

if [ -f "$DEST/embeddings.npy" ] && [ -f "$DEST/queries.npy" ]; then
  python3 - <<PY
import numpy as np
from pathlib import Path
p = Path("$DEST") / "embeddings.npy"
n = np.load(p, mmap_mode="r").shape[0]
print(f"MiniLM cache already present: {p} N={n}")
PY
  if [ "${FORCE_MINILM:-0}" != "1" ]; then
    exit 0
  fi
fi

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r "$SCRIPT_DIR/requirements.txt"
python3 -m pip install -q "sentence-transformers>=2.2.0" "datasets>=2.14.0"

export DEST MINILM_N MINILM_Q MINILM_MODEL
python3 - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

dest = Path(os.environ["DEST"])
dest.mkdir(parents=True, exist_ok=True)
n = int(os.environ["MINILM_N"])
nq = int(os.environ["MINILM_Q"])
model_id = os.environ["MINILM_MODEL"]

print(f"[minilm] loading {n} MS MARCO passages + {nq} queries via HuggingFace datasets")
from datasets import load_dataset

# BEIR MS MARCO corpus: _id, title, text. Queries: _id, text.
corpus = load_dataset("BeIR/msmarco", "corpus", split="corpus")
queries = load_dataset("BeIR/msmarco", "queries", split="queries")

passages = []
for i, row in enumerate(corpus):
    if i >= n:
        break
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    blob = (title + " " + text).strip() if title else text
    passages.append({"id": int(i), "text": blob[:2000]})

q_rows = []
for i, row in enumerate(queries):
    if i >= nq:
        break
    q_rows.append({"id": int(i), "text": (row.get("text") or "").strip()[:500]})

print(f"[minilm] embedding with {model_id} ...")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(model_id)
vec = model.encode(
    [p["text"] or " " for p in passages],
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(np.float32)
qvec = model.encode(
    [q["text"] or " " for q in q_rows],
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(np.float32)

np.save(dest / "embeddings.npy", vec)
np.save(dest / "queries.npy", qvec)
with (dest / "passages.jsonl").open("w", encoding="utf-8") as f:
    for p in passages:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
with (dest / "query_texts.jsonl").open("w", encoding="utf-8") as f:
    for q in q_rows:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
print(f"[minilm] wrote {vec.shape} embeddings and {qvec.shape} queries -> {dest}")
PY
