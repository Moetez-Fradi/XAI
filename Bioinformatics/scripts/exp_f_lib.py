"""Shared helpers for Experiment F — per-query workflow timing."""
from __future__ import annotations
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
import numpy as np
from exp_e_lib import which_tmalign

def load_fasta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    cur_id: str | None = None
    parts: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if cur_id is not None:
                out[cur_id] = ''.join(parts)
            cur_id = line[1:].split()[0]
            parts = []
        else:
            parts.append(line)
    if cur_id is not None:
        out[cur_id] = ''.join(parts)
    return out

def parse_foldseek_hits(path: Path, *, query_id: str, topk: int) -> list[str]:
    hits: list[tuple[float, str]] = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            q = Path(parts[0]).stem.split('.')[0]
            s = Path(parts[1]).stem.split('.')[0]
            if q != query_id or q == s:
                continue
            try:
                score = float(parts[4]) if len(parts) > 4 else float(parts[3])
            except ValueError:
                score = float(parts[3]) if len(parts) > 3 else 0.0
            hits.append((score, s))
    hits.sort(key=lambda x: -x[0])
    seen: set[str] = set()
    out: list[str] = []
    for _sc, hid in hits:
        if hid in seen:
            continue
        seen.add(hid)
        out.append(hid)
        if len(out) >= topk:
            break
    return out

def parse_blast_hits(path: Path, *, topk: int) -> list[str]:
    hits: list[tuple[float, str]] = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            q, s = (parts[0], parts[1])
            try:
                score = float(parts[2]) if len(parts) > 2 else 0.0
            except ValueError:
                score = 0.0
            if q == s:
                continue
            hits.append((score, s))
    hits.sort(key=lambda x: -x[0])
    seen: set[str] = set()
    out: list[str] = []
    for _sc, hid in hits:
        if hid in seen:
            continue
        seen.add(hid)
        out.append(hid)
        if len(out) >= topk:
            break
    return out

def which_or_die(name: str) -> str:
    from common_bio import BIO_ROOT
    candidates = [BIO_ROOT / 'tools/blast/bin' / name, BIO_ROOT / 'tools/mamba/envs/bio-tools/bin' / name]
    for p in candidates:
        if p.exists() and os.access(p, os.X_OK):
            return str(p.resolve())
    p = shutil.which(name)
    if not p:
        raise FileNotFoundError(f'{name} not on PATH')
    return p

def which_optional(name: str) -> str | None:
    try:
        return which_or_die(name)
    except FileNotFoundError:
        return None

class ESM2SingleEmbedder:

    def __init__(self, *, model_dir: Path, device: str='auto', max_len: int=1024) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        from embed_esm2 import cuda_usable, mean_pool, pick_device
        self.device = pick_device(device)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModel.from_pretrained(str(model_dir))
        self.model.eval()
        self.model.to(self.device)
        self._mean_pool = mean_pool
        self._torch = torch

    def embed_sequence(self, sequence: str) -> tuple[list[float], float]:
        t0 = time.perf_counter()
        with self._torch.inference_mode():
            enc = self.tokenizer([sequence], return_tensors='pt', padding=True, truncation=True, max_length=self.max_len)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            emb = self._mean_pool(out.last_hidden_state, enc['attention_mask'])
            vec = emb[0].detach().float().cpu().numpy().tolist()
        return (vec, time.perf_counter() - t0)

def time_foldseek_query(*, foldseek: str, query_pdb: Path, target_db: Path, tmp_root: Path, sensitivity: float, max_seqs: int, threads: int) -> tuple[float, list[str]]:
    tmp_root.mkdir(parents=True, exist_ok=True)
    hits_path = tmp_root / 'hits.m8'
    t0 = time.perf_counter()
    subprocess.run([foldseek, 'easy-search', str(query_pdb), str(target_db), str(hits_path), str(tmp_root / 'fs_tmp'), '-s', str(sensitivity), '--max-seqs', str(max_seqs), '--threads', str(threads), '--format-output', 'query,target,evalue,bits,alntmscore'], check=True, capture_output=True)
    elapsed = time.perf_counter() - t0
    hits = parse_foldseek_hits(hits_path, query_id=query_pdb.stem, topk=max_seqs)
    return (elapsed, hits)

def time_blast_query(*, blastp: str, query_seq: str, query_id: str, db_prefix: Path, tmp_root: Path, evalue: float, max_target_seqs: int, num_threads: int) -> tuple[float, list[str]]:
    tmp_root.mkdir(parents=True, exist_ok=True)
    qfa = tmp_root / 'query.fasta'
    out_tsv = tmp_root / 'hits.tsv'
    qfa.write_text(f'>{query_id}\n{query_seq}\n')
    t0 = time.perf_counter()
    subprocess.run([blastp, '-query', str(qfa), '-db', str(db_prefix), '-out', str(out_tsv), '-outfmt', '6 qseqid sseqid bitscore evalue', '-evalue', str(evalue), '-max_target_seqs', str(max_target_seqs), '-num_threads', str(num_threads)], check=True, capture_output=True)
    elapsed = time.perf_counter() - t0
    hits = parse_blast_hits(out_tsv, topk=max_target_seqs)
    return (elapsed, hits)
_TM_RE = re.compile('TM-score\\s*=\\s*([0-9.]+)')

def time_tmalign_candidates(*, tmalign: str, query_pdb: Path, struct_dir: Path, candidates: list[str], max_candidates: int) -> tuple[float, int]:
    t0 = time.perf_counter()
    n = 0
    for hid in candidates[:max_candidates]:
        t_pdb = struct_dir / f'{hid}.pdb'
        if not t_pdb.exists():
            continue
        subprocess.run([tmalign, str(query_pdb), str(t_pdb)], capture_output=True, text=True, check=False)
        n += 1
    return (time.perf_counter() - t0, n)

def time_xqdrant_query(client: Any, *, collection: str, vector: list[float], limit: int, dims_top: int, train_filter: dict | None) -> tuple[float, int]:
    t0 = time.perf_counter()
    pts = client.query(collection, vector, limit=limit, with_payload=True, with_dims_explained={'top': dims_top}, query_filter=train_filter)
    elapsed = time.perf_counter() - t0
    n_attr = sum((1 for p in pts if p.get('dims_explained')))
    return (elapsed, n_attr)

def summarize_times(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {'n': 0, 'mean_s': None, 'median_s': None, 'p25_s': None, 'p75_s': None, 'total_s': None}
    arr = np.array(values, dtype=np.float64)
    return {'n': int(len(arr)), 'mean_s': round(float(arr.mean()), 3), 'median_s': round(float(np.median(arr)), 3), 'p25_s': round(float(np.percentile(arr, 25)), 3), 'p75_s': round(float(np.percentile(arr, 75)), 3), 'total_s': round(float(arr.sum()), 3)}
