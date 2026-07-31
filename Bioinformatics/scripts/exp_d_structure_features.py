#!/usr/bin/env python3
"""Structure-based thermostability proxies for Experiment D.

Computes per-chain:
  - ion_pair_density: oppositely charged residue pairs within cutoff (Å)
  - packing_density: 1 - mean(relative SASA) via Biopython Shrake-Rupley

Caches JSON under data/annotations/exp_d/<chain_id>.json
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotate_dssp import MAX_ASA, find_structure, materialize_structure, load_chain_targets  # noqa: E402
from common_bio import ensure_dir, resolve_path, write_json  # noqa: E402

NEG = frozenset("DE")
POS = frozenset("KRH")


def _aa1(res) -> str:
    from Bio.PDB.Polypeptide import protein_letters_3to1

    try:
        return protein_letters_3to1.get(res.resname, "X")
    except Exception:
        return "X"


def _min_heavy_distance(res1, res2) -> float:
    best = float("inf")
    for a1 in res1:
        if a1.element == "H":
            continue
        for a2 in res2:
            if a2.element == "H":
                continue
            d = float(a1 - a2)
            if d < best:
                best = d
    return best


def ion_pairs_from_chain(chain, cutoff: float) -> tuple[int, float]:
    residues = [r for r in chain if r.id[0] == " "]
    n = len(residues) or 1
    charged: list[tuple[int, object, str]] = []
    for r in residues:
        aa = _aa1(r)
        if aa in NEG:
            charged.append((-1, r, aa))
        elif aa in POS:
            charged.append((1, r, aa))

    count = 0
    for i, (c1, r1, _) in enumerate(charged):
        for c2, r2, _ in charged[i + 1 :]:
            if c1 * c2 != -1:
                continue
            if _min_heavy_distance(r1, r2) <= cutoff:
                count += 1
    return count, count / n


def packing_density_from_chain(struct, chain) -> tuple[float, float]:
    """Return (mean_rsa, packing_density) with packing_density = 1 - mean(RSA)."""
    from Bio.PDB.SASA import ShrakeRupley

    sr = ShrakeRupley()
    sr.compute(struct, level="R")
    rsas: list[float] = []
    for res in chain:
        if res.id[0] != " ":
            continue
        aa = _aa1(res)
        sasa = float(getattr(res, "sasa", 0.0) or 0.0)
        denom = MAX_ASA.get(aa, 200.0)
        rsas.append(min(1.0, sasa / denom) if denom > 0 else 0.0)
    if not rsas:
        return 0.0, 0.0
    mean_rsa = float(sum(rsas) / len(rsas))
    return mean_rsa, 1.0 - mean_rsa


def compute_chain_features(
    struct_path: Path,
    chain_id: str,
    *,
    ion_cutoff: float,
) -> dict:
    from Bio.PDB import MMCIFParser, PDBParser

    if struct_path.suffix == ".cif" or struct_path.name.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", str(struct_path))
    model = next(struct.get_models())
    ch = chain_id[-1] if "_" in chain_id else chain_id
    if ch not in model:
        for c in model:
            if c.id.upper() == ch.upper():
                ch = c.id
                break
        else:
            raise ValueError(f"chain {chain_id} not in structure")
    chain = model[ch]
    n_pairs, ion_density = ion_pairs_from_chain(chain, ion_cutoff)
    mean_rsa, packing = packing_density_from_chain(struct, chain)
    n_res = sum(1 for r in chain if r.id[0] == " ")
    return {
        "ion_pair_count": int(n_pairs),
        "ion_pair_density": float(ion_density),
        "mean_sasa": float(mean_rsa),
        "packing_density": float(packing),
        "n_residues_struct": int(n_res),
        "sasa_method": "biopython_shrake_rupley",
    }


def load_cached(out_dir: Path, chain_id: str) -> dict:
    p = out_dir / f"{chain_id}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def ensure_structure_features(
    chain_ids: set[str],
    corpus_csv: Path,
    out_dir: Path,
    *,
    ion_cutoff: float = 4.0,
    pdb_dir: Path | None = None,
) -> None:
    ensure_dir(out_dir)
    pdb_dir = pdb_dir or resolve_path("data/raw/pdb")
    todo = sorted(chain_ids)
    print(f"  Structure features for {len(todo)} chains (ion pairs + SASA packing)...")
    for cid in todo:
        out_path = out_dir / f"{cid}.json"
        if out_path.exists():
            cached = json.loads(out_path.read_text())
            if cached.get("ion_pair_density") is not None and cached.get("packing_density") is not None:
                continue
        targets = [t for t in load_chain_targets(corpus_csv, None, {cid})]
        if not targets:
            print(f"  WARN structure features: no corpus row for {cid}", file=sys.stderr)
            continue
        t = targets[0]
        src = find_structure(pdb_dir, t["pdb_id"])
        if not src:
            print(f"  WARN structure features: no PDB for {cid}", file=sys.stderr)
            continue
        try:
            with tempfile.TemporaryDirectory(prefix="exp_d_sf_") as td:
                struct_path = materialize_structure(src, Path(td))
                feats = compute_chain_features(struct_path, t["chain"], ion_cutoff=ion_cutoff)
                payload = {"chain_id": cid, "pdb_id": t["pdb_id"], "chain": t["chain"], **feats}
                write_json(out_path, payload)
        except Exception as e:
            print(f"  WARN structure features {cid}: {e}", file=sys.stderr)


def merge_structure_features(dssp_feats: dict, struct_feats: dict) -> dict:
    out = dict(dssp_feats)
    for k in ("ion_pair_count", "ion_pair_density", "packing_density", "mean_sasa"):
        if struct_feats.get(k) is not None:
            out[k] = struct_feats[k]
    if out.get("packing_density") is not None and out.get("packing_proxy") is None:
        out["packing_proxy"] = out["packing_density"]
    return out
