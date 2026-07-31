"""Run mkdssp on corpus chains and write structural feature tables.

Writes:
  data/annotations/dssp/<chain_id>.json   # per-residue + aggregates
  data/annotations/dssp_chain_features.csv
  data/annotations/dssp_checkpoint.json   # resume cursor

Resume skips existing JSON unless --restart.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import BIO_ROOT, BatchProgress, ensure_dir, load_yaml, raw_dir, resolve_path, write_json
KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
MAX_ASA = {'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0, 'C': 167.0, 'Q': 225.0, 'E': 223.0, 'G': 104.0, 'H': 224.0, 'I': 197.0, 'L': 201.0, 'K': 236.0, 'M': 224.0, 'F': 240.0, 'P': 159.0, 'S': 155.0, 'T': 172.0, 'W': 285.0, 'Y': 263.0, 'V': 174.0}

def find_mkdssp(cfg_path: str | Path | None) -> str:
    candidates: list[Path] = []
    pref = str(cfg_path or '').strip()
    if pref and pref.lower() not in {'auto', 'mkdssp', 'dssp'}:
        candidates.append(resolve_path(pref))
    candidates.extend([BIO_ROOT / 'tools/mamba/envs/bio-tools/bin/mkdssp', BIO_ROOT / 'tools/mamba/envs/bio-tools/bin/dssp', Path('/usr/bin/mkdssp'), Path('/usr/bin/dssp')])
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and os.access(p, os.X_OK):
            return str(p.resolve())
    which = shutil.which('mkdssp') or shutil.which('dssp')
    if which:
        return which
    cfg_label = '(auto)' if pref.lower() in {'auto', 'mkdssp', 'dssp', ''} else str(resolve_path(pref))
    raise FileNotFoundError(f'mkdssp/dssp not found. Install one of:\n  Project conda (recommended):\n    export MAMBA_ROOT_PREFIX=$PWD/tools/mamba\n    ./tools/bin/micromamba install -y -n bio-tools -c conda-forge -c bioconda dssp\n  Or: ./fetch_tools_linux.sh\n  Arch AUR (not in official repos): yay -S dssp   OR   paru -S dssp\n  Debian/Ubuntu: sudo apt-get install dssp\nConfigured: {cfg_label}')

def find_structure(pdb_dir: Path, pdb_id: str) -> Path | None:
    for name in (f'{pdb_id.upper()}.cif.gz', f'{pdb_id.upper()}.cif', f'{pdb_id}.cif.gz', f'{pdb_id}.cif', f'{pdb_id.upper()}.pdb'):
        cand = pdb_dir / name
        if cand.exists() and cand.stat().st_size > 0:
            return cand
    return None

def materialize_structure(src: Path, tmpdir: Path) -> Path:
    if src.name.endswith('.cif.gz'):
        dest = tmpdir / src.name[:-3]
        with gzip.open(src, 'rb') as fin, dest.open('wb') as fout:
            shutil.copyfileobj(fin, fout)
        return dest
    if src.suffix == '.gz':
        dest = tmpdir / src.stem
        with gzip.open(src, 'rb') as fin, dest.open('wb') as fout:
            shutil.copyfileobj(fin, fout)
        return dest
    return src

def chain_b_factors(structure_path: Path, chain_id: str) -> dict[int, float]:
    from Bio.PDB import MMCIFParser, PDBParser
    if structure_path.suffix == '.cif' or structure_path.name.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    struct = parser.get_structure('x', str(structure_path))
    model = next(struct.get_models())
    out: dict[int, float] = {}
    if chain_id not in model:
        for ch in model:
            if ch.id.upper() == chain_id.upper():
                chain_id = ch.id
                break
        else:
            return out
    chain = model[chain_id]
    for res in chain:
        if res.id[0] != ' ':
            continue
        vals = [atom.get_bfactor() for atom in res if atom.element != 'H' and atom.get_name() in {'CA', 'N', 'C', 'O'}]
        if not vals:
            vals = [atom.get_bfactor() for atom in res if atom.element != 'H']
        if vals:
            out[res.id[1]] = float(sum(vals) / len(vals))
    return out

def run_dssp_biopython(structure_path: Path, mkdssp: str, chain_id: str) -> list[dict]:
    from Bio.PDB import MMCIFParser, PDBParser, DSSP
    if structure_path.suffix == '.cif' or structure_path.name.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    struct = parser.get_structure('x', str(structure_path))
    model = next(struct.get_models())
    dssp = DSSP(model, str(structure_path), dssp=mkdssp)
    rows: list[dict] = []
    for key, val in dssp.property_dict.items():
        ch, res_id = key
        if str(ch).upper() != chain_id.upper():
            continue
        aa = val[1]
        ss = val[2]
        try:
            rsa = float(val[3])
            if rsa != rsa:
                rsa = 0.0
        except (TypeError, ValueError):
            rsa = 0.0
        resseq = res_id[1]
        rows.append({'resseq': int(resseq), 'aa': aa if aa != 'X' else 'X', 'ss': ss, 'rsa': rsa})
    rows.sort(key=lambda r: r['resseq'])
    return rows

def parse_dssp_file(dssp_path: Path, chain_id: str) -> list[dict]:
    rows: list[dict] = []
    started = False
    with dssp_path.open() as f:
        for line in f:
            if line.startswith('  #  RESIDUE'):
                started = True
                continue
            if not started or len(line) < 35:
                continue
            ch = line[11].strip() if len(line) > 11 else ''
            if ch.upper() != chain_id.upper():
                parts = line.split()
                if len(parts) < 6:
                    continue
                continue
            aa = line[13].strip()
            ss = line[16].strip() or '-'
            try:
                acc = float(line[34:38])
            except ValueError:
                acc = 0.0
            try:
                resseq = int(line[5:10])
            except ValueError:
                continue
            if aa == '!' or not aa:
                continue
            max_asa = MAX_ASA.get(aa, 200.0)
            rows.append({'resseq': resseq, 'aa': aa, 'ss': ss, 'rsa': min(acc / max_asa, 1.5) if max_asa else 0.0, 'acc': acc})
    return rows

def binned_profile(residues: list[dict], bfactors: dict[int, float], n_bins: int) -> list[float]:
    n_bins = max(int(n_bins), 1)
    n_feat = 5
    out = [0.0] * (n_bins * n_feat)
    if not residues:
        return out
    bins: list[list[dict]] = [[] for _ in range(n_bins)]
    n = len(residues)
    for i, r in enumerate(residues):
        b = min(int(i * n_bins / n), n_bins - 1)
        bins[b].append(r)
    for bi, chunk in enumerate(bins):
        if not chunk:
            continue
        helix = sheet = coil = 0
        hydros: list[float] = []
        rsas: list[float] = []
        for r in chunk:
            ss = (r.get('ss') or '-')[0]
            if ss in {'H', 'G', 'I'}:
                helix += 1
            elif ss in {'E', 'B'}:
                sheet += 1
            else:
                coil += 1
            aa = r.get('aa', 'X')
            if aa in KD:
                hydros.append(KD[aa])
            if r.get('rsa') is not None:
                rsas.append(float(r['rsa']))
        m = len(chunk)
        base = bi * n_feat
        out[base + 0] = helix / m
        out[base + 1] = sheet / m
        out[base + 2] = coil / m
        out[base + 3] = float(sum(hydros) / len(hydros)) if hydros else 0.0
        out[base + 4] = float(sum(rsas) / len(rsas)) if rsas else 0.0
    return out

def aggregate(residues: list[dict], bfactors: dict[int, float]) -> dict:
    n = len(residues) or 1
    helix = sheet = coil = 0
    hydros: list[float] = []
    rsas: list[float] = []
    bfs: list[float] = []
    for r in residues:
        ss = (r.get('ss') or '-')[0]
        if ss in {'H', 'G', 'I'}:
            helix += 1
        elif ss in {'E', 'B'}:
            sheet += 1
        else:
            coil += 1
        aa = r.get('aa', 'X')
        if aa in KD:
            hydros.append(KD[aa])
        if r.get('rsa') is not None:
            rsas.append(float(r['rsa']))
        bf = bfactors.get(int(r['resseq']))
        if bf is not None:
            bfs.append(bf)
    return {'n_residues': len(residues), 'frac_helix': helix / n, 'frac_sheet': sheet / n, 'frac_coil': coil / n, 'mean_hydrophobicity': float(sum(hydros) / len(hydros)) if hydros else 0.0, 'mean_rsa': float(sum(rsas) / len(rsas)) if rsas else 0.0, 'mean_b_factor': float(sum(bfs) / len(bfs)) if bfs else 0.0}

def load_chain_targets(corpus_csv: Path, limit: int | None, chain_ids: set[str] | None):
    rows = []
    with corpus_csv.open() as f:
        for rec in csv.DictReader(f):
            cid = rec['chain_id']
            if chain_ids is not None and cid not in chain_ids:
                continue
            rows.append(rec)
            if limit is not None and len(rows) >= limit:
                break
    return rows

def _iter_exp_a_records(jsonl: Path):
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    legacy = jsonl.parent.parent / 'per_query.json'
    if legacy.exists():
        data = json.loads(legacy.read_text())
        for rec in data.get('queries') or []:
            yield rec

def chains_from_exp_a(jsonl: Path, max_neighbors: int) -> set[str]:
    out: set[str] = set()
    for rec in _iter_exp_a_records(jsonl):
        out.add(rec['query'])
        for n in (rec.get('neighbors') or [])[:max_neighbors]:
            out.add(n['chain_id'])
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--restart', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--from-exp-a', action='store_true', help='Only annotate chains appearing in Exp A results (+ optional train map set later)')
    ap.add_argument('--max-neighbors', type=int, default=None, help='Exp A neighbors to include with --from-exp-a (default: exp_b.yaml max_neighbors_per_query)')
    ap.add_argument('--also-train', type=int, default=None, help='Also annotate N train chains for dim-feature map (with --from-exp-a)')
    ap.add_argument('--store-residues', action='store_true', help='Store per-residue DSSP rows in each JSON (large; needed later for Exp E/PyMOL)')
    ap.add_argument('--store-profiles', action='store_true', help='Store compact binned structural profiles for Exp B v2 (default n_bins from exp_b_v2.yaml)')
    ap.add_argument('--n-bins', type=int, default=None, help='Override profile bin count (with --store-profiles)')
    args = ap.parse_args()
    bcfg = load_yaml('exp_b.yaml')
    v2cfg: dict = {}
    try:
        v2cfg = load_yaml('exp_b_v2.yaml')
    except FileNotFoundError:
        pass
    corpus_cfg = load_yaml('corpus.yaml')
    smoke = bool(args.smoke)
    out_dir = ensure_dir(resolve_path(bcfg['annotations_dir']))
    features_csv = resolve_path(bcfg['features_table'])
    ckpt_path = resolve_path('data/annotations/dssp_checkpoint.json')
    mkdssp = find_mkdssp(bcfg.get('mkdssp_bin', 'tools/mamba/envs/bio-tools/bin/mkdssp'))
    pdb_dir = raw_dir('pdb')
    corpus_dir = resolve_path(corpus_cfg['smoke_outdir'] if smoke else corpus_cfg['paper_outdir'])
    corpus_csv = corpus_dir / 'chains.csv'
    if args.restart:
        print(f'--restart: clearing {out_dir}')
        if out_dir.exists():
            shutil.rmtree(out_dir)
        ensure_dir(out_dir)
        ckpt_path.unlink(missing_ok=True)
        features_csv.unlink(missing_ok=True)
    chain_filter: set[str] | None = None
    if args.from_exp_a:
        jsonl = resolve_path(bcfg['exp_a_smoke_jsonl'] if smoke else bcfg['exp_a_jsonl'])
        max_neigh = args.max_neighbors
        if max_neigh is None:
            max_neigh = int(bcfg.get('max_neighbors_per_query', 5))
        chain_filter = chains_from_exp_a(jsonl, max_neigh)
        n_train = args.also_train
        if n_train is None:
            n_train = int(bcfg.get('n_train_for_map', 3000)) if not smoke else 80
        train_ids = []
        with corpus_csv.open() as f:
            for rec in csv.DictReader(f):
                if rec.get('split') == 'train':
                    train_ids.append(rec['chain_id'])
        import random
        rng = random.Random(int(bcfg.get('map_random_seed', 42)))
        rng.shuffle(train_ids)
        chain_filter |= set(train_ids[:n_train])
        print(f'Target chains from Exp A + {n_train} train: {len(chain_filter)}')
    targets = load_chain_targets(corpus_csv, args.limit, chain_filter)
    if not targets:
        print('ERROR: no chains to annotate', file=sys.stderr)
        return 1

    def needs_work(cid: str) -> bool:
        jp = out_dir / f'{cid}.json'
        if args.restart or not jp.exists():
            return True
        if args.store_profiles:
            try:
                data = json.loads(jp.read_text())
            except json.JSONDecodeError:
                return True
            if not data.get('binned_profile'):
                return True
        return False
    pending = [t for t in targets if needs_work(t['chain_id'])]
    if args.store_profiles and (not args.restart):
        n_bins = int(args.n_bins if args.n_bins is not None else v2cfg.get('n_bins', 32))
        backfill = 0
        for t in targets:
            cid = t['chain_id']
            if any((p['chain_id'] == cid for p in pending)):
                continue
            jp = out_dir / f'{cid}.json'
            if not jp.exists():
                continue
            data = json.loads(jp.read_text())
            if data.get('binned_profile'):
                continue
            residues = data.get('residues')
            if not residues:
                continue
            data['binned_profile'] = {'n_bins': n_bins, 'features': ['frac_helix', 'frac_sheet', 'frac_coil', 'mean_hydrophobicity', 'mean_rsa'], 'values': binned_profile(residues, {}, n_bins)}
            write_json(jp, data)
            backfill += 1
        if backfill:
            print(f'Backfilled binned_profile on {backfill} existing JSONs (from residues)')
    print(f'DSSP annotate: {len(targets)} targets, {len(pending)} pending, mkdssp={mkdssp}')
    progress = BatchProgress(max(len(pending), 1), label='dssp')
    t0 = time.time()
    ok = fail = 0
    if not pending:
        progress.tick(skipped=True, force_print=True)
    for i, rec in enumerate(pending):
        cid = rec['chain_id']
        pdb_id = rec['pdb_id']
        chain = rec['chain']
        src = find_structure(pdb_dir, pdb_id)
        if src is None:
            fail += 1
            progress.tick(ok=False, force_print=i + 1 >= len(pending))
            continue
        try:
            with tempfile.TemporaryDirectory(prefix='dssp_') as td:
                tdir = Path(td)
                struct_path = materialize_structure(src, tdir)
                residues = run_dssp_biopython(struct_path, mkdssp, chain)
                if not residues:
                    dssp_out = tdir / 'out.dssp'
                    subprocess.run([mkdssp, str(struct_path), str(dssp_out)], check=True, capture_output=True, text=True)
                    residues = parse_dssp_file(dssp_out, chain)
                bfac = chain_b_factors(struct_path, chain)
                agg = aggregate(residues, bfac)
                payload = {'chain_id': cid, 'pdb_id': pdb_id, 'chain': chain, 'split': rec.get('split'), 'scop_fold': rec.get('scop_fold'), 'structure_file': str(src.relative_to(BIO_ROOT)), 'features': agg, 'n_residues_dssp': len(residues)}
                if args.store_residues:
                    payload['residues'] = residues
                if args.store_profiles:
                    n_bins = int(args.n_bins if args.n_bins is not None else v2cfg.get('n_bins', 32))
                    payload['binned_profile'] = {'n_bins': n_bins, 'features': ['frac_helix', 'frac_sheet', 'frac_coil', 'mean_hydrophobicity', 'mean_rsa'], 'values': binned_profile(residues, bfac, n_bins)}
                write_json(out_dir / f'{cid}.json', payload)
                ok += 1
                progress.tick(ok=True, force_print=i + 1 >= len(pending))
        except Exception as e:
            fail += 1
            if fail <= 15:
                print(f'  FAIL {cid}: {e}', file=sys.stderr)
            progress.tick(ok=False, force_print=i + 1 >= len(pending))
        if (i + 1) % 50 == 0:
            write_json(ckpt_path, {'pending_done': i + 1, 'ok': ok, 'fail': fail, 'updated_unix': int(time.time())})
    feat_names = list(bcfg.get('features') or [])
    json_files = sorted(out_dir.glob('*.json'))
    with features_csv.open('w', newline='') as f:
        fields = ['chain_id', 'pdb_id', 'chain', 'split', 'scop_fold', 'n_residues', *feat_names]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for jp in json_files:
            data = json.loads(jp.read_text())
            feats = data.get('features') or {}
            row = {'chain_id': data['chain_id'], 'pdb_id': data.get('pdb_id', ''), 'chain': data.get('chain', ''), 'split': data.get('split', ''), 'scop_fold': data.get('scop_fold', ''), 'n_residues': feats.get('n_residues', data.get('n_residues_dssp', 0))}
            for name in feat_names:
                row[name] = feats.get(name, '')
            w.writerow(row)
    write_json(ckpt_path, {'complete': True, 'n_json': len(json_files), 'ok_this_run': ok, 'fail_this_run': fail, 'features_table': str(features_csv.relative_to(BIO_ROOT)), 'elapsed_s': round(time.time() - t0, 1), 'updated_unix': int(time.time())})
    print(f'DSSP done: wrote {len(json_files)} chain JSONs, table={features_csv} (ok={ok} fail={fail} this run)')
    return 0 if ok or json_files else 1
if __name__ == '__main__':
    raise SystemExit(main())
