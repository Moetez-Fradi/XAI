"""Generate PyMOL session for Exp D thermo/meso pair visualization.

Colors cartoon by per-residue solvent exposure (RSA) from DSSP — a thermostability
correlate. Also writes charged-exposure binary coloring.

Outputs under plot_scripts/outputs/pymol/:
  exp_d_groel_session.pml
  exp_d_<meso>_rsa.csv, exp_d_<thermo>_rsa.csv
  exp_d_<meso>_charged.csv, exp_d_<thermo>_charged.csv

Run in PyMOL:
  pymol plot_scripts/outputs/pymol/exp_d_groel_session.pml
  # then: png exp_d_groel_rsa.png, 2400, 1800
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import BIO_ROOT, PYMOL_OUT, RESULTS, ensure_out
ANNOT = BIO_ROOT / 'data' / 'annotations' / 'dssp'
PDB_DIR = BIO_ROOT / 'data' / 'raw' / 'pdb'

def pick_pair(metrics: dict) -> dict:
    lit = [p for p in metrics['pairs'] if p.get('source') == 'literature' and p.get('xq_profile_spearman') is not None]
    if not lit:
        lit = metrics['pairs']
    return max(lit, key=lambda p: float(p['xq_profile_spearman']))

def load_residue_table(chain_id: str) -> list[dict]:
    p = ANNOT / f'{chain_id}.json'
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return data.get('residues') or []

def write_color_csv(rows: list[dict], field: str, out: Path) -> None:
    with out.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['resi', 'value'])
        for r in rows:
            w.writerow([r['resseq'], f'{float(r.get(field) or 0):.4f}'])

def write_charged_csv(rows: list[dict], out: Path) -> None:
    charged = set('DEKR')
    with out.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['resi', 'value'])
        for r in rows:
            aa = r.get('aa', 'X')
            rsa = float(r.get('rsa') or 0)
            val = 1.0 if aa in charged and rsa > 0.25 else 0.0
            w.writerow([r['resseq'], f'{val:.1f}'])

def find_structure(pdb_id: str) -> str | None:
    for name in (f'{pdb_id.upper()}.cif.gz', f'{pdb_id.upper()}.cif', f'{pdb_id.lower()}.cif.gz'):
        p = PDB_DIR / name
        if p.exists():
            return str(p.relative_to(BIO_ROOT))
    return None

def main() -> int:
    ensure_out()
    metrics_path = RESULTS / 'exp_d' / 'metrics.json'
    if not metrics_path.exists():
        print('ERROR: run Exp D first', file=sys.stderr)
        return 1
    metrics = json.loads(metrics_path.read_text())
    pair = pick_pair(metrics)
    meso, thermo = (pair['meso_chain_id'], pair['thermo_chain_id'])
    meso_pdb, thermo_pdb = (meso.split('_')[0], thermo.split('_')[0])
    meso_ch, thermo_ch = (meso.split('_')[1], thermo.split('_')[1])
    meso_rows = load_residue_table(meso)
    thermo_rows = load_residue_table(thermo)
    if not meso_rows or not thermo_rows:
        print('ERROR: missing DSSP annotations with residues. Re-run exp_d_thermo.py', file=sys.stderr)
        return 1
    prefix = PYMOL_OUT / 'exp_d'
    meso_rsa = Path(f'{prefix}_{meso}_rsa.csv')
    thermo_rsa = Path(f'{prefix}_{thermo}_rsa.csv')
    meso_chg = Path(f'{prefix}_{meso}_charged.csv')
    thermo_chg = Path(f'{prefix}_{thermo}_charged.csv')
    write_color_csv(meso_rows, 'rsa', meso_rsa)
    write_color_csv(thermo_rows, 'rsa', thermo_rsa)
    write_charged_csv(meso_rows, meso_chg)
    write_charged_csv(thermo_rows, thermo_chg)
    meso_struct = find_structure(meso_pdb)
    thermo_struct = find_structure(thermo_pdb)
    if not meso_struct or not thermo_struct:
        print('ERROR: PDB files not found in data/raw/pdb/', file=sys.stderr)
        return 1
    pml = PYMOL_OUT / 'exp_d_groel_session.pml'
    rel = lambda p: str(p.relative_to(BIO_ROOT))
    lines = ['# Exp D PyMOL — meso vs thermo GroEL (RSA coloring)', f"# Pair: {meso} vs {thermo} | xq_spearman={pair.get('xq_profile_spearman')}", 'reinitialize', 'bg_color white', 'set cartoon_fancy_helices, 1', '', f'load {meso_struct}, meso', f'remove meso and not chain {meso_ch}', 'create meso_chain, meso', 'delete meso', '', f'load {thermo_struct}, thermo', f'remove thermo and not chain {thermo_ch}', 'create thermo_chain, thermo', 'delete thermo', '', 'python', 'import csv', 'from pymol import cmd', f"for obj, csvpath in [('meso_chain', r'{rel(meso_rsa)}'), ('thermo_chain', r'{rel(thermo_rsa)}')]:", '    with open(csvpath) as f:', '        for row in csv.DictReader(f):', "            resi = row['resi']", "            val = float(row['value'])", "            cmd.alter(f'{obj} and resi {resi}', f'b={val}')", '    cmd.sort(obj)', 'python end', '', 'spectrum b, blue_white_red, meso_chain, minimum=0, maximum=1', 'spectrum b, blue_white_red, thermo_chain, minimum=0, maximum=1', 'show cartoon, meso_chain thermo_chain', 'align thermo_chain, meso_chain', 'translate [-25,0,0], meso_chain', 'translate [25,0,0], thermo_chain', 'zoom meso_chain thermo_chain, 5', '', '# Export: png plot_scripts/outputs/pymol/exp_d_groel_rsa.png, 2400, 1800']
    pml.write_text('\n'.join(lines) + '\n')
    pml2 = PYMOL_OUT / 'exp_d_groel_charged_session.pml'
    lines2 = ['# Exp D PyMOL — charged exposed surface (red)', f'# Pair: {meso} vs {thermo}', 'reinitialize', 'bg_color white', f'load {meso_struct}, meso', f'remove meso and not chain {meso_ch}', 'create meso_chain, meso', 'delete meso', f'load {thermo_struct}, thermo', f'remove thermo and not chain {thermo_ch}', 'create thermo_chain, thermo', 'delete thermo', 'python', 'import csv', 'from pymol import cmd', f"for obj, csvpath in [('meso_chain', r'{rel(meso_chg)}'), ('thermo_chain', r'{rel(thermo_chg)}')]:", '    with open(csvpath) as f:', '        for row in csv.DictReader(f):', "            resi = row['resi']", "            val = float(row['value'])", "            cmd.alter(f'{obj} and resi {resi}', f'b={val}')", '    cmd.sort(obj)', 'python end', 'color gray80, meso_chain thermo_chain', 'spectrum b, white_red, meso_chain thermo_chain, minimum=0, maximum=1', 'show cartoon, meso_chain thermo_chain', 'align thermo_chain, meso_chain', 'translate [-25,0,0], meso_chain', 'translate [25,0,0], thermo_chain', 'zoom meso_chain thermo_chain, 5', '# Export: png plot_scripts/outputs/pymol/exp_d_groel_charged.png, 2400, 1800']
    pml2.write_text('\n'.join(lines2) + '\n')
    print(f'  → {pml.relative_to(BIO_ROOT)}')
    print(f'  → {pml2.relative_to(BIO_ROOT)}')
    print(f"  Pair: {meso} ↔ {thermo} (xq_spearman={pair.get('xq_profile_spearman'):.3f})")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
