"""Shared paths and matplotlib style for paper figures."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
BIO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = BIO_ROOT / 'results'
OUTDIR = BIO_ROOT / 'plot_scripts' / 'outputs'
PYMOL_OUT = OUTDIR / 'pymol'
PALETTE = {'xqdrant': '#2563eb', 'esm2': '#7c3aed', 'blast': '#059669', 'foldseek': '#d97706', 'tmalign': '#dc2626', 'same_fold': '#2563eb', 'diff_fold': '#94a3b8', 'pos': '#059669', 'neg': '#dc2626', 'null': '#cbd5e1', 'literature': '#2563eb', 'structure_map': '#059669', 'auto_fold': '#94a3b8'}

def setup_style() -> None:
    mpl.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.spines.top': False, 'axes.spines.right': False})

def ensure_out() -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PYMOL_OUT.mkdir(parents=True, exist_ok=True)
    return OUTDIR

def save(fig: plt.Figure, name: str) -> Path:
    ensure_out()
    png = OUTDIR / f'{name}.png'
    pdf = OUTDIR / f'{name}.pdf'
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print(f'  → {png.relative_to(BIO_ROOT)}')
    print(f'  → {pdf.relative_to(BIO_ROOT)}')
    return png

def load_json(path: Path) -> dict:
    return json.loads(path.read_text())

def read_tsv(path: Path) -> list[dict[str, str]]:
    import csv
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))
