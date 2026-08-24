#!/usr/bin/env python3
"""Copy regenerated experiment plots into paper/figures/ (opt-in).

By default the submission figures stay frozen. Pass --force to overwrite
a named figure from the newest matching experiments/*/plots/*.png.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
PAPER_FIG = ROOT / "paper" / "figures"

# newest plot stem -> paper filename
MAP = {
    "fig02_cost_model": "fig02_cost_model.png",
    "fig06_systems_load": "fig06_systems_load.png",
    "fig08_minilm_m1": "fig08_minilm_m1.png",
    "fig12_scale_ef": "fig12_scale_ef.png",
    "fig13_focus_baselines": "fig13_focus_baselines.png",
    "fig17_x4_jaccard": "fig17_x4_jaccard.png",
    "fig20_filter_m2": "fig20_filter_m2.png",
    "fig21_sq_char": "fig21_sq_char.png",
    "fig22_projected_ceiling": "fig22_projected_ceiling.png",
    "fig23_case_study": "fig23_case_study.png",
    "option1_recall_vs_ratio_d128": "fig07_m1_sift.png",
    "option2_recall_heatmap_d128": "fig09_m2_sift_recall.png",
    "option2_speedup_heatmap_d128": "fig11_m2_sift_speedup.png",
    "option2_recall_heatmap_d768": "fig10_m2_d768_recall.png",
    "xcut1_kernel_gather_vs_repack": "fig14_k1_kernels.png",
}


def newest(stem: str) -> Path | None:
    hits = sorted(EXPERIMENTS.glob(f"*/plots/{stem}.png"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="overwrite paper/figures")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    for stem, dest_name in MAP.items():
        src = newest(stem)
        dest = PAPER_FIG / dest_name
        if src is None:
            print(f"  missing {stem}")
            continue
        print(f"  {src.relative_to(ROOT)} -> paper/figures/{dest_name}")
        if args.dry_run:
            continue
        if dest.exists() and not args.force:
            print("    skipped (frozen; pass --force)")
            continue
        shutil.copy2(src, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
