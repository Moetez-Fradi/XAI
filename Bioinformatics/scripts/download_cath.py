#!/usr/bin/env python3
"""Download CATH domain lists → data/raw/cath/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import (  # noqa: E402
    download_cfg,
    download_file,
    gunzip_to,
    raw_dir,
    write_json,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = download_cfg()["cath"]
    out = raw_dir("cath")
    files_meta: list[dict] = []
    for item in cfg["files"]:
        dest = out / item["name"]
        download_file(item["url"], dest, force=args.force)
        entry = {"name": item["name"], "url": item["url"]}
        if dest.name.endswith(".gz"):
            plain = gunzip_to(dest)
            entry["uncompressed"] = plain.name
        files_meta.append(entry)

    write_json(
        out / "download_manifest.json",
        {"source": "CATH", "files": files_meta, "outdir": str(out)},
    )
    print(f"CATH ready under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
