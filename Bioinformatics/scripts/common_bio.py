"""Shared paths and HTTP helpers for Bioinformatics download / experiment scripts."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

BIO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = BIO_ROOT / "configs"


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIGS / name
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def resolve_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (BIO_ROOT / p)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def paths_cfg() -> dict[str, Any]:
    return load_yaml("paths.yaml")


def download_cfg() -> dict[str, Any]:
    return load_yaml("download.yaml")


def raw_dir(key: str) -> Path:
    """Return data/raw/<key> using paths.yaml when present."""
    cfg = paths_cfg()
    mapping = {
        "scop": cfg.get("scop_dir", "data/raw/scop"),
        "cath": cfg.get("cath_dir", "data/raw/cath"),
        "pdb": cfg.get("pdb_dir", "data/raw/pdb"),
        "uniprot": cfg.get("uniprot_dir", "data/raw/uniprot"),
        "pfam": cfg.get("pfam_dir", "data/raw/pfam"),
        "thermo": cfg.get("thermo_dir", "data/raw/thermo"),
        "sifts": cfg.get("sifts_dir", "data/raw/sifts"),
    }
    if key not in mapping:
        raise KeyError(f"Unknown raw dir key: {key}")
    return ensure_dir(resolve_path(mapping[key]))


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def format_bytes(n: float) -> str:
    n = float(max(0, n))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} TB"


def format_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN
        return "?"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class BatchProgress:
    """Track multi-item download progress with % and ETA."""

    def __init__(self, total: int, label: str = "progress") -> None:
        self.total = max(total, 1)
        self.label = label
        self.done = 0
        self.failed = 0
        self.skipped = 0
        self.bytes_downloaded = 0
        self.t0 = time.time()
        self._last_print = 0.0

    def tick(
        self,
        *,
        ok: bool = True,
        skipped: bool = False,
        nbytes: int = 0,
        force_print: bool = False,
        every_s: float = 2.0,
    ) -> None:
        if skipped:
            self.skipped += 1
        elif ok:
            self.done += 1
        else:
            self.failed += 1
        self.bytes_downloaded += max(0, nbytes)
        now = time.time()
        finished = self.completed >= self.total
        if force_print or finished or (now - self._last_print) >= every_s:
            self.print_status()
            self._last_print = now

    @property
    def completed(self) -> int:
        return self.done + self.failed + self.skipped

    def print_status(self) -> None:
        elapsed = max(time.time() - self.t0, 1e-6)
        pct = 100.0 * self.completed / self.total
        rate = self.completed / elapsed  # items/s
        remaining = max(self.total - self.completed, 0)
        eta = remaining / rate if rate > 0 else float("inf")
        bps = self.bytes_downloaded / elapsed
        print(
            f"  [{self.label}] {self.completed}/{self.total} ({pct:5.1f}%) | "
            f"ok={self.done} skip={self.skipped} fail={self.failed} | "
            f"{format_bytes(self.bytes_downloaded)} @ {format_bytes(bps)}/s | "
            f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}",
            flush=True,
        )


def _download_with_curl(
    url: str,
    tmp: Path,
    *,
    timeout: int,
    max_retries: int,
    user_agent: str,
    quiet: bool,
    label: str,
) -> None:
    """Prefer curl: resume (-C -), retries, and better FTP/HTTP stall handling."""
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        str(max(max_retries, 5)),
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        str(min(timeout, 60)),
        "--max-time",
        "0",  # no overall cap; large Pfam/Swiss-Prot need hours
        "-A",
        user_agent,
        "-C",
        "-",
        "-o",
        str(tmp),
        url,
    ]
    if quiet:
        cmd.insert(1, "-sS")
    else:
        cmd.insert(1, "--progress-bar")
        print(f"  GET (curl resume) {label}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise OSError(f"curl failed ({proc.returncode}) for {url}")
    if not tmp.exists() or tmp.stat().st_size <= 0:
        raise OSError(f"curl produced empty file for {url}")


def _download_with_urllib(
    url: str,
    tmp: Path,
    *,
    timeout: int,
    max_retries: int,
    user_agent: str,
    quiet: bool,
    label: str,
) -> None:
    """urllib fallback with HTTP Range resume + Content-Length checks."""
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            existing = tmp.stat().st_size if tmp.exists() else 0
            headers = {"User-Agent": user_agent}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
            if not quiet:
                resume = f" resume@{format_bytes(existing)}" if existing else ""
                print(f"  GET [{attempt}/{max_retries}] {label}{resume}", flush=True)
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                total_hdr = resp.headers.get("Content-Length")
                total_n = int(total_hdr) if total_hdr and total_hdr.isdigit() else None
                # 206 Partial Content: Content-Length is remaining bytes
                if status == 206 and total_n is not None:
                    expected_final = existing + total_n
                    mode = "ab"
                    got = existing
                elif status == 200 and existing > 0:
                    # Server ignored Range — restart
                    mode = "wb"
                    got = 0
                    expected_final = total_n
                else:
                    mode = "wb" if existing == 0 else "ab"
                    got = 0 if mode == "wb" else existing
                    expected_final = total_n if mode == "wb" else (
                        existing + total_n if total_n is not None else None
                    )
                t0 = time.time()
                last_ui = 0.0
                with tmp.open(mode) as out:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        got += len(chunk)
                        now = time.time()
                        if (not quiet) and (now - last_ui) >= 0.5:
                            elapsed = max(now - t0, 1e-6)
                            speed = (got - (existing if mode == "ab" else 0)) / elapsed
                            if expected_final:
                                pct = 100.0 * got / expected_final
                                eta = (
                                    (expected_final - got) / speed
                                    if speed > 0
                                    else float("inf")
                                )
                                msg = (
                                    f"\r    {label}: {pct:5.1f}% "
                                    f"{format_bytes(got)}/{format_bytes(expected_final)} "
                                    f"@ {format_bytes(speed)}/s ETA {format_duration(eta)}   "
                                )
                            else:
                                msg = (
                                    f"\r    {label}: {format_bytes(got)} "
                                    f"@ {format_bytes(speed)}/s (size unknown)   "
                                )
                            sys.stdout.write(msg)
                            sys.stdout.flush()
                            last_ui = now
                if not quiet and expected_final:
                    sys.stdout.write("\n")
                final_size = tmp.stat().st_size
                if expected_final is not None and final_size != expected_final:
                    raise OSError(
                        f"incomplete download of {label}: got {final_size} bytes, "
                        f"expected {expected_final}"
                    )
            return
        except (HTTPError, URLError, TimeoutError, OSError) as e:
            last_err = e
            if not quiet:
                print(f"  warn: {e}", file=sys.stderr)
            # Keep .partial for resume on next attempt
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def download_file(
    url: str,
    dest: Path,
    *,
    force: bool = False,
    timeout: int = 120,
    max_retries: int = 3,
    user_agent: str = "XQdrant-Bioinformatics/0.1 (research; +local)",
    quiet: bool = False,
    progress_label: str | None = None,
) -> Path:
    """Download url → dest with retries, resume, byte progress, % and ETA."""
    ensure_dir(dest.parent)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        if not quiet:
            print(f"  skip (exists): {dest.name} ({format_bytes(dest.stat().st_size)})")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".partial")
    if force and tmp.exists():
        tmp.unlink(missing_ok=True)
    label = progress_label or dest.name

    try:
        if shutil.which("curl"):
            _download_with_curl(
                url,
                tmp,
                timeout=timeout,
                max_retries=max_retries,
                user_agent=user_agent,
                quiet=quiet,
                label=label,
            )
        else:
            _download_with_urllib(
                url,
                tmp,
                timeout=timeout,
                max_retries=max_retries,
                user_agent=user_agent,
                quiet=quiet,
                label=label,
            )
    except Exception:
        # Leave .partial in place for a later resume; do not delete.
        raise

    tmp.replace(dest)
    size = dest.stat().st_size
    if not quiet:
        print(f"  wrote {dest.name} ({format_bytes(size)})", flush=True)
    return dest


def gunzip_to(src_gz: Path, dest: Path | None = None) -> Path:
    import gzip
    import shutil

    if dest is None:
        if not src_gz.name.endswith(".gz"):
            raise ValueError(f"Expected .gz file: {src_gz}")
        dest = src_gz.with_suffix("")
    ensure_dir(dest.parent)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip decompress (exists): {dest}")
        return dest
    print(f"  decompress {src_gz.name} → {dest.name}")
    try:
        with gzip.open(src_gz, "rb") as f_in, dest.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return dest
