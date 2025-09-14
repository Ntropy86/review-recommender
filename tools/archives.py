#!/usr/bin/env python3
"""
Move non-essential scripts in ./nlp to ./nlp/_archive with safety guards.

Defaults to KEEP these files (edit as you like):
  - 10_product_prep.py
  - 11_build_product_embeddings.py
  - 12_product_prep.py
  - 21_build_review_embeddings.py   (future/new)
  - test.py                         (helper in nlp/)
Everything else in nlp/ (except _archive/) is moved.

Usage examples:
  python tools/archive_unused_scripts.py --dry-run
  python tools/archive_unused_scripts.py --list
  python tools/archive_unused_scripts.py --keep 04_embeddings.py,05_clustering.py
  python tools/archive_unused_scripts.py --keep-pattern "*/keep_*.py,*/experiments/*.py"
"""

from __future__ import annotations
import argparse, os, re, shutil
from pathlib import Path
from datetime import datetime
import fnmatch

ROOT = Path(__file__).resolve().parents[1]  # repo root (assumes tools/…)
NLP_DIR = ROOT / "nlp"
ARCHIVE_DIR = NLP_DIR / "_archive"

DEFAULT_KEEP = {
    "10_product_prep.py",
    "11_build_product_embeddings.py",
    "12_product_prep.py",
    "21_build_review_embeddings.py",
    "test.py",
}

def parse_args():
    ap = argparse.ArgumentParser(description="Archive unused scripts from nlp/ to nlp/_archive.")
    ap.add_argument("--keep", type=str, default="",
                    help="Comma-separated filenames (relative to nlp/) to KEEP (additive to defaults).")
    ap.add_argument("--keep-pattern", type=str, default="",
                    help="Comma-separated glob patterns to KEEP (e.g., '*/keep_*.py,*/experiments/*.py').")
    ap.add_argument("--ext", type=str, default=".py",
                    help="File extension to consider (default: .py). Use '*' to include all files.")
    ap.add_argument("--dry-run", action="store_true", help="Show what would move; make no changes.")
    ap.add_argument("--list", action="store_true", help="Only list candidates; no moves.")
    return ap.parse_args()

def should_keep(p: Path, keep_names: set[str], keep_globs: list[str]) -> bool:
    name = p.name
    rel = p.relative_to(NLP_DIR).as_posix()
    if name in keep_names:
        return True
    for pat in keep_globs:
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(name, pat):
            return True
    return False

def unique_dest(dst_dir: Path, name: str) -> Path:
    base = name
    stem, dot, ext = base.partition(".")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = dst_dir / name
    if not candidate.exists():
        return candidate
    # add timestamp; if still exists, add counter
    candidate = dst_dir / f"{stem}.{ts}.{ext or 'py'}"
    i = 1
    while candidate.exists():
        candidate = dst_dir / f"{stem}.{ts}.{i}.{ext or 'py'}"
        i += 1
    return candidate

def main():
    args = parse_args()
    if not NLP_DIR.exists():
        raise SystemExit(f"[ERR] Missing directory: {NLP_DIR}")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    add_keep = {s.strip() for s in args.keep.split(",") if s.strip()}
    keep_names = set(DEFAULT_KEEP) | add_keep
    keep_globs = [s.strip() for s in args.keep_pattern.split(",") if s.strip()]

    # collect candidates
    consider_all = args.ext == "*"
    to_move = []
    kept = []
    for p in sorted(NLP_DIR.glob("*")):
        if p.name == "_archive":
            continue
        if not p.is_file():
            continue
        if not consider_all and p.suffix != args.ext:
            continue
        if should_keep(p, keep_names, keep_globs):
            kept.append(p)
        else:
            to_move.append(p)

    # report
    print(f"[nlp] base: {NLP_DIR}")
    print(f"[archive] dest: {ARCHIVE_DIR}")
    print(f"[keep] defaults: {sorted(DEFAULT_KEEP)}")
    if add_keep:
        print(f"[keep] extra:    {sorted(add_keep)}")
    if keep_globs:
        print(f"[keep] patterns: {keep_globs}")
    print(f"[scan] kept={len(kept)}  to_move={len(to_move)}\n")

    if args.list or args.dry_run:
        if kept:
            print("Kept:")
            for k in kept:
                print("  -", k.name)
        if to_move:
            print("\nWould move:")
            for m in to_move:
                print("  -", m.name, "->", unique_dest(ARCHIVE_DIR, m.name).name)
        if args.list:
            return
        print("\n[dry-run] No changes made.")
        return

    # execute moves
    moved = 0
    for src in to_move:
        dst = unique_dest(ARCHIVE_DIR, src.name)
        shutil.move(str(src), str(dst))
        print(f"[move] {src.name} -> _archive/{dst.name}")
        moved += 1

    print(f"\n[done] moved {moved} file(s). Kept {len(kept)} file(s).")

if __name__ == "__main__":
    main()
