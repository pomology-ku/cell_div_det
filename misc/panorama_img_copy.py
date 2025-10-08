#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copy images by Kobetu ID from a mapping XML and an ID list.
- XML: <Kobetu ID="..." ImagePath="...">
- ID list: text file, 1 ID per line

Features:
- Only search in the given source directory (no subdir search)
- Dry-run / verbose / move / force
- Optional renaming: "<ID>_<basename>" (default) or keep original name
- Optional normalization toggle for XML ImagePath matching
"""

import argparse
import os
import sys
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(
        description="Copy images referenced by Kobetu IDs from XML mapping.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("-x", "--xml", required=True, help="Path to MappingInformation XML file")
    p.add_argument("-l", "--id-list", required=True, help="Text file: 1 Kobetu ID per line")
    p.add_argument("-s", "--src", required=True, help="Source directory (flat search only)")
    p.add_argument("-o", "--outdir", required=True, help="Output directory to copy images to")
    p.add_argument("-k", "--keep-name", action="store_true",
                   help="Keep original filename (default: rename to '<ID>_<basename>')")
    p.add_argument("-d", "--dry-run", action="store_true", help="Do not copy, just show actions")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logs")
    p.add_argument("-m", "--move", action="store_true", help="Move instead of copy")
    p.add_argument("-f", "--force", action="store_true",
                   help="Overwrite if a file with the same name exists in outdir")
    p.add_argument("-e", "--ext-fallback", default="",
                   help="If image not found, try these extensions (comma-separated, e.g. 'tif,jpg,png')")
    p.add_argument("--no-normalize", dest="do_normalize", action="store_false",
                   help="Do NOT normalize XML ImagePath before matching")
    p.set_defaults(do_normalize=True)
    return p.parse_args()

def read_id_list(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            ids.append(t)
    return ids

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mapping = {}
    for k in root.findall(".//Kobetu"):
        kid = k.attrib.get("ID")
        img = k.attrib.get("ImagePath")
        if kid and img:
            mapping[str(kid)] = img
    return mapping

def case_insensitive_match(candidates, target_name):
    """Return Path if target_name matches (case-insensitive) among candidates."""
    target_cf = target_name.casefold()
    for p in candidates:
        if p.name.casefold() == target_cf:
            return p
    return None

def normalize_filename(name: str) -> str:
    """
    Normalize XML ImagePath for matching:
    - Strip leading/trailing spaces
    - Remove leading numeric prefixes (e.g., '00002 ' -> '')
    """
    name = name.strip()
    parts = name.split(maxsplit=1)
    if parts and parts[0].isdigit() and len(parts) > 1:
        return parts[1]
    return name

def find_image(src_dir: Path, image_path: str, ext_candidates: list[str], do_normalize: bool) -> Path | None:
    # 正規化オプション
    if do_normalize:
        image_path = normalize_filename(image_path)

    rel = Path(image_path)
    cand = src_dir / rel.name
    if cand.exists():
        return cand

    # src_dir直下で大文字小文字無視マッチ
    files = [p for p in src_dir.iterdir() if p.is_file()]
    match = case_insensitive_match(files, rel.name)
    if match:
        return match

    # 拡張子fallback
    stem = rel.stem
    if ext_candidates:
        extset = {e.lower() for e in ext_candidates}
        for p in files:
            if p.stem == stem and p.suffix.lstrip(".").lower() in extset:
                return p

    return None

def safe_copy_or_move(src: Path, dst: Path, move: bool, force: bool, dry_run: bool, verbose: bool):
    if dst.exists():
        if force:
            if verbose or dry_run:
                print(f"[INFO] Overwriting: {dst}")
            if not dry_run:
                if dst.is_file():
                    dst.unlink()
                else:
                    shutil.rmtree(dst)
        else:
            raise FileExistsError(f"Destination exists: {dst}")

    if verbose or dry_run:
        op = "MOVE" if move else "COPY"
        print(f"[{op}] {src}  -->  {dst}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)

def main():
    args = parse_args()
    src_dir = Path(args.src)
    outdir = Path(args.outdir)

    if not src_dir.exists():
        print(f"[ERROR] Source directory not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    ids = read_id_list(args.id_list)
    mapping = parse_xml(args.xml)
    ext_fallbacks = [e.strip() for e in args.ext_fallback.split(",") if e.strip()] if args.ext_fallback else []

    if args.verbose:
        print(f"[INFO] IDs to process: {len(ids)}")
        print(f"[INFO] XML entries: {len(mapping)}")
        print(f"[INFO] Normalize ImagePath: {args.do_normalize}")

    missing_in_xml = []
    not_found_on_disk = []
    copied = 0

    for kid in ids:
        img_path = mapping.get(str(kid))
        if img_path is None:
            missing_in_xml.append(kid)
            if args.verbose:
                print(f"[WARN] ID {kid}: not found in XML")
            continue

        src_path = find_image(src_dir, img_path, ext_fallbacks, args.do_normalize)
        if src_path is None:
            not_found_on_disk.append((kid, img_path))
            if args.verbose:
                print(f"[WARN] ID {kid}: image not found for '{img_path}'")
            continue

        basename = src_path.name
        out_name = basename if args.keep_name else f"{kid}_{basename}"

        dst_path = outdir / out_name
        try:
            safe_copy_or_move(src_path, dst_path, move=args.move, force=args.force,
                              dry_run=args.dry_run, verbose=args.verbose)
            copied += 1
        except FileExistsError as e:
            print(f"[SKIP] {e}", file=sys.stderr)

    print("\n=== Summary ===")
    print(f"Requested IDs : {len(ids)}")
    print(f"Copied/Moved  : {copied}")
    print(f"Missing in XML: {len(missing_in_xml)}")
    print(f"Not found     : {len(not_found_on_disk)}")

if __name__ == "__main__":
    main()

