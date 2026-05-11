#!/usr/bin/env python3
"""filter_top_level_mapping.py

Produce a slimmed-down copy of one (or every) ``mappings/{Resource}_mapping.csv``
that keeps only top-level FHIR elements and their first-level children.

A FHIR element's depth is measured by counting dot-separated segments in the
``Full FHIR Path`` column AFTER stripping the leading resource name. Rows
whose depth exceeds ``--max-depth`` (default 2) are dropped. Examples for
default depth 2:

    Person.id                                 -> kept (depth 1)
    Person.identifier                         -> kept (depth 1)
    Person.identifier.type                    -> kept (depth 2)
    Person.identifier.type.coding             -> dropped (depth 3)
    HealthcareService.contact.address         -> kept (depth 2)
    HealthcareService.contact.address.line[0] -> dropped (depth 3)

Array indices like ``meta.tag[0]`` count as part of the same segment, so
``meta.tag[0]`` is depth 2, while ``meta.tag[0].code`` is depth 3.

Output is written to ``mappings_filtered/{Resource}_mapping.csv`` by default;
the original mapping file is never modified.

Usage::

    python filter_top_level_mapping.py --resource HealthcareService
    python filter_top_level_mapping.py --all
    python filter_top_level_mapping.py --resource Person --max-depth 1
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MAPPINGS_DIR = SCRIPT_DIR / "mappings"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "mappings_filtered"
DEFAULT_MAX_DEPTH = 2

FULL_PATH_COLUMN = "Full FHIR Path"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("filter")


def path_depth(full_fhir_path: str) -> int:
    """Return the number of segments below the resource root.

    ``Person`` itself is depth 0; ``Person.identifier`` is depth 1;
    ``Person.identifier.type`` is depth 2; and so on. Empty paths return -1
    so callers can easily skip them.
    """
    cleaned = full_fhir_path.strip()
    if not cleaned:
        return -1
    parts = cleaned.split(".")
    return max(len(parts) - 1, 0)


def find_full_path_index(header: list[str]) -> int:
    try:
        return header.index(FULL_PATH_COLUMN)
    except ValueError as exc:
        raise ValueError(
            f"Mapping CSV is missing required column {FULL_PATH_COLUMN!r}; "
            f"got header {header!r}"
        ) from exc


def filter_mapping_file(
    src_path: Path,
    dst_path: Path,
    max_depth: int,
) -> tuple[int, int]:
    """Filter ``src_path`` into ``dst_path``. Returns ``(kept, total)``."""
    with src_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            log.warning("Skipping empty file %s", src_path)
            return 0, 0
        full_path_idx = find_full_path_index(header)
        kept_rows: list[list[str]] = []
        total = 0
        for row in reader:
            total += 1
            full_path = row[full_path_idx] if full_path_idx < len(row) else ""
            depth = path_depth(full_path)
            if depth < 0:
                continue
            if depth <= max_depth:
                kept_rows.append(row)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(kept_rows)
    return len(kept_rows), total


def discover_mapping_files(mappings_dir: Path) -> list[Path]:
    """Return every primary ``*_mapping.csv`` (excluding backups/cerner exports)."""
    files: list[Path] = []
    for path in sorted(mappings_dir.glob("*_mapping.csv")):
        name = path.name
        if name.endswith(".bak"):
            continue
        if "_cerner_" in name:
            continue
        files.append(path)
    return files


def resolve_targets(
    mappings_dir: Path,
    resource: str | None,
    do_all: bool,
) -> list[Path]:
    if do_all:
        targets = discover_mapping_files(mappings_dir)
        if not targets:
            raise SystemExit(f"No *_mapping.csv files found in {mappings_dir}")
        return targets
    if not resource:
        raise SystemExit("Either --resource or --all must be supplied")
    candidate = mappings_dir / f"{resource}_mapping.csv"
    if not candidate.exists():
        raise SystemExit(f"Mapping file not found: {candidate}")
    return [candidate]


def run(
    resource: str | None,
    do_all: bool,
    mappings_dir: Path,
    output_dir: Path,
    max_depth: int,
) -> int:
    if max_depth < 0:
        log.error("--max-depth must be >= 0 (got %d)", max_depth)
        return 2
    targets = resolve_targets(mappings_dir, resource, do_all)
    log.info(
        "Filtering %d mapping file(s) into %s (max depth = %d)",
        len(targets),
        output_dir,
        max_depth,
    )
    grand_kept = 0
    grand_total = 0
    for src in targets:
        dst = output_dir / src.name
        kept, total = filter_mapping_file(src, dst, max_depth)
        grand_kept += kept
        grand_total += total
        log.info(
            "%-40s %4d / %4d rows kept -> %s",
            src.name,
            kept,
            total,
            dst.relative_to(SCRIPT_DIR) if dst.is_relative_to(SCRIPT_DIR) else dst,
        )
    log.info(
        "Done. %d / %d rows retained across %d file(s).",
        grand_kept,
        grand_total,
        len(targets),
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter mappings/{Resource}_mapping.csv to keep only top-level "
            "FHIR elements (and their first-level children by default)."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resource",
        help="FHIR resource name to filter, e.g. HealthcareService.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Filter every *_mapping.csv in --mappings-dir.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help=(
            "Maximum FHIR-path depth to retain (segments after the resource "
            "name). Default 2 keeps Resource.x and Resource.x.y."
        ),
    )
    parser.add_argument(
        "--mappings-dir",
        type=Path,
        default=DEFAULT_MAPPINGS_DIR,
        help="Directory containing the source *_mapping.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write filtered *_mapping.csv files into.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(
        resource=args.resource,
        do_all=args.all,
        mappings_dir=args.mappings_dir,
        output_dir=args.output_dir,
        max_depth=args.max_depth,
    )


if __name__ == "__main__":
    sys.exit(main())
