#!/usr/bin/env python3
"""Extract Cerner code columns and their code set numbers from mapping results.

The recommender writes ``results/{Resource}_mapping.csv`` files with a
``Cerner Source Fields`` column containing pipe-separated ``TABLE.COLUMN``
candidates. This script scans those result mappings, keeps code fields such as
``PRSNL_SERVICE_RESOURCE_RELTN.SERVICE_RESOURCE_CD``, looks them up in the
parsed Cerner data dictionary, and writes a two-column CSV: ``Complete Path``
(``TABLE.COLUMN``) and ``Value Set Number`` (Cerner ``code_set`` from
``columns.json``). The same field merged across resources appears once.

Usage::

    python extract_code_value_sets.py --input results/Device_mapping.csv
    python extract_code_value_sets.py --all
    python extract_code_value_sets.py --all --output results/code_value_sets.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_COLUMNS_JSON = SCRIPT_DIR / "CERNER" / "2018.01 Main" / "parsed_json" / "columns.json"

SOURCE_FIELD_COLUMNS = ("Cerner Source Fields", "Source Fields")

FIELD_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\.([A-Za-z][A-Za-z0-9_]*)\b")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("code-value-sets")


@dataclass
class ColumnInfo:
    code_set: str = ""
    data_type: str = ""
    nullable: str = ""
    definition: str = ""


@dataclass
class CodeUsage:
    table_name: str
    column_name: str
    code_set: str

    @property
    def source_field(self) -> str:
        return f"{self.table_name}.{self.column_name}"


def normalize_identifier(value: str) -> str:
    return value.strip().upper()


def load_column_catalog(columns_json: Path) -> dict[tuple[str, str], ColumnInfo]:
    with columns_json.open("r", encoding="utf-8") as fh:
        rows: list[dict[str, Any]] = json.load(fh)

    catalog: dict[tuple[str, str], ColumnInfo] = {}
    for row in rows:
        table_name = normalize_identifier(str(row.get("table_name", "")))
        column_name = normalize_identifier(str(row.get("column_name", "")))
        if not table_name or not column_name:
            continue
        catalog[(table_name, column_name)] = ColumnInfo(
            code_set=str(row.get("code_set", "") or "").strip(),
            data_type=str(row.get("data_type", "") or "").strip(),
            nullable=str(row.get("nullable", "") or "").strip(),
            definition=str(row.get("definition", "") or "").strip(),
        )
    return catalog


def find_column_index(header: list[str], names: tuple[str, ...] | str) -> int:
    candidates = (names,) if isinstance(names, str) else names
    for name in candidates:
        if name in header:
            return header.index(name)
    raise ValueError(f"CSV is missing required column {candidates!r}; got {header!r}")


def extract_source_fields(cell: str) -> list[tuple[str, str]]:
    """Return normalized ``(table, column)`` pairs from a source-field cell."""
    fields: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in FIELD_RE.finditer(cell):
        field = (normalize_identifier(match.group(1)), normalize_identifier(match.group(2)))
        if field not in seen:
            seen.add(field)
            fields.append(field)
    return fields


def discover_result_mappings(results_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in results_dir.glob("*_mapping.csv")
        if path.is_file() and "_cerner_" not in path.name
    )


def is_code_field(column_name: str, info: ColumnInfo | None) -> bool:
    return column_name.endswith("_CD") or bool(info and info.code_set)


def add_mapping_file_usages(
    mapping_csv: Path,
    catalog: dict[tuple[str, str], ColumnInfo],
    usages: dict[tuple[str, str], CodeUsage],
    only_with_code_set: bool,
) -> None:
    with mapping_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            log.warning("Skipping empty mapping file %s", mapping_csv)
            return

        source_idx = find_column_index(header, SOURCE_FIELD_COLUMNS)

        for row in reader:
            source_cell = row[source_idx] if source_idx < len(row) else ""

            for table_name, column_name in extract_source_fields(source_cell):
                info = catalog.get((table_name, column_name))
                if not is_code_field(column_name, info):
                    continue
                if only_with_code_set and not (info and info.code_set):
                    continue

                info = info or ColumnInfo()
                key = (table_name, column_name)
                if key not in usages:
                    usages[key] = CodeUsage(
                        table_name=table_name,
                        column_name=column_name,
                        code_set=info.code_set,
                    )


def write_usages(output_csv: Path, usages: list[CodeUsage]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(usages, key=lambda u: (u.source_field, u.code_set))
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["Complete Path", "Value Set Number"],
        )
        writer.writeheader()
        for usage in rows:
            writer.writerow(
                {
                    "Complete Path": usage.source_field,
                    "Value Set Number": usage.code_set,
                }
            )


def default_output_path(input_csv: Path | None, all_results: bool, results_dir: Path) -> Path:
    if all_results or input_csv is None:
        return results_dir / "code_value_sets.csv"
    return input_csv.with_name(f"{input_csv.stem}_code_value_sets.csv")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract code-valued Cerner source fields from result mappings.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--input", type=Path, help="Single results/{Resource}_mapping.csv file to scan.")
    target.add_argument("--all", action="store_true", help="Scan every *_mapping.csv file in --results-dir.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--columns-json", type=Path, default=DEFAULT_COLUMNS_JSON)
    parser.add_argument("--output", type=Path, help="Output CSV path.")
    parser.add_argument(
        "--only-with-code-set",
        action="store_true",
        help="Drop *_CD fields that are not found in columns.json or have an empty code_set.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    columns_json = args.columns_json.resolve()
    results_dir = args.results_dir.resolve()

    if not columns_json.exists():
        log.error("Cerner columns JSON not found: %s", columns_json)
        return 1

    if args.all:
        input_files = discover_result_mappings(results_dir)
        if not input_files:
            log.error("No *_mapping.csv files found in %s", results_dir)
            return 1
    else:
        input_files = [args.input.resolve()]
        if not input_files[0].exists():
            log.error("Input mapping CSV not found: %s", input_files[0])
            return 1

    catalog = load_column_catalog(columns_json)
    usages: dict[tuple[str, str], CodeUsage] = {}
    for mapping_csv in input_files:
        add_mapping_file_usages(
            mapping_csv=mapping_csv,
            catalog=catalog,
            usages=usages,
            only_with_code_set=args.only_with_code_set,
        )

    sorted_usages = sorted(
        usages.values(),
        key=lambda item: (item.table_name, item.column_name),
    )
    output_csv = (args.output or default_output_path(args.input, args.all, results_dir)).resolve()
    write_usages(output_csv, sorted_usages)
    log.info("Wrote %s code fields to %s", len(sorted_usages), output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
