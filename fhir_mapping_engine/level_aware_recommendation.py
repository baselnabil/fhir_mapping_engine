"""Level-aware Cerner recommendation helpers.

This module keeps the optional level-aware ranking separate from the default
hybrid scorer in ``recommend_cerner_mapping.py``. The strategy uses seed table
order as a priority signal, carries that priority through FK expansion, enriches
candidate text with table context, and gates weak-table candidates unless their
field-level score is strong.
"""
from __future__ import annotations

import json
import re
import string
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PUNCT_TR = str.maketrans({char: " " for char in string.punctuation})
CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")
INDEX_RE = re.compile(r"\[\d+\]")


@dataclass(frozen=True)
class LevelAwareConfig:
    """Tuning constants for conservative table-aware reranking."""

    table_influence: float = 0.35
    table_relevance_floor: float = 0.35
    weak_table_min_field_score: float = 0.75


@dataclass(frozen=True)
class TablePriority:
    table: str
    distance: int
    seed_rank: int
    level: int
    priority: float


@dataclass(frozen=True)
class TableSummary:
    description: str
    column_descriptions: dict[str, str]


def normalize(text: str) -> str:
    return " ".join(text.lower().translate(PUNCT_TR).split())


def split_camel(text: str) -> str:
    return CAMEL_RE.sub(" ", text)


def build_table_priorities(
    seed_tables: list[str],
    adjacency: dict[str, set[str]],
    max_distance: int,
) -> dict[str, TablePriority]:
    """Expand tables and assign bounded priority from seed order and FK distance."""

    unique_seeds: list[str] = []
    seen_seeds: set[str] = set()
    for table in seed_tables:
        upper = table.upper()
        if upper and upper not in seen_seeds:
            unique_seeds.append(upper)
            seen_seeds.add(upper)

    if not unique_seeds:
        return {}

    best: dict[str, tuple[int, int]] = {}
    queue: deque[tuple[str, int, int]] = deque()
    for seed_rank, table in enumerate(unique_seeds):
        best[table] = (0, seed_rank)
        queue.append((table, 0, seed_rank))

    while queue:
        table, distance, seed_rank = queue.popleft()
        if distance >= max_distance:
            continue
        for neighbor in adjacency.get(table, set()):
            next_distance = distance + 1
            current = best.get(neighbor)
            candidate = (next_distance, seed_rank)
            if current is not None and current <= candidate:
                continue
            best[neighbor] = candidate
            queue.append((neighbor, next_distance, seed_rank))

    seed_count = max(len(unique_seeds), 1)
    priorities: dict[str, TablePriority] = {}
    for table, (distance, seed_rank) in best.items():
        distance_component = max(
            0.0,
            1.0 - distance / max(max_distance + 1, 1),
        )
        rank_component = 1.0 - (0.35 * seed_rank / seed_count)
        priority = max(0.0, min(1.0, distance_component * rank_component))
        priorities[table] = TablePriority(
            table=table,
            distance=distance,
            seed_rank=seed_rank,
            level=seed_rank + distance,
            priority=priority,
        )
    return priorities


def load_table_summaries(path: Path | None) -> dict[str, TableSummary]:
    """Load optional table and column descriptions from table_summaries.json."""

    if path is None or not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    items: list[Any]
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = []
        for table_name, value in raw.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("table_name", table_name)
                items.append(item)
    else:
        return {}

    summaries: dict[str, TableSummary] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        table = str(item.get("table_name") or item.get("name") or "").upper()
        if not table:
            continue
        column_descriptions: dict[str, str] = {}
        for column in item.get("columns") or []:
            if not isinstance(column, dict):
                continue
            column_name = str(column.get("name") or column.get("column_name") or "").upper()
            if not column_name:
                continue
            description = str(
                column.get("description") or column.get("definition") or ""
            ).strip()
            if description:
                column_descriptions[column_name] = description
        summaries[table] = TableSummary(
            description=str(item.get("description") or "").strip(),
            column_descriptions=column_descriptions,
        )
    return summaries


def enrich_fhir_query_text(fhir_path: str, base_text: str) -> str:
    """Add path hierarchy context so generic FHIR tails match in the right scope."""

    cleaned_path = INDEX_RE.sub("", fhir_path)
    hierarchy = " ".join(split_camel(part) for part in cleaned_path.split("."))
    tail = cleaned_path.split(".")[-1] if cleaned_path else fhir_path
    tail_phrase = split_camel(tail)
    pieces = [base_text, cleaned_path.replace(".", " "), hierarchy, tail_phrase]
    return " ".join(piece for piece in pieces if piece).strip()


def build_enriched_column_text(
    column: dict[str, Any],
    table_summaries: dict[str, TableSummary],
) -> str:
    """Build Cerner-side text with table, column, type, and summary context."""

    table = str(column.get("table") or "").upper()
    column_name = str(column.get("column") or "").upper()
    summary = table_summaries.get(table)
    summary_column_text = ""
    table_description = ""
    if summary is not None:
        table_description = summary.description
        summary_column_text = summary.column_descriptions.get(column_name, "")

    term_aliases = " ".join(
        sorted((column.get("name_terms") or set()) | (column.get("table_terms") or set()))
    )
    pieces = [
        table,
        table_description,
        column_name,
        str(column.get("definition") or ""),
        summary_column_text,
        str(column.get("data_type") or ""),
        term_aliases,
    ]
    return " ".join(piece for piece in pieces if piece).strip()


def attach_level_context(
    columns: list[dict[str, Any]],
    table_priorities: dict[str, TablePriority],
    table_summaries: dict[str, TableSummary],
) -> list[dict[str, Any]]:
    """Return column copies carrying level-aware table metadata and enriched text."""

    enriched_columns: list[dict[str, Any]] = []
    for column in columns:
        table = str(column.get("table") or "").upper()
        priority = table_priorities.get(table)
        if priority is None:
            continue
        enriched = dict(column)
        enriched["distance"] = priority.distance
        enriched["table_priority"] = priority.priority
        enriched["table_level"] = priority.level
        enriched["table_seed_rank"] = priority.seed_rank
        enriched["definition_text"] = build_enriched_column_text(enriched, table_summaries)
        enriched_columns.append(enriched)
    return enriched_columns


def table_relevance_score(column: dict[str, Any]) -> float:
    return max(0.0, min(1.0, float(column.get("table_priority") or 0.0)))


def field_score(
    name_score: float,
    desc_score: float,
    type_score: float,
    dist_score: float,
    weights: dict[str, float],
) -> float:
    return (
        weights["name"] * name_score
        + weights["desc"] * desc_score
        + weights["type"] * type_score
        + weights["dist"] * dist_score
    )


def keep_candidate(
    base_score: float,
    table_relevance: float,
    min_score: float,
    config: LevelAwareConfig,
) -> bool:
    if base_score < min_score:
        return False
    if table_relevance >= config.table_relevance_floor:
        return True
    return base_score >= config.weak_table_min_field_score


def apply_table_weight(
    base_score: float,
    table_relevance: float,
    config: LevelAwareConfig,
) -> float:
    influence = max(0.0, min(1.0, config.table_influence))
    return base_score * (1.0 - influence + influence * table_relevance)


NameSimilarityFn = Callable[[set[str], dict[str, Any], str], float]
TypeCompatFn = Callable[[str, str, set[str]], float]
DistanceScoreFn = Callable[[int, int], float]


def rank_candidates_for_row(
    fhir_terms: set[str],
    fhir_path: str,
    fhir_type: str,
    desc_row: Any,
    columns: list[dict[str, Any]],
    weights: dict[str, float],
    max_distance: int,
    top_k: int,
    min_score: float,
    name_similarity_fn: NameSimilarityFn,
    type_compat_fn: TypeCompatFn,
    distance_score_fn: DistanceScoreFn,
    config: LevelAwareConfig | None = None,
) -> list[tuple[float, dict[str, float], dict[str, Any]]]:
    """Rank candidates with table-aware damping and weak-table suppression."""

    cfg = config or LevelAwareConfig()
    tail = INDEX_RE.sub("", fhir_path.split(".", 1)[-1] if "." in fhir_path else fhir_path)
    scored: list[tuple[float, dict[str, float], dict[str, Any]]] = []
    for idx, column in enumerate(columns):
        n = name_similarity_fn(fhir_terms, column, tail)
        d = float(desc_row[idx]) if desc_row is not None else 0.0
        t = type_compat_fn(fhir_type, column["data_type"], column["name_terms"])
        z = distance_score_fn(int(column["distance"]), max_distance)
        base = field_score(n, d, t, z, weights)
        table_relevance = table_relevance_score(column)
        if not keep_candidate(base, table_relevance, min_score, cfg):
            continue
        total = apply_table_weight(base, table_relevance, cfg)
        scored.append(
            (
                total,
                {
                    "name": n,
                    "desc": d,
                    "type": t,
                    "dist": z,
                    "field": base,
                    "table": table_relevance,
                },
                column,
            )
        )
    scored.sort(
        key=lambda item: (
            -item[0],
            int(item[2].get("table_level", 0)),
            int(item[2]["distance"]),
            item[2]["table"],
            item[2]["column"],
        )
    )
    if top_k > 0:
        return scored[:top_k]
    return scored
