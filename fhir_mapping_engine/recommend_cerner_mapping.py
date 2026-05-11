#!/usr/bin/env python3
"""recommend_cerner_mapping.py

Recommend Cerner source columns for each FHIR element of a resource and fill
the `Source Fields` column of an existing `mappings/{Resource}_mapping.csv`,
writing the completed CSV to `results/{Resource}_mapping.csv`.

Candidate pool = supplied Cerner tables + tables reachable from them through
the parsed FK relationship graph, up to ``--max-distance`` hops (default 2;
pass ``0`` for strict supplied-tables-only).

When ``--tables`` is omitted the script auto-suggests seed tables. By default
it ranks every Cerner table with the hybrid column-catalog scorer (same signals
as column recommendation). Pass ``--suggest-tables`` alone (no integer) to
use description-driven ranking from ``table_summaries.json`` (same logic as
``suggest_cerner_tables.py``). Pass ``--suggest-tables N`` for the legacy
catalog scorer only, taking the top ``N`` seeds. ``--num-seed-tables`` overrides
the seed count when you use bare ``--suggest-tables`` or omit ``--suggest-tables``.
``--suggest-only`` prints suggestions and exits without modifying any CSV.

Column-level scoring is hybrid; each sub-score is normalized to 0..1:

  * name_score   token overlap (with abbreviation aliases) + RapidFuzz
  * desc_score   TF-IDF cosine between FHIR element definition and Cerner
                 column definition (optionally sentence-transformers)
  * type_bonus   compatibility between FHIR element type and Cerner data type
  * dist_score   1 - distance / (max_distance + 1) so closer tables rank up

Table-level scoring aggregates the per-column scores:

  * name_score    table-name vs. resource-name similarity
  * content_score average over FHIR elements of the best column-vs-element
                  score within the table
  * coverage      fraction of FHIR elements with at least one column in the
                  table scoring above ``--coverage-threshold``

Usage::

    # Explicit seed tables (existing behavior)
    python recommend_cerner_mapping.py \
        --resource Person --tables PERSON_PATIENT PERSON

    # Auto-suggest 5 tables (catalog scorer) and run the mapping with them as seeds
    python recommend_cerner_mapping.py --resource Person

    # Same seed count using description-driven suggestion (table_summaries.json)
    python recommend_cerner_mapping.py --resource Person --suggest-tables

    # Just print suggestions, don't touch the CSV
    python recommend_cerner_mapping.py --resource Person --suggest-only
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import string
import sys
import urllib.error
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import level_aware_recommendation as level_aware
import suggest_cerner_tables as cerner_table_suggest

try:
    from rapidfuzz import fuzz  # type: ignore
except ImportError:  # pragma: no cover - import guard
    print(
        "ERROR: rapidfuzz is required. Install with: pip install rapidfuzz",
        file=sys.stderr,
    )
    raise

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except ImportError:  # pragma: no cover - import guard
    print(
        "ERROR: scikit-learn is required. Install with: pip install scikit-learn",
        file=sys.stderr,
    )
    raise


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COLUMNS_JSON = Path(
    "/home/basel/Hevelian/Specifications/CERNER/2018.01 Main/parsed_json/columns.json"
)
DEFAULT_RELATIONSHIPS_JSON = Path(
    "/home/basel/Hevelian/Specifications/CERNER/2018.01 Main/parsed_json/relationships.json"
)
DEFAULT_TABLE_SUMMARIES_JSON = Path(
    "/home/basel/Hevelian/Specifications/CERNER/2018.01 Main/parsed_json/table_summaries.json"
)
DEFAULT_MAPPINGS_DIR = SCRIPT_DIR / "mappings"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "mappings" / "_cache"

DEFAULT_MAX_DISTANCE = 2
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.05
DEFAULT_WEIGHTS: dict[str, float] = {
    "name": 0.45,
    "desc": 0.45,
    "type": 0.5,
    "dist": 0.5,
}

DEFAULT_SUGGEST_TABLES = 5
# argparse const when ``--suggest-tables`` is passed without ``N`` (description-based engine).
_SUGGEST_TABLES_SUMMARIES_ENGINE = "__summaries_engine__"
DEFAULT_SUGGEST_PREFILTER = 80
DEFAULT_COVERAGE_THRESHOLD = 0.30
DEFAULT_TABLE_WEIGHTS: dict[str, float] = {
    "name": 0.40,
    "content": 0.40,
    "coverage": 0.20,
}

# Cerner-style abbreviation -> expanded token aliases.
TOKEN_ALIASES: dict[str, set[str]] = {
    "cd": {"code"},
    "dt": {"date"},
    "tm": {"time"},
    "id": {"identifier"},
    "ind": {"indicator", "boolean"},
    "flag": {"indicator", "boolean"},
    "nbr": {"number"},
    "num": {"number"},
    "qty": {"quantity"},
    "amt": {"amount"},
    "txt": {"text", "string"},
    "addr": {"address"},
    "ph": {"phone"},
    "loc": {"location"},
    "org": {"organization"},
    "ref": {"reference"},
    "vc2": {"string"},
    "char": {"string"},
    "freq": {"frequency"},
    "stat": {"status"},
    "prsnl": {"personnel", "practitioner"},
    "encntr": {"encounter"},
    "comm": {"communication"},
}

# FHIR-element synonym expansion so generic FHIR terms map to Cerner vocab.
FHIR_SYNONYMS: dict[str, set[str]] = {
    "active": {"status", "indicator", "ind", "effective"},
    "address": {"addr", "street", "city", "state", "postal", "zip", "country", "line"},
    "birth": {"born", "dob"},
    "birthdate": {"birth", "born", "dob", "date"},
    "deceased": {"death", "died", "dead", "expired"},
    "family": {"surname", "last"},
    "given": {"first", "middle"},
    "gender": {"sex"},
    "identifier": {"id", "alias", "mrn", "ident"},
    "marital": {"marriage", "spouse"},
    "name": {"first", "last", "full", "formatted", "prefix", "suffix"},
    "telecom": {"phone", "email", "contact", "fax", "mobile"},
    "language": {"lang", "communication"},
    "photo": {"image", "picture"},
    "managingorganization": {"organization", "facility"},
    "contact": {"telecom", "phone", "email"},
    "communication": {"language", "lang"},
    "link": {"reference", "ref"},
}

PUNCT_TR = str.maketrans({c: " " for c in string.punctuation})
CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")
INDEX_RE = re.compile(r"\[\d+\]")

# Accept either header on input; always emit the canonical name on output.
SOURCE_COLUMN_ALIASES = ("Cerner Source Fields", "Source Fields")
SOURCE_COLUMN_CANONICAL = "Cerner Source Fields"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("recommend")


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    return " ".join(text.lower().translate(PUNCT_TR).split())


def split_camel(text: str) -> str:
    return CAMEL_RE.sub(" ", text)


def tokenize(text: str) -> set[str]:
    base = set(normalize(split_camel(text)).split())
    expanded = set(base)
    for tok in base:
        expanded.update(TOKEN_ALIASES.get(tok, set()))
    return expanded


def fhir_path_terms(fhir_path: str) -> set[str]:
    tail = fhir_path.split(".", 1)[-1] if "." in fhir_path else fhir_path
    tail_clean = INDEX_RE.sub("", tail)
    base = tokenize(tail_clean)
    expanded = set(base)
    for tok in base:
        expanded.update(FHIR_SYNONYMS.get(tok, set()))
    compact = normalize(tail_clean).replace(" ", "")
    expanded.update(FHIR_SYNONYMS.get(compact, set()))
    return expanded


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def relationship_adjacency(relationships: Any) -> dict[str, set[str]]:
    """Build an undirected FK adjacency map: TABLE -> set(TABLE)."""
    adj: dict[str, set[str]] = defaultdict(set)
    if isinstance(relationships, dict):
        for table, related in relationships.items():
            src = str(table).upper()
            for tgt_table in related or []:
                tgt = str(tgt_table).upper()
                if tgt and tgt != src:
                    adj[src].add(tgt)
                    adj[tgt].add(src)
        return adj
    if isinstance(relationships, list):
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            parent = str(rel.get("parent_table") or "").upper()
            child = str(rel.get("child_table") or "").upper()
            if parent and child and parent != child:
                adj[parent].add(child)
                adj[child].add(parent)
    return adj


def expand_tables_bfs(
    seed_tables: list[str],
    adjacency: dict[str, set[str]],
    max_distance: int,
) -> dict[str, int]:
    """BFS up to ``max_distance`` hops. Returns table -> distance from seed.

    ``max_distance == 0`` keeps only the seed tables (no FK expansion).
    """
    distances: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()
    for table in seed_tables:
        upper = table.upper()
        if upper not in distances:
            distances[upper] = 0
            queue.append((upper, 0))
    while queue:
        table, dist = queue.popleft()
        if dist >= max_distance:
            continue
        for neighbor in adjacency.get(table, set()):
            if neighbor not in distances:
                distances[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
    return distances


def load_columns_in_tables(
    columns_path: Path,
    table_distances: dict[str, int],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Return (columns, tables_actually_seen_in_catalog)."""
    raw = load_json(columns_path)
    if not isinstance(raw, list):
        raise ValueError(f"Unexpected columns.json shape: {type(raw).__name__}")

    columns: list[dict[str, Any]] = []
    seen_tables: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        table = str(entry.get("table_name") or "").upper()
        column = str(entry.get("column_name") or "").upper()
        if not table or not column or table not in table_distances:
            continue
        seen_tables.add(table)
        definition = str(entry.get("definition") or "")
        data_type = str(entry.get("data_type") or "")
        name_terms = tokenize(column)
        table_terms = tokenize(table)
        columns.append(
            {
                "table": table,
                "column": column,
                "data_type": data_type,
                "definition": definition,
                "distance": table_distances[table],
                "name_terms": name_terms,
                "table_terms": table_terms,
                "definition_text": f"{table} {column} {definition}".strip(),
            }
        )
    return columns, seen_tables


# ---------------------------------------------------------------------------
# FHIR element definitions (StructureDefinition snapshot)
# ---------------------------------------------------------------------------


def fetch_fhir_definitions(resource: str, cache_dir: Path) -> dict[str, dict[str, str]]:
    """Fetch FHIR R4 StructureDefinition snapshot elements; cache locally.

    Returns a dict ``path -> {short, definition}``. Empty dict means we could
    not retrieve definitions (network error); description similarity then
    degrades to FHIR-path-token-only matching.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"fhir_element_defs_{resource}.json"
    if cache_file.exists():
        try:
            cached = load_json(cache_file)
            if isinstance(cached, dict):
                log.info(
                    "Loaded %d cached FHIR element definitions from %s",
                    len(cached),
                    cache_file,
                )
                return cached
        except Exception as exc:  # noqa: BLE001
            log.warning("Cache file %s is unreadable (%s); refetching", cache_file, exc)

    url = f"https://hl7.org/fhir/R4/{resource.lower()}.profile.json"
    log.info("Fetching FHIR definitions from %s", url)
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "recommend-cerner-mapping/1.0"},
        )
        with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        log.warning(
            "Could not fetch FHIR definitions (%s). Falling back to path-only matching.",
            exc,
        )
        return {}

    elements = (payload.get("snapshot") or {}).get("element") or []
    defs: dict[str, dict[str, str]] = {}
    for element in elements:
        path = element.get("path") or ""
        if not path:
            continue
        defs[path] = {
            "short": element.get("short") or "",
            "definition": element.get("definition") or "",
        }
    with cache_file.open("w", encoding="utf-8") as fh:
        json.dump(defs, fh, indent=2)
    log.info("Cached %d FHIR element definitions to %s", len(defs), cache_file)
    return defs


def lookup_fhir_definition(
    full_path: str,
    defs: dict[str, dict[str, str]],
) -> tuple[str, str]:
    cleaned = INDEX_RE.sub("", full_path)
    entry = defs.get(cleaned) or defs.get(full_path)
    if not entry:
        # Try matching against base path (e.g. Person.identifier matches identifier rows)
        base = cleaned.split("[", 1)[0]
        entry = defs.get(base)
    if not entry:
        return "", ""
    return entry.get("short") or "", entry.get("definition") or ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


_TYPE_RAW_MAX = 5.0


def type_compat(fhir_type: str, data_type: str, name_terms: set[str]) -> float:
    ftype = normalize(fhir_type)
    dtype = normalize(data_type)
    raw = 0.0
    if not ftype or not dtype:
        return 0.0
    if "date" in ftype and "date" in dtype:
        raw = 5.0
    elif ftype in {"datetime", "instant"} and "date" in dtype:
        raw = 5.0
    elif "boolean" in ftype and ({"ind", "indicator", "flag"} & name_terms):
        raw = 3.0
    elif "code" in ftype and ({"cd", "code"} & name_terms):
        raw = 3.0
    elif ftype in {"integer", "positiveint", "integer64", "unsignedint"} and "number" in dtype:
        raw = 2.0
    elif ftype == "decimal" and ("number" in dtype or "float" in dtype):
        raw = 2.0
    elif ftype in {"string", "uri", "url", "id", "markdown"} and any(
        token in dtype for token in ("vc2", "char", "long")
    ):
        raw = 2.0
    return min(raw / _TYPE_RAW_MAX, 1.0)


def name_similarity(
    fhir_terms: set[str],
    column: dict[str, Any],
    fhir_path_tail: str,
) -> float:
    col_terms = column["name_terms"] | column["table_terms"]
    if fhir_terms and col_terms:
        intersect = len(fhir_terms & col_terms)
        union = len(fhir_terms | col_terms)
        jaccard = intersect / union if union else 0.0
    else:
        jaccard = 0.0
    table_col = f"{column['table']}.{column['column']}"
    fuzzy = fuzz.token_set_ratio(fhir_path_tail.lower(), table_col.lower()) / 100.0
    return 0.5 * jaccard + 0.5 * fuzzy


def distance_score(distance: int, max_distance: int) -> float:
    denom = max(max_distance + 1, 1)
    return max(0.0, 1.0 - distance / denom)


# ---------------------------------------------------------------------------
# Table-level suggestion (uses the same column-level scoring under the hood)
# ---------------------------------------------------------------------------


def collect_all_tables(columns_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Group every column in columns.json by uppercased table name."""
    raw = load_json(columns_path)
    if not isinstance(raw, list):
        raise ValueError(f"Unexpected columns.json shape: {type(raw).__name__}")
    by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        table = str(entry.get("table_name") or "").upper()
        column = str(entry.get("column_name") or "").upper()
        if not table or not column:
            continue
        definition = str(entry.get("definition") or "")
        data_type = str(entry.get("data_type") or "")
        by_table[table].append(
            {
                "table": table,
                "column": column,
                "data_type": data_type,
                "definition": definition,
                "name_terms": tokenize(column),
                "table_terms": tokenize(table),
                "definition_text": f"{table} {column} {definition}".strip(),
            }
        )
    return by_table


def resource_query_terms(resource: str) -> set[str]:
    """Tokens describing the FHIR resource itself (used for name pre-filtering)."""
    base = tokenize(resource)
    expanded = set(base)
    for tok in base:
        expanded.update(FHIR_SYNONYMS.get(tok, set()))
    return expanded


def score_table_name(
    resource: str,
    resource_terms: set[str],
    table_name: str,
) -> float:
    """Blend Jaccard token overlap with RapidFuzz fuzzy score."""
    table_terms = tokenize(table_name)
    if resource_terms and table_terms:
        intersect = len(resource_terms & table_terms)
        union = len(resource_terms | table_terms)
        jaccard = intersect / union if union else 0.0
    else:
        jaccard = 0.0
    fuzzy = fuzz.token_set_ratio(resource.lower(), table_name.lower()) / 100.0
    return 0.5 * jaccard + 0.5 * fuzzy


def suggest_seed_tables(
    resource: str,
    fhir_paths: list[str],
    fhir_types: list[str],
    fhir_query_texts: list[str],
    by_table: dict[str, list[dict[str, Any]]],
    n_suggestions: int,
    prefilter_k: int,
    coverage_threshold: float,
    column_weights: dict[str, float],
    table_weights: dict[str, float],
    use_embeddings: bool,
) -> list[tuple[str, float, dict[str, float]]]:
    """Rank Cerner tables for the resource using column-level scores.

    Returns a list of (table, combined_score, breakdown) sorted descending,
    truncated to ``n_suggestions``.
    """
    if not by_table:
        return []
    res_terms = resource_query_terms(resource)
    name_scores = {tbl: score_table_name(resource, res_terms, tbl) for tbl in by_table}

    prefilter_k = max(prefilter_k, n_suggestions)
    candidates = sorted(name_scores.items(), key=lambda kv: -kv[1])[:prefilter_k]
    candidate_tables = [tbl for tbl, _ in candidates]
    log.info(
        "Pre-filtered %d candidate tables (out of %d) for content scoring",
        len(candidate_tables),
        len(by_table),
    )

    columns: list[dict[str, Any]] = []
    for tbl in candidate_tables:
        columns.extend(by_table[tbl])
    if not columns or not fhir_paths:
        return [
            (tbl, name_scores[tbl], {"name": name_scores[tbl], "content": 0.0, "coverage": 0.0})
            for tbl in candidate_tables[:n_suggestions]
        ]

    column_texts = [c["definition_text"] for c in columns]
    sim_matrix = build_description_similarity_matrix(
        fhir_query_texts, column_texts, use_embeddings
    )
    fhir_terms_per_row = [fhir_path_terms(p) for p in fhir_paths]
    fhir_tails = [
        INDEX_RE.sub("", p.split(".", 1)[-1] if "." in p else p) for p in fhir_paths
    ]

    table_to_col_indices: dict[str, list[int]] = defaultdict(list)
    for idx, col in enumerate(columns):
        table_to_col_indices[col["table"]].append(idx)

    # Renormalize the column weights without the distance term (no FK BFS yet).
    w_sum = column_weights["name"] + column_weights["desc"] + column_weights["type"]
    if w_sum <= 0:
        w_name = w_desc = w_type = 1.0 / 3
    else:
        w_name = column_weights["name"] / w_sum
        w_desc = column_weights["desc"] / w_sum
        w_type = column_weights["type"] / w_sum

    t_sum = table_weights["name"] + table_weights["content"] + table_weights["coverage"]
    if t_sum <= 0:
        tw_name = tw_content = tw_coverage = 1.0 / 3
    else:
        tw_name = table_weights["name"] / t_sum
        tw_content = table_weights["content"] / t_sum
        tw_coverage = table_weights["coverage"] / t_sum

    results: list[tuple[str, float, dict[str, float]]] = []
    for tbl in candidate_tables:
        col_indices = table_to_col_indices.get(tbl, [])
        if not col_indices:
            continue
        per_element_best: list[float] = []
        for elem_idx in range(len(fhir_paths)):
            tail = fhir_tails[elem_idx]
            best = 0.0
            for col_idx in col_indices:
                column = columns[col_idx]
                n = name_similarity(fhir_terms_per_row[elem_idx], column, tail)
                d = float(sim_matrix[elem_idx, col_idx])
                t = type_compat(
                    fhir_types[elem_idx], column["data_type"], column["name_terms"]
                )
                score = w_name * n + w_desc * d + w_type * t
                if score > best:
                    best = score
            per_element_best.append(best)

        content_avg = sum(per_element_best) / max(len(per_element_best), 1)
        covered = sum(1 for s in per_element_best if s >= coverage_threshold)
        coverage = covered / max(len(per_element_best), 1)
        combined = (
            tw_name * name_scores[tbl]
            + tw_content * content_avg
            + tw_coverage * coverage
        )
        results.append(
            (
                tbl,
                combined,
                {
                    "name": name_scores[tbl],
                    "content": content_avg,
                    "coverage": coverage,
                    "n_columns": float(len(col_indices)),
                },
            )
        )

    results.sort(key=lambda item: (-item[1], item[0]))
    return results[:n_suggestions]


def format_table_suggestions(
    suggestions: list[tuple[str, float, dict[str, float]]],
) -> str:
    if not suggestions:
        return "(no candidates)"
    lines = []
    for rank, (tbl, score, br) in enumerate(suggestions, start=1):
        lines.append(
            "  {rank:>2}. {tbl:<40s} score={score:.3f}  "
            "name={name:.2f} content={content:.2f} coverage={coverage:.2f} "
            "cols={cols:d}".format(
                rank=rank,
                tbl=tbl,
                score=score,
                name=br.get("name", 0.0),
                content=br.get("content", 0.0),
                coverage=br.get("coverage", 0.0),
                cols=int(br.get("n_columns", 0)),
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def load_mapping_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows


def write_mapping_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def find_column_index(header: list[str], name: str) -> int:
    try:
        return header.index(name)
    except ValueError as exc:
        raise ValueError(
            f"Mapping CSV is missing required column {name!r}; got header {header!r}"
        ) from exc


def find_source_column_index(header: list[str]) -> int:
    """Locate the Cerner source column (accepts the canonical or legacy name)."""
    for alias in SOURCE_COLUMN_ALIASES:
        if alias in header:
            return header.index(alias)
    raise ValueError(
        f"Mapping CSV must contain one of {SOURCE_COLUMN_ALIASES!r}; got header {header!r}"
    )


def pad_row(row: list[str], width: int) -> list[str]:
    if len(row) >= width:
        return row
    return row + [""] * (width - len(row))


def cell(row: list[str], index: int) -> str:
    return row[index] if 0 <= index < len(row) else ""


def canonicalize_header(header: list[str]) -> list[str]:
    """Rename a legacy 'Source Fields' header cell to the canonical name."""
    new_header = list(header)
    for idx, name in enumerate(new_header):
        if name == SOURCE_COLUMN_CANONICAL:
            return new_header
    for idx, name in enumerate(new_header):
        if name == "Source Fields":
            new_header[idx] = SOURCE_COLUMN_CANONICAL
            return new_header
    return new_header


_LEGACY_RECOMMEND_RE = re.compile(r"\[recommend\].*?(?=(?:\s*\|\s*)|$)", re.DOTALL)


def strip_legacy_recommend_segment(notes: str) -> str:
    """Remove any prior ``[recommend] ...`` segment left in the Notes column."""
    if not notes:
        return notes
    cleaned = _LEGACY_RECOMMEND_RE.sub("", notes).strip(" |").strip()
    return cleaned


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def parse_weights(weights_str: str) -> dict[str, float]:
    return _parse_weight_string(weights_str, DEFAULT_WEIGHTS)


def parse_table_weights(weights_str: str) -> dict[str, float]:
    return _parse_weight_string(weights_str, DEFAULT_TABLE_WEIGHTS)


def _parse_weight_string(weights_str: str, defaults: dict[str, float]) -> dict[str, float]:
    weights = dict(defaults)
    if not weights_str:
        return weights
    for piece in weights_str.split(","):
        piece = piece.strip()
        if not piece or "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        key = key.strip().lower()
        if key not in weights:
            log.warning("Unknown weight key %r (allowed: %s)", key, ",".join(weights))
            continue
        try:
            weights[key] = float(value.strip())
        except ValueError:
            log.warning("Ignoring weight %r (not a number)", piece)
    total = sum(weights.values())
    if total <= 0:
        log.warning("Weight sum is zero; falling back to defaults")
        return dict(defaults)
    return {k: v / total for k, v in weights.items()}


def build_fhir_query_text(fhir_path: str, defs: dict[str, dict[str, str]]) -> str:
    short, definition = lookup_fhir_definition(fhir_path, defs)
    tail = INDEX_RE.sub("", fhir_path.split(".", 1)[-1] if "." in fhir_path else fhir_path)
    tail_phrase = " ".join(split_camel(tail).replace(".", " ").split())
    pieces = [tail_phrase, short, definition]
    return " ".join(p for p in pieces if p).strip()


def build_description_similarity_matrix(
    fhir_texts: list[str],
    column_texts: list[str],
    use_embeddings: bool,
) -> Any:
    """Return an (n_fhir, n_columns) cosine-similarity matrix in [0, 1]."""
    if not fhir_texts or not column_texts:
        import numpy as np

        return np.zeros((len(fhir_texts), len(column_texts)))
    if use_embeddings:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise SystemExit(
                "sentence-transformers not installed; pip install sentence-transformers"
            ) from exc
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        f_emb = model.encode(fhir_texts, show_progress_bar=False, normalize_embeddings=True)
        c_emb = model.encode(column_texts, show_progress_bar=False, normalize_embeddings=True)
        sim = f_emb @ c_emb.T
        # cosine of normalized vectors lives in [-1, 1]; clamp to [0, 1]
        import numpy as np

        return np.clip(sim, 0.0, 1.0)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]+\b",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    corpus = column_texts + fhir_texts
    vectorizer.fit(corpus)
    fhir_vec = vectorizer.transform(fhir_texts)
    col_vec = vectorizer.transform(column_texts)
    sim = cosine_similarity(fhir_vec, col_vec)
    import numpy as np

    return np.clip(sim, 0.0, 1.0)


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
) -> list[tuple[float, dict[str, float], dict[str, Any]]]:
    tail = INDEX_RE.sub("", fhir_path.split(".", 1)[-1] if "." in fhir_path else fhir_path)
    scored: list[tuple[float, dict[str, float], dict[str, Any]]] = []
    for idx, column in enumerate(columns):
        n = name_similarity(fhir_terms, column, tail)
        d = float(desc_row[idx]) if desc_row is not None else 0.0
        t = type_compat(fhir_type, column["data_type"], column["name_terms"])
        z = distance_score(int(column["distance"]), max_distance)
        total = (
            weights["name"] * n
            + weights["desc"] * d
            + weights["type"] * t
            + weights["dist"] * z
        )
        if total < min_score:
            continue
        scored.append((total, {"name": n, "desc": d, "type": t, "dist": z}, column))
    scored.sort(
        key=lambda item: (
            -item[0],
            item[2]["distance"],
            item[2]["table"],
            item[2]["column"],
        )
    )
    if top_k > 0:
        return scored[:top_k]
    return scored


def format_source_fields(matches: list[tuple[float, dict[str, float], dict[str, Any]]]) -> str:
    """Render the top candidates into a single ``TABLE.COL | TABLE.COL | ...`` cell.

    The pipe separator is used (rather than ``;``) so that spreadsheet tools
    which auto-detect ``;`` as a CSV delimiter (Excel/LibreOffice in many
    European locales) keep all candidates in a single cell.
    """
    if not matches:
        return ""
    return " | ".join(f"{m[2]['table']}.{m[2]['column']}" for m in matches)


def _load_fhir_rows(
    resource: str,
    mapping_csv: Path,
    cache_dir: Path,
) -> tuple[
    list[str],
    list[list[str]],
    int,
    int,
    int,
    int,
    int,
    list[int],
    list[str],
    list[str],
    list[str],
]:
    """Read the mapping CSV and parse FHIR rows.

    Returns (header, rows, width, full_path_idx, type_idx, source_idx, notes_idx,
    fhir_indices, fhir_paths, fhir_types, fhir_query_texts).
    """
    log.info("Loading FHIR scaffold %s", mapping_csv)
    header, rows = load_mapping_csv(mapping_csv)
    full_path_idx = find_column_index(header, "Full FHIR Path")
    type_idx = find_column_index(header, "Element Type")
    source_idx = find_source_column_index(header)
    if header[source_idx] != SOURCE_COLUMN_CANONICAL:
        log.info(
            "Renaming column %r to %r",
            header[source_idx],
            SOURCE_COLUMN_CANONICAL,
        )
    header = canonicalize_header(header)
    width = len(header)
    notes_idx = header.index("Notes") if "Notes" in header else -1

    fhir_defs = fetch_fhir_definitions(resource, cache_dir)
    fhir_indices: list[int] = []
    fhir_paths: list[str] = []
    fhir_types: list[str] = []
    fhir_query_texts: list[str] = []
    for row_idx, raw_row in enumerate(rows):
        row = pad_row(raw_row, width)
        rows[row_idx] = row
        fhir_path = cell(row, full_path_idx).strip()
        if not fhir_path:
            continue
        fhir_indices.append(row_idx)
        fhir_paths.append(fhir_path)
        fhir_types.append(cell(row, type_idx).strip())
        fhir_query_texts.append(build_fhir_query_text(fhir_path, fhir_defs))

    return (
        header,
        rows,
        width,
        full_path_idx,
        type_idx,
        source_idx,
        notes_idx,
        fhir_indices,
        fhir_paths,
        fhir_types,
        fhir_query_texts,
    )


def run(
    resource: str,
    seed_tables: list[str] | None,
    columns_path: Path,
    relationships_path: Path,
    table_summaries_path: Path | None,
    mappings_dir: Path,
    results_dir: Path,
    cache_dir: Path,
    max_distance: int,
    top_k: int,
    min_score: float,
    weights: dict[str, float],
    recommendation_mode: str,
    use_embeddings: bool,
    n_suggest: int,
    suggest_prefilter: int,
    coverage_threshold: float,
    table_weights: dict[str, float],
    suggest_only: bool,
    table_suggestion_engine: str,
) -> int:
    mapping_csv = mappings_dir / f"{resource}_mapping.csv"
    if not mapping_csv.exists():
        log.error("Mapping CSV not found: %s", mapping_csv)
        return 2

    (
        header,
        rows,
        width,
        full_path_idx,
        type_idx,
        source_idx,
        notes_idx,
        fhir_indices,
        fhir_paths,
        fhir_types,
        fhir_query_texts,
    ) = _load_fhir_rows(resource, mapping_csv, cache_dir)

    if not fhir_indices:
        log.error("No FHIR rows with a Full FHIR Path found in %s", mapping_csv)
        return 4

    auto_suggest = suggest_only or not seed_tables
    if auto_suggest:
        if table_suggestion_engine == "summaries":
            if not table_summaries_path or not table_summaries_path.exists():
                log.error(
                    "Description-based table suggestion requires an existing "
                    "--table-summaries-json (same file as suggest_cerner_tables.py). "
                    "Got: %s",
                    table_summaries_path,
                )
                return 2
            log.info(
                "Suggesting seed tables via table_summaries.json (%s)",
                table_summaries_path,
            )
            try:
                fhir_text = cerner_table_suggest.load_fhir_profile_text(
                    resource=resource,
                    mapping_csv=mapping_csv,
                    cache_dir=cache_dir,
                    include_generic_elements=False,
                )
                summary_tables = cerner_table_suggest.load_table_summaries(
                    table_summaries_path
                )
            except (FileNotFoundError, ValueError, OSError) as exc:
                log.error("%s", exc)
                return 2
            cov_thr = cerner_table_suggest.DEFAULT_COVERAGE_THRESHOLD
            cov_cand = 0
            weights_sm = cerner_table_suggest.parse_weights("")
            suggestions_sm = cerner_table_suggest.score_tables(
                fhir_text=fhir_text,
                tables=summary_tables,
                top_n=max(n_suggest, 1),
                weights=weights_sm,
                coverage_threshold=cov_thr,
                coverage_candidates=cov_cand,
                include_names=False,
                use_embeddings=use_embeddings,
            )
            log.info(
                "Top %d suggested seed tables for %s (summaries engine):\n%s",
                len(suggestions_sm),
                resource,
                cerner_table_suggest.format_suggestions(suggestions_sm),
            )
            if suggest_only:
                print(f"\nSuggested Cerner tables for {resource}:")
                print(cerner_table_suggest.format_suggestions(suggestions_sm))
                return 0
            if not seed_tables:
                seed_tables = [s.table_name for s in suggestions_sm]
                if not seed_tables:
                    log.error("No seed tables could be suggested. Aborting.")
                    return 3
                log.info(
                    "Using description-suggested seed tables: %s",
                    ", ".join(seed_tables),
                )
        else:
            log.info("Loading entire Cerner catalog for table suggestion: %s", columns_path)
            by_table_all = collect_all_tables(columns_path)
            log.info("Catalog contains %d tables", len(by_table_all))
            suggestions = suggest_seed_tables(
                resource=resource,
                fhir_paths=fhir_paths,
                fhir_types=fhir_types,
                fhir_query_texts=fhir_query_texts,
                by_table=by_table_all,
                n_suggestions=max(n_suggest, 1),
                prefilter_k=suggest_prefilter,
                coverage_threshold=coverage_threshold,
                column_weights=weights,
                table_weights=table_weights,
                use_embeddings=use_embeddings,
            )
            log.info(
                "Top %d suggested seed tables for %s:\n%s",
                len(suggestions),
                resource,
                format_table_suggestions(suggestions),
            )
            if suggest_only:
                print(f"\nSuggested seed tables for {resource}:")
                print(format_table_suggestions(suggestions))
                return 0
            if not seed_tables:
                seed_tables = [tbl for tbl, _, _ in suggestions]
                if not seed_tables:
                    log.error("No seed tables could be suggested. Aborting.")
                    return 3
                log.info("Using auto-suggested seed tables: %s", ", ".join(seed_tables))

    assert seed_tables is not None  # narrowed by the auto-suggest branch above

    log.info("Loading FK relationships %s", relationships_path)
    adjacency = relationship_adjacency(load_json(relationships_path))
    table_priorities: dict[str, level_aware.TablePriority] = {}
    if recommendation_mode == "level-aware":
        log.info("Using level-aware recommendation mode")
        table_priorities = level_aware.build_table_priorities(
            seed_tables, adjacency, max_distance
        )
        table_distances = {
            table: priority.distance for table, priority in table_priorities.items()
        }
    else:
        table_distances = expand_tables_bfs(seed_tables, adjacency, max_distance)
    log.info(
        "Expanded %d seed tables to %d candidate tables (max_distance=%d)",
        len(seed_tables),
        len(table_distances),
        max_distance,
    )

    log.info("Loading Cerner columns %s", columns_path)
    columns, seen_tables = load_columns_in_tables(columns_path, table_distances)
    log.info(
        "Collected %d Cerner candidate columns across %d tables",
        len(columns),
        len(seen_tables),
    )
    seed_uppercase = {t.upper() for t in seed_tables}
    missing = sorted(seed_uppercase - seen_tables)
    if missing:
        log.warning("Seed tables with no columns in catalog: %s", ", ".join(missing))
    if not columns:
        log.error("No Cerner columns matched the supplied tables (after expansion). Aborting.")
        return 3

    fhir_texts_for_similarity = fhir_query_texts
    if recommendation_mode == "level-aware":
        table_summaries = level_aware.load_table_summaries(table_summaries_path)
        if table_summaries_path and not table_summaries:
            log.info(
                "No table summaries loaded from %s; using catalog column definitions only",
                table_summaries_path,
            )
        columns = level_aware.attach_level_context(
            columns, table_priorities, table_summaries
        )
        fhir_texts_for_similarity = [
            level_aware.enrich_fhir_query_text(path, text)
            for path, text in zip(fhir_paths, fhir_query_texts, strict=True)
        ]

    log.info("Computing description similarity (TF-IDF=%s)", not use_embeddings)
    column_texts = [c["definition_text"] for c in columns]
    sim_matrix = build_description_similarity_matrix(
        fhir_texts_for_similarity, column_texts, use_embeddings
    )

    fhir_terms_per_row = [fhir_path_terms(p) for p in fhir_paths]

    updated = 0
    for local_idx, row_idx in enumerate(fhir_indices):
        if recommendation_mode == "level-aware":
            matches = level_aware.rank_candidates_for_row(
                fhir_terms_per_row[local_idx],
                fhir_paths[local_idx],
                fhir_types[local_idx],
                sim_matrix[local_idx],
                columns,
                weights,
                max_distance,
                top_k,
                min_score,
                name_similarity_fn=name_similarity,
                type_compat_fn=type_compat,
                distance_score_fn=distance_score,
            )
        else:
            matches = rank_candidates_for_row(
                fhir_terms_per_row[local_idx],
                fhir_paths[local_idx],
                fhir_types[local_idx],
                sim_matrix[local_idx],
                columns,
                weights,
                max_distance,
                top_k,
                min_score,
            )
        rows[row_idx][source_idx] = format_source_fields(matches)
        if notes_idx >= 0:
            rows[row_idx][notes_idx] = strip_legacy_recommend_segment(
                cell(rows[row_idx], notes_idx)
            )
        if matches:
            updated += 1

    result_csv = results_dir / mapping_csv.name
    write_mapping_csv(result_csv, header, rows)

    by_distance: dict[int, int] = defaultdict(int)
    for d in table_distances.values():
        by_distance[d] += 1
    distance_summary = ", ".join(
        f"d={d}:{count}" for d, count in sorted(by_distance.items())
    )
    log.info(
        "Done. Filled Source Fields for %d/%d FHIR rows. Wrote result CSV to %s. "
        "Tables in candidate pool by distance: %s",
        updated,
        len(fhir_indices),
        result_csv,
        distance_summary,
    )
    seed_in_pool = [t for t in sorted(seed_uppercase) if t in seen_tables]
    log.info("Seed tables retained: %s", ", ".join(seed_in_pool) or "(none)")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recommend Cerner source columns for each FHIR element of a resource and "
            "write the completed CSV to results/{Resource}_mapping.csv."
        )
    )
    parser.add_argument("--resource", required=True, help="FHIR resource name, e.g. Person")
    parser.add_argument(
        "--tables",
        nargs="+",
        default=None,
        help=(
            "Seed Cerner table names (case-insensitive). If omitted, the script "
            "auto-suggests seeds (catalog scorer by default, or summaries-based "
            "when --suggest-tables is passed alone)."
        ),
    )
    parser.add_argument(
        "--suggest-tables",
        nargs="?",
        const=_SUGGEST_TABLES_SUMMARIES_ENGINE,
        metavar="N",
        default=argparse.SUPPRESS,
        help=(
            "Seed-table selection when --tables is omitted. Pass alone (no value) "
            "to rank seeds using table_summaries.json descriptions "
            "(same logic as suggest_cerner_tables.py), respecting --embeddings. "
            "Pass an integer N to use the embedded catalog hybrid scorer and keep "
            "the top N seeds (legacy). If this option is omitted, use the catalog "
            f"scorer with --num-seed-tables (default {DEFAULT_SUGGEST_TABLES})."
        ),
    )
    parser.add_argument(
        "--num-seed-tables",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of seed tables when auto-suggesting without --suggest-tables N "
            f"(catalog default engine or bare --suggest-tables). Default: "
            f"{DEFAULT_SUGGEST_TABLES}. Ignored when --tables is set or "
            "--suggest-tables has an explicit integer."
        ),
    )
    parser.add_argument(
        "--suggest-only",
        action="store_true",
        help="Print the auto-suggested tables and exit; do not modify the mapping CSV.",
    )
    parser.add_argument(
        "--suggest-prefilter",
        type=int,
        default=DEFAULT_SUGGEST_PREFILTER,
        help=(
            "How many tables to keep after the cheap name pre-filter before "
            "running content scoring. Larger = slower but more thorough."
        ),
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=DEFAULT_COVERAGE_THRESHOLD,
        help=(
            "Per-element column score above which a FHIR element counts as "
            "'covered' by a table for the coverage sub-score."
        ),
    )
    parser.add_argument(
        "--table-weights",
        type=str,
        default="",
        help=(
            "Comma-separated key=value list for table suggestion, e.g. "
            "'name=0.4,content=0.4,coverage=0.2'. Renormalized to sum to 1."
        ),
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=DEFAULT_MAX_DISTANCE,
        help="FK BFS hop limit from seed tables. 0 = strict supplied tables only.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Maximum candidates per FHIR row. 0 = keep all above --min-score.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Drop candidates whose combined score is below this threshold.",
    )
    parser.add_argument(
        "--columns-json",
        type=Path,
        default=DEFAULT_COLUMNS_JSON,
        help="Path to parsed Cerner columns.json.",
    )
    parser.add_argument(
        "--relationships-json",
        type=Path,
        default=DEFAULT_RELATIONSHIPS_JSON,
        help="Path to parsed Cerner relationships.json.",
    )
    parser.add_argument(
        "--table-summaries-json",
        type=Path,
        default=DEFAULT_TABLE_SUMMARIES_JSON,
        help=(
            "Optional path to parsed Cerner table_summaries.json. Used by "
            "--recommendation-mode level-aware to enrich table/column descriptions."
        ),
    )
    parser.add_argument(
        "--mappings-dir",
        type=Path,
        default=DEFAULT_MAPPINGS_DIR,
        help="Directory containing {Resource}_mapping.csv files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where completed {Resource}_mapping.csv files are written.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for cached FHIR element definitions.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help=(
            "Comma-separated key=value list, e.g. 'name=0.45,desc=0.35,type=0.10,dist=0.10'. "
            "Weights are renormalized to sum to 1."
        ),
    )
    parser.add_argument(
        "--recommendation-mode",
        choices=("hybrid", "level-aware"),
        default="hybrid",
        help=(
            "Recommendation strategy. 'hybrid' keeps the existing scorer; "
            "'level-aware' uses ordered seed tables as priority levels and "
            "scores columns in table context."
        ),
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Use sentence-transformers/all-MiniLM-L6-v2 for description similarity.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    weights = parse_weights(args.weights)
    table_weights = parse_table_weights(args.table_weights)

    raw_st = getattr(args, "suggest_tables", None)
    num_override = getattr(args, "num_seed_tables", None)
    if raw_st is None:
        table_suggestion_engine = "catalog"
        n_suggest = max(num_override if num_override is not None else DEFAULT_SUGGEST_TABLES, 1)
    elif raw_st == _SUGGEST_TABLES_SUMMARIES_ENGINE:
        table_suggestion_engine = "summaries"
        n_suggest = max(num_override if num_override is not None else DEFAULT_SUGGEST_TABLES, 1)
    else:
        table_suggestion_engine = "catalog"
        try:
            n_suggest = max(int(raw_st), 1)
        except ValueError:
            log.error("--suggest-tables must be omitted, bare, or an integer.")
            return 2

    return run(
        resource=args.resource,
        seed_tables=args.tables,
        columns_path=args.columns_json,
        relationships_path=args.relationships_json,
        table_summaries_path=args.table_summaries_json,
        mappings_dir=args.mappings_dir,
        results_dir=args.results_dir,
        cache_dir=args.cache_dir,
        max_distance=max(0, args.max_distance),
        top_k=args.top_k,
        min_score=args.min_score,
        weights=weights,
        recommendation_mode=args.recommendation_mode,
        use_embeddings=args.embeddings,
        n_suggest=n_suggest,
        suggest_prefilter=max(args.suggest_prefilter, 1),
        coverage_threshold=args.coverage_threshold,
        table_weights=table_weights,
        suggest_only=args.suggest_only,
        table_suggestion_engine=table_suggestion_engine,
    )


if __name__ == "__main__":
    sys.exit(main())
