#!/usr/bin/env python3
"""Suggest Cerner seed tables from FHIR and Cerner descriptions.

This script is intentionally table-focused. It does not recommend columns or
write mapping CSVs; it only ranks Cerner tables for a FHIR resource.

Unlike the table suggestion embedded in ``recommend_cerner_mapping.py``, the
primary signal here is description similarity:

* FHIR resource and element descriptions from the HL7 StructureDefinition
* Cerner table descriptions from ``table_summaries.json``
* Optional reranking from Cerner column descriptions for table coverage

Table names are kept out of the default score because Cerner naming often does
not match FHIR naming. They are only used as a deterministic tie-breaker.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import string
import sys
import urllib.error
import urllib.request
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except ImportError:  # pragma: no cover - optional fast path
    TfidfVectorizer = None  # type: ignore[assignment]
    cosine_similarity = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MAPPINGS_DIR = SCRIPT_DIR / "mappings"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "mappings" / "_cache"
DEFAULT_TABLE_SUMMARIES_JSON = Path(
    "/home/basel/Hevelian/Specifications/CERNER/2018.01 Main/parsed_json/table_summaries.json"
)
DEFAULT_TOP_N = 10
DEFAULT_COVERAGE_THRESHOLD = 0.18
DEFAULT_WEIGHTS: dict[str, float] = {
    "description": 0.55,
    "columns": 0.35,
    "coverage": 0.10,
}

GENERIC_FHIR_TAILS = {
    "id",
    "meta",
    "implicitRules",
    "language",
    "text",
    "contained",
    "extension",
    "modifierExtension",
}

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("suggest_tables")

PUNCT_TR = str.maketrans({char: " " for char in string.punctuation})
CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")
INDEX_RE = re.compile(r"\[\d+\]")


@dataclass(frozen=True)
class FhirProfileText:
    resource_text: str
    element_texts: list[str]
    element_paths: list[str]


@dataclass(frozen=True)
class CernerTableSummary:
    table_name: str
    description: str
    columns: list[dict[str, str]]


@dataclass(frozen=True)
class TableSuggestion:
    table_name: str
    score: float
    description_score: float
    column_score: float
    coverage: float
    n_columns: int
    description: str


def normalize(text: str) -> str:
    return " ".join(text.lower().translate(PUNCT_TR).split())


def split_camel(text: str) -> str:
    return CAMEL_RE.sub(" ", text)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_mapping_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows


def find_column_index(header: list[str], name: str) -> int:
    try:
        return header.index(name)
    except ValueError as exc:
        raise ValueError(
            f"Mapping CSV is missing required column {name!r}; got header {header!r}"
        ) from exc


def fetch_fhir_definitions(resource: str, cache_dir: Path) -> dict[str, dict[str, str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"fhir_element_defs_{resource}.json"
    if cache_file.exists():
        cached = load_json(cache_file)
        if isinstance(cached, dict):
            log.info(
                "Loaded %d cached FHIR element definitions from %s",
                len(cached),
                cache_file,
            )
            return cached

    url = f"https://hl7.org/fhir/R4/{resource.lower()}.profile.json"
    log.info("Fetching FHIR definitions from %s", url)
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "suggest-cerner-tables/1.0"},
        )
        with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        log.warning(
            "Could not fetch FHIR definitions (%s). Falling back to CSV descriptions.",
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
        return "", ""
    return entry.get("short") or "", entry.get("definition") or ""


def build_fhir_query_text(fhir_path: str, defs: dict[str, dict[str, str]]) -> str:
    short, definition = lookup_fhir_definition(fhir_path, defs)
    tail = INDEX_RE.sub("", fhir_path.split(".", 1)[-1] if "." in fhir_path else fhir_path)
    tail_phrase = " ".join(split_camel(tail).replace(".", " ").split())
    pieces = [tail_phrase, short, definition]
    return " ".join(piece for piece in pieces if piece).strip()


def tokenize_for_tfidf(text: str) -> list[str]:
    tokens = [
        token
        for token in normalize(split_camel(text)).split()
        if len(token) > 1 and token not in STOP_WORDS
    ]
    bigrams = [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]
    return tokens + bigrams


def build_tfidf_similarity_fallback(
    fhir_texts: list[str],
    cerner_texts: list[str],
) -> list[list[float]]:
    corpus = cerner_texts + fhir_texts
    tokenized = [tokenize_for_tfidf(text) for text in corpus]
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))

    n_docs = len(tokenized)
    idf = {
        token: math.log((1 + n_docs) / (1 + freq)) + 1.0
        for token, freq in doc_freq.items()
    }

    vectors: list[dict[str, float]] = []
    norms: list[float] = []
    for tokens in tokenized:
        counts = Counter(tokens)
        vec = {
            token: (1.0 + math.log(count)) * idf[token]
            for token, count in counts.items()
        }
        norm = math.sqrt(sum(weight * weight for weight in vec.values()))
        vectors.append(vec)
        norms.append(norm)

    cerner_vectors = vectors[: len(cerner_texts)]
    cerner_norms = norms[: len(cerner_texts)]
    fhir_vectors = vectors[len(cerner_texts) :]
    fhir_norms = norms[len(cerner_texts) :]

    matrix: list[list[float]] = []
    for f_vec, f_norm in zip(fhir_vectors, fhir_norms):
        row: list[float] = []
        for c_vec, c_norm in zip(cerner_vectors, cerner_norms):
            if f_norm == 0.0 or c_norm == 0.0:
                row.append(0.0)
                continue
            if len(f_vec) < len(c_vec):
                dot = sum(weight * c_vec.get(token, 0.0) for token, weight in f_vec.items())
            else:
                dot = sum(f_vec.get(token, 0.0) * weight for token, weight in c_vec.items())
            row.append(max(0.0, min(dot / (f_norm * c_norm), 1.0)))
        matrix.append(row)
    return matrix


def matrix_value(matrix: Any, row: int, col: int) -> float:
    try:
        return float(matrix[row, col])
    except TypeError:
        return float(matrix[row][col])


def build_description_similarity_matrix(
    fhir_texts: list[str],
    cerner_texts: list[str],
    use_embeddings: bool,
) -> Any:
    """Return an (n_fhir, n_cerner) cosine-similarity matrix in [0, 1]."""
    if not fhir_texts or not cerner_texts:
        return [[0.0 for _ in cerner_texts] for _ in fhir_texts]
    if use_embeddings:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise SystemExit(
                "sentence-transformers not installed; pip install sentence-transformers"
            ) from exc
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        f_emb = model.encode(fhir_texts, show_progress_bar=False, normalize_embeddings=True)
        c_emb = model.encode(cerner_texts, show_progress_bar=False, normalize_embeddings=True)
        sim = f_emb @ c_emb.T
        import numpy as np

        return np.clip(sim, 0.0, 1.0)

    if TfidfVectorizer is None or cosine_similarity is None:
        log.info("scikit-learn not installed; using built-in TF-IDF scorer")
        return build_tfidf_similarity_fallback(fhir_texts, cerner_texts)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]+\b",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        stop_words="english",
    )
    corpus = cerner_texts + fhir_texts
    vectorizer.fit(corpus)
    fhir_vec = vectorizer.transform(fhir_texts)
    cerner_vec = vectorizer.transform(cerner_texts)
    sim = cosine_similarity(fhir_vec, cerner_vec)
    import numpy as np

    return np.clip(sim, 0.0, 1.0)


def parse_weights(weights_str: str) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if weights_str:
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
        weights = dict(DEFAULT_WEIGHTS)
        total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def path_tail(fhir_path: str) -> str:
    tail = fhir_path.split(".", 1)[-1] if "." in fhir_path else fhir_path
    return tail.split(".", 1)[0].split("[", 1)[0]


def load_fhir_profile_text(
    resource: str,
    mapping_csv: Path,
    cache_dir: Path,
    include_generic_elements: bool,
) -> FhirProfileText:
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")

    header, rows = load_mapping_csv(mapping_csv)
    full_path_idx = find_column_index(header, "Full FHIR Path")
    description_idx = header.index("Description") if "Description" in header else -1
    fhir_defs = fetch_fhir_definitions(resource, cache_dir)

    resource_def = fhir_defs.get(resource, {})
    resource_pieces = [
        resource,
        resource_def.get("short", ""),
        resource_def.get("definition", ""),
    ]

    element_texts: list[str] = []
    element_paths: list[str] = []
    for row in rows:
        fhir_path = row[full_path_idx].strip() if len(row) > full_path_idx else ""
        if not fhir_path:
            continue
        if not include_generic_elements and path_tail(fhir_path) in GENERIC_FHIR_TAILS:
            continue

        fhir_text = build_fhir_query_text(fhir_path, fhir_defs)
        csv_description = (
            row[description_idx].strip()
            if description_idx >= 0 and len(row) > description_idx
            else ""
        )
        combined = " ".join(
            piece for piece in [fhir_path, fhir_text, csv_description] if piece
        ).strip()
        if combined:
            element_paths.append(fhir_path)
            element_texts.append(combined)

    if not element_texts:
        raise ValueError(f"No FHIR paths found in {mapping_csv}")

    # Resource text is intentionally description-heavy, with element text added
    # so sparse resource definitions still carry resource-specific meaning.
    resource_text = " ".join(
        piece for piece in [*resource_pieces, *element_texts] if piece
    ).strip()
    return FhirProfileText(
        resource_text=resource_text,
        element_texts=element_texts,
        element_paths=element_paths,
    )


def load_table_summaries(path: Path) -> list[CernerTableSummary]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Unexpected table_summaries.json shape: {type(raw).__name__}")

    tables: list[CernerTableSummary] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        table_name = str(entry.get("table_name") or "").upper()
        if not table_name:
            continue
        description = str(entry.get("description") or "")
        columns: list[dict[str, str]] = []
        for column in entry.get("columns") or []:
            if not isinstance(column, dict):
                continue
            name = str(column.get("name") or "").upper()
            column_description = str(column.get("description") or "")
            if name or column_description:
                columns.append({"name": name, "description": column_description})
        tables.append(
            CernerTableSummary(
                table_name=table_name,
                description=description,
                columns=columns,
            )
        )
    return tables


def _table_description_text(table: CernerTableSummary, include_names: bool) -> str:
    pieces = [table.description]
    if include_names:
        pieces.insert(0, table.table_name)
    return " ".join(piece for piece in pieces if piece).strip()


def _column_description_text(
    table_name: str,
    column: dict[str, str],
    include_names: bool,
) -> str:
    pieces = [column.get("description", "")]
    if include_names:
        pieces.insert(0, f"{table_name} {column.get('name', '')}")
    return " ".join(piece for piece in pieces if piece).strip()


def score_tables(
    fhir_text: FhirProfileText,
    tables: list[CernerTableSummary],
    top_n: int,
    weights: dict[str, float],
    coverage_threshold: float,
    coverage_candidates: int,
    include_names: bool,
    use_embeddings: bool,
) -> list[TableSuggestion]:
    if not tables:
        return []

    table_texts = [_table_description_text(table, include_names) for table in tables]
    log.info("Scoring %d Cerner table descriptions", len(tables))
    desc_matrix = build_description_similarity_matrix(
        [fhir_text.resource_text],
        table_texts,
        use_embeddings,
    )
    description_scores = [float(score) for score in desc_matrix[0]]

    candidate_count = len(tables) if coverage_candidates <= 0 else coverage_candidates
    candidate_count = min(max(candidate_count, top_n), len(tables))
    candidate_indices = sorted(
        range(len(tables)),
        key=lambda idx: (-description_scores[idx], tables[idx].table_name),
    )[:candidate_count]

    column_scores_by_table: dict[str, float] = defaultdict(float)
    coverage_by_table: dict[str, float] = defaultdict(float)
    column_count_by_table: dict[str, int] = defaultdict(int)

    column_texts: list[str] = []
    column_table_names: list[str] = []
    for idx in candidate_indices:
        table = tables[idx]
        for column in table.columns:
            text = _column_description_text(table.table_name, column, include_names)
            if not normalize(text):
                continue
            column_texts.append(text)
            column_table_names.append(table.table_name)
            column_count_by_table[table.table_name] += 1

    if column_texts:
        log.info(
            "Scoring column-description coverage for %d columns in %d tables",
            len(column_texts),
            len(candidate_indices),
        )
        col_matrix = build_description_similarity_matrix(
            fhir_text.element_texts,
            column_texts,
            use_embeddings,
        )
        table_to_col_indices: dict[str, list[int]] = defaultdict(list)
        for col_idx, table_name in enumerate(column_table_names):
            table_to_col_indices[table_name].append(col_idx)

        for idx in candidate_indices:
            table = tables[idx]
            col_indices = table_to_col_indices.get(table.table_name, [])
            if not col_indices:
                continue

            per_element_best: list[float] = []
            for elem_idx in range(len(fhir_text.element_texts)):
                best = max(matrix_value(col_matrix, elem_idx, col_idx) for col_idx in col_indices)
                per_element_best.append(best)

            column_scores_by_table[table.table_name] = sum(per_element_best) / len(
                per_element_best
            )
            coverage_by_table[table.table_name] = sum(
                1 for score in per_element_best if score >= coverage_threshold
            ) / len(per_element_best)

    suggestions: list[TableSuggestion] = []
    for idx in candidate_indices:
        table = tables[idx]
        description_score = description_scores[idx]
        column_score = column_scores_by_table[table.table_name]
        coverage = coverage_by_table[table.table_name]
        total = (
            weights["description"] * description_score
            + weights["columns"] * column_score
            + weights["coverage"] * coverage
        )
        suggestions.append(
            TableSuggestion(
                table_name=table.table_name,
                score=total,
                description_score=description_score,
                column_score=column_score,
                coverage=coverage,
                n_columns=column_count_by_table[table.table_name] or len(table.columns),
                description=table.description,
            )
        )

    suggestions.sort(
        key=lambda item: (
            -item.score,
            -item.description_score,
            -item.column_score,
            item.table_name,
        )
    )
    return suggestions[:top_n]


def format_suggestions(suggestions: list[TableSuggestion]) -> str:
    if not suggestions:
        return "(no candidates)"
    lines: list[str] = []
    for rank, suggestion in enumerate(suggestions, start=1):
        description = " ".join(suggestion.description.split())
        if len(description) > 110:
            description = f"{description[:107]}..."
        lines.append(
            "  {rank:>2}. {table:<40s} score={score:.3f}  "
            "desc={desc:.2f} columns={columns:.2f} coverage={coverage:.2f} "
            "cols={cols:d}  {description}".format(
                rank=rank,
                table=suggestion.table_name,
                score=suggestion.score,
                desc=suggestion.description_score,
                columns=suggestion.column_score,
                coverage=suggestion.coverage,
                cols=suggestion.n_columns,
                description=description,
            )
        )
    return "\n".join(lines)


def write_suggestions_csv(path: Path, suggestions: list[TableSuggestion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "rank",
                "table_name",
                "score",
                "description_score",
                "column_score",
                "coverage",
                "n_columns",
                "description",
            ]
        )
        for rank, suggestion in enumerate(suggestions, start=1):
            writer.writerow(
                [
                    rank,
                    suggestion.table_name,
                    f"{suggestion.score:.6f}",
                    f"{suggestion.description_score:.6f}",
                    f"{suggestion.column_score:.6f}",
                    f"{suggestion.coverage:.6f}",
                    suggestion.n_columns,
                    suggestion.description,
                ]
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Suggest Cerner tables by comparing FHIR descriptions to Cerner table "
            "and column descriptions."
        )
    )
    parser.add_argument("--resource", required=True, help="FHIR resource name, e.g. Patient.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of table suggestions to print. Default: {DEFAULT_TOP_N}.",
    )
    parser.add_argument(
        "--table-summaries-json",
        type=Path,
        default=DEFAULT_TABLE_SUMMARIES_JSON,
        help="Path to Cerner parsed_json/table_summaries.json.",
    )
    parser.add_argument(
        "--mappings-dir",
        type=Path,
        default=DEFAULT_MAPPINGS_DIR,
        help="Directory containing {Resource}_mapping.csv.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for cached FHIR StructureDefinition descriptions.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=DEFAULT_COVERAGE_THRESHOLD,
        help=(
            "Element-to-column description score counted as coverage. Default: "
            f"{DEFAULT_COVERAGE_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--coverage-candidates",
        type=int,
        default=0,
        help=(
            "How many tables receive column-description reranking after all table "
            "descriptions are scored. 0 means all tables. Default: 0."
        ),
    )
    parser.add_argument(
        "--weights",
        default="",
        help=(
            "Comma-separated key=value list for description,columns,coverage. "
            "Default: description=0.55,columns=0.35,coverage=0.10."
        ),
    )
    parser.add_argument(
        "--include-names",
        action="store_true",
        help=(
            "Include Cerner table/column names in the text sent to the similarity "
            "model. Off by default because descriptions are usually more accurate."
        ),
    )
    parser.add_argument(
        "--include-generic-elements",
        action="store_true",
        help="Keep generic FHIR metadata elements such as id/meta/text in the query.",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Use sentence-transformers/all-MiniLM-L6-v2 instead of TF-IDF.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to write the ranked suggestions as CSV.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    weights = parse_weights(args.weights)
    mapping_csv = args.mappings_dir / f"{args.resource}_mapping.csv"

    try:
        fhir_text = load_fhir_profile_text(
            resource=args.resource,
            mapping_csv=mapping_csv,
            cache_dir=args.cache_dir,
            include_generic_elements=args.include_generic_elements,
        )
        tables = load_table_summaries(args.table_summaries_json)
        suggestions = score_tables(
            fhir_text=fhir_text,
            tables=tables,
            top_n=max(args.top_n, 1),
            weights=weights,
            coverage_threshold=args.coverage_threshold,
            coverage_candidates=args.coverage_candidates,
            include_names=args.include_names,
            use_embeddings=args.embeddings,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("%s", exc)
        return 2

    print(f"\nSuggested Cerner tables for {args.resource}:")
    print(format_suggestions(suggestions))
    if args.output_csv:
        write_suggestions_csv(args.output_csv, suggestions)
        log.info("Wrote table suggestions to %s", args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
