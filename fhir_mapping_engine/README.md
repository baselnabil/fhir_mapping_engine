<<<<<<< HEAD
# Specifications — FHIR ↔ Cerner Mapping Workbench

This repository is a **specification workbench** for designing and maintaining
mappings between **HL7 FHIR R4 resources** and **Oracle Cerner Millennium**
source columns. For every FHIR resource we care about it provides:

- a JSON skeleton of the resource (`mappings/{Resource}.json`),
- a CSV mapping spreadsheet keyed by FHIR element path
  (`mappings/{Resource}_mapping.csv`),
- a slimmed-down "top-level only" copy of that spreadsheet
  (`mappings_filtered/{Resource}_mapping.csv`),
- two Python tools that automate the two most repetitive parts of the
  authoring workflow:
  1. **`recommend_cerner_mapping.py`** — auto-suggests Cerner tables and
     fills the `Cerner Source Fields` column with the most likely candidate
     columns for each FHIR element.
  2. **`filter_top_level_mapping.py`** — produces a depth-limited copy of a
     mapping CSV (top-level FHIR elements + first-level children by default).

A reference Mirth Connect FHIR integration (`Cerner/mirth-fhir/`) is included
for context — it ships the channels and code template that some teams use to
expose Cerner data as FHIR resources, and is independent from the Python
tooling.

---

## Repository layout

```
Specifications/
├── README.md                         (this file)
├── requirements.txt                  Python deps for the two scripts
├── recommend_cerner_mapping.py       Cerner column / table recommender
├── filter_top_level_mapping.py       Depth-limited mapping CSV exporter
│
├── mappings/                         Authoring workspace (one set per resource)
│   ├── {Resource}.json               FHIR R4 resource skeleton
│   ├── {Resource}_mapping.csv        Editable mapping spreadsheet
│   ├── {Resource}_mapping.csv.bak    Auto-backup (created by the recommender)
│   ├── Patient_cerner_mapping.csv    Legacy / alternative-format Patient map
│   └── _cache/
│       └── fhir_element_defs_*.json  Cached FHIR StructureDefinition snapshots
│
├── mappings_filtered/                Output of filter_top_level_mapping.py
│   └── {Resource}_mapping.csv        depth ≤ 2 view of the mapping above
│
└── Cerner/
    └── mirth-fhir/                   Reference Mirth Connect FHIR channels
        ├── README.md
        ├── mirth-fhir1.0.0.jar
        └── *.xml                     channels, code templates, global scripts
```

The tooling expects the parsed Cerner data dictionary to live **outside this
repo** at:

```
/home/basel/Hevelian/Fhir_automation/CERNER/2018.01 Main/parsed_json/
├── columns.json          (column catalog: table_name, column_name, data_type, definition)
├── relationships.json    (foreign-key adjacency)
├── tables.json
├── indexes.json
└── ...
```

Both paths are configurable via CLI flags (`--columns-json`,
`--relationships-json`); the constants in
`recommend_cerner_mapping.py` (`DEFAULT_COLUMNS_JSON`,
`DEFAULT_RELATIONSHIPS_JSON`) just provide the defaults.

---

## The mapping CSV schema

Every `mappings/{Resource}_mapping.csv` shares the same header. Empty
columns between `FHIR Path` and `Element Type` are intentional: they implement
a visual indentation ladder, where the FHIR path is written into a deeper
column the deeper the element sits in the resource tree.

| Column | Meaning |
| --- | --- |
| `Outline` | Optional outline number (`0.1`, `1.`, …) for human navigation. |
| `FHIR Path` … (6 unnamed columns) | Indented FHIR element path. The depth of the column expresses nesting; only one of these cells is filled per row. |
| `Element Type` | FHIR data type of the element (`id`, `string`, `Coding`, `Reference(Patient)`, …). |
| `FHIR Cardinality` | The standard FHIR cardinality, e.g. `0..1`, `1..*`. |
| `Mapping Type` | One of `Direct`, `Transform`, `Fixed`, `Nested`, … – describes *how* the source feeds the FHIR element. |
| `Cerner Source Fields` | Pipe-separated list of `TABLE.COLUMN` candidates. Auto-filled by `recommend_cerner_mapping.py`. |
| `Fixed Value / Code System` | Constant value (for `Mapping Type = Fixed`) or canonical code system URL. |
| `Required in source` | Whether the source data must be present. |
| `Notes` | Free-form notes; the recommender preserves these and only strips legacy `[recommend] …` segments it inserted in earlier versions. |
| `Required/Extensible value set in FHIR/NPHIES` | Bound value-set reference (FHIR or NPHIES). |
| `Full FHIR Path` | The flat, unambiguous path (e.g. `Person.identifier[0].value`). **This is the primary key the tools rely on.** |

> Backwards compatibility: the recommender accepts either `Cerner Source Fields`
> *or* the legacy `Source Fields` header on input and rewrites it to the
> canonical name `Cerner Source Fields` on output.

`Patient_cerner_mapping.csv` is an alternative, **flat** legacy format
(`Output Column,FHIR Path,FHIR Cardinality,Type,Cerner Source,Confidence,Notes`)
left in place for reference. The tooling does not touch it (`filter_top_level_mapping.py`
explicitly skips files whose name contains `_cerner_`).

---

## Setup

### 1. Python environment

A `.venv/` is already provisioned in this folder. From scratch you would do:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the runtime essentials:

```
rapidfuzz>=3.6
scikit-learn>=1.3
numpy>=1.24
# Optional: only needed when running with --embeddings
# sentence-transformers>=2.5
```

If you want the (optional) sentence-transformer description scoring
(`--embeddings`), install it explicitly:

```bash
pip install sentence-transformers
```

### 2. Cerner catalog

`recommend_cerner_mapping.py` reads `columns.json` and `relationships.json`.
Either keep them at the default location shown above, or override per-run:

```bash
python recommend_cerner_mapping.py --resource Person \
    --columns-json /path/to/columns.json \
    --relationships-json /path/to/relationships.json
```

### 3. FHIR element definitions

The recommender enriches its description-similarity scoring with FHIR
element `short` and `definition` strings fetched from
`https://hl7.org/fhir/R4/{resource}.profile.json`. Results are cached to
`mappings/_cache/fhir_element_defs_{Resource}.json`. The script degrades
gracefully (path-token-only matching) when the network is unavailable.

---

## Tool 1 — `recommend_cerner_mapping.py`

Auto-fills the `Cerner Source Fields` column of an existing
`mappings/{Resource}_mapping.csv` with ranked Cerner column candidates.
A `.bak` of the original CSV is written before any mutation.

### What it does, in order

1. **Load the mapping CSV** and parse every row that has a `Full FHIR Path`.
2. **Fetch (and cache) FHIR element definitions** for the target resource.
3. **Pick seed Cerner tables**:
   - if `--tables T1 T2 …` was provided, those are the seeds;
   - otherwise the script ranks every Cerner table against the resource's
     FHIR elements and uses the top `--suggest-tables` (default 5) as seeds.
4. **Expand seeds via the FK graph** with BFS up to `--max-distance` hops
   (default 2; `0` = strict supplied tables only). All columns in the
   reachable tables become the candidate pool.
5. **Score every FHIR element × candidate column** with a hybrid score:

   | Sub-score | Range | Description |
   | --- | --- | --- |
   | `name`  | 0..1 | Token overlap (with abbreviation / FHIR-synonym aliases) blended 50/50 with `rapidfuzz.fuzz.token_set_ratio`. |
   | `desc`  | 0..1 | TF-IDF cosine between the FHIR element definition (`short` + `definition` + path tail) and the Cerner column description (`TABLE COLUMN definition`). With `--embeddings`, uses `sentence-transformers/all-MiniLM-L6-v2` instead. |
   | `type`  | 0..1 | Compatibility heuristics between the FHIR data type and the Cerner `data_type` (e.g. `dateTime` ↔ `*_DT_TM`, `boolean` ↔ `*_IND`/`*_FLAG`, `code` ↔ `*_CD`). |
   | `dist`  | 0..1 | `1 − distance / (max_distance + 1)`, so closer tables rank higher. |

   Final score = weighted sum (default `name=0.45, desc=0.35, type=0.10, dist=0.10`,
   renormalized so the weights sum to 1). Override with `--weights "name=0.5,desc=0.3,type=0.1,dist=0.1"`.

6. **Keep the top `--top-k` candidates** per FHIR element above
   `--min-score`, render them as `TABLE.COL | TABLE.COL | …`, and write them
   into the `Cerner Source Fields` column. The pipe separator avoids
   collisions with locales that auto-detect `;` as a CSV delimiter.

### Table suggestion (when `--tables` is omitted)

The same column-level score drives a table-level aggregate:

| Sub-score | Description |
| --- | --- |
| `name` | Resource-name vs. table-name similarity (Jaccard + RapidFuzz). |
| `content` | For each FHIR element, take the best column score within the table; average across elements. |
| `coverage` | Fraction of FHIR elements whose best column-score in the table is at least `--coverage-threshold` (default 0.30). |

Default table weights: `name=0.40, content=0.40, coverage=0.20`
(override via `--table-weights`). To save time on the ~10k-table catalog
the script first prefilters by name to `--suggest-prefilter` (default 80)
candidates before running content scoring.

### Description-first table suggestions

Use `suggest_cerner_tables.py` when you only want seed-table suggestions and
want table descriptions to drive the ranking:

```bash
python3 suggest_cerner_tables.py --resource HealthcareService --top-n 10
```

This script compares the FHIR resource/element descriptions against every
Cerner table description in `table_summaries.json`, then reranks by how well
the table's column descriptions cover the FHIR elements. Cerner table and
column names are excluded from the default score because names often differ
from FHIR naming; pass `--include-names` only when you want names to influence
the similarity model.

### CLI reference

```
python recommend_cerner_mapping.py --resource RESOURCE [options]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--resource` | (required) | FHIR resource name, e.g. `Person`, `HealthcareService`. |
| `--tables T1 T2 …` | auto | Seed Cerner tables (case-insensitive). Skips auto-suggestion. |
| `--suggest-tables N` | 5 | How many seed tables to auto-suggest when `--tables` is omitted. |
| `--suggest-only` | off | Print the auto-suggested tables and exit; do not modify the CSV. |
| `--suggest-prefilter K` | 80 | How many tables survive the cheap name pre-filter before content scoring. |
| `--coverage-threshold X` | 0.30 | Per-element score above which a column "covers" a FHIR element. |
| `--table-weights "name=…,content=…,coverage=…"` | balanced | Weights for table-level aggregation. |
| `--max-distance D` | 2 | FK BFS hop limit from the seed tables. `0` = no expansion. |
| `--top-k K` | 7 | Max candidates per FHIR row. `0` = keep all above `--min-score`. |
| `--min-score X` | 0.05 | Drop candidates below this combined score. |
| `--columns-json PATH` | hard-coded default | Cerner column catalog. |
| `--relationships-json PATH` | hard-coded default | Cerner FK adjacency. |
| `--mappings-dir PATH` | `./mappings` | Where to find `{Resource}_mapping.csv`. |
| `--cache-dir PATH` | `./mappings/_cache` | Where to cache FHIR element definitions. |
| `--weights "name=…,desc=…,type=…,dist=…"` | balanced | Column-level scoring weights. |
| `--embeddings` | off | Use `sentence-transformers/all-MiniLM-L6-v2` for description similarity. |

### Worked examples

```bash
source .venv/bin/activate

python recommend_cerner_mapping.py --resource Person \
    --tables PERSON_PATIENT PERSON

python recommend_cerner_mapping.py --resource Person

python recommend_cerner_mapping.py --resource HealthcareService --suggest-only

python recommend_cerner_mapping.py --resource Patient \
    --tables PERSON PERSON_ALIAS \
    --max-distance 3 --top-k 10 \
    --weights "name=0.5,desc=0.4,type=0.05,dist=0.05" \
    --embeddings
```

After a run the script logs a per-distance summary, e.g.:

```
INFO Done. Filled Source Fields for 124/152 FHIR rows.
     Tables in candidate pool by distance: d=0:2, d=1:11, d=2:34
INFO Seed tables retained: PERSON, PERSON_PATIENT
```

---

## Tool 2 — `filter_top_level_mapping.py`

Produces a slimmed-down copy of a mapping CSV that keeps only top-level
elements (and their first-level children, by default). The original
CSV is never modified — output goes to
`mappings_filtered/{Resource}_mapping.csv`.

Depth is measured as the number of dot-separated segments **after** the
resource name; array indices like `tag[0]` count as part of the same
segment. With the default `--max-depth 2`:

```
Person.id                                 -> kept (depth 1)
Person.identifier                         -> kept (depth 1)
Person.identifier.type                    -> kept (depth 2)
Person.identifier.type.coding             -> dropped (depth 3)
HealthcareService.contact.address         -> kept (depth 2)
HealthcareService.contact.address.line[0] -> dropped (depth 3)
```

### CLI reference

```
python filter_top_level_mapping.py [--resource X | --all] [options]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--resource` | — | Filter just `mappings/{Resource}_mapping.csv`. |
| `--all` | off | Filter every primary `*_mapping.csv` in `--mappings-dir`. Skips `*.bak` and `*_cerner_*` files. |
| `--max-depth N` | 2 | Maximum FHIR-path depth to retain. |
| `--mappings-dir PATH` | `./mappings` | Source directory. |
| `--output-dir PATH` | `./mappings_filtered` | Destination directory. |

### Examples

```bash
python filter_top_level_mapping.py --resource HealthcareService
python filter_top_level_mapping.py --all
python filter_top_level_mapping.py --resource Person --max-depth 1
```

---

## Recommended authoring workflow

1. **Start a new resource**: drop a fresh `mappings/{Resource}_mapping.csv`
   (with at least the `Full FHIR Path`, `Element Type`, and an empty
   `Cerner Source Fields` / `Source Fields` column) into `mappings/`.
2. **Pick seed tables**:
   ```bash
   python recommend_cerner_mapping.py --resource {Resource} --suggest-only
   ```
   Inspect the ranked output, then either trust the suggestion or pick
   your own.
3. **Generate candidates**:
   ```bash
   python recommend_cerner_mapping.py --resource {Resource} \
       --tables T1 T2 …
   ```
   The CSV's `Cerner Source Fields` column is filled in place; the
   original is preserved as `{Resource}_mapping.csv.bak`.
4. **Manually refine** the CSV — keep / drop candidates, fill in
   `Mapping Type`, `Fixed Value / Code System`, etc.
5. **Re-run** the recommender any time the seed list, weights, or
   thresholds change. The `.bak` is only created the first time so
   subsequent runs do not clobber your original.
6. **Export a top-level summary** for review or downstream consumers:
   ```bash
   python filter_top_level_mapping.py --all
   ```

---

## Caveats and design notes

- **Recommendations are a starting point, not ground truth.** The
  scoring is unsupervised and tuned on naming heuristics; always
  inspect the output before committing.
- **Backups.** The recommender writes `{Resource}_mapping.csv.bak` only
  on the first run; subsequent runs overwrite the live CSV but leave
  the backup untouched, so the very first version is always
  recoverable.
- **Header migration.** If a CSV still uses the legacy `Source Fields`
  header, the recommender renames it to `Cerner Source Fields` on
  write. `filter_top_level_mapping.py` is header-agnostic and copies
  whatever header it finds.
- **Notes column.** The recommender deliberately strips any legacy
  `[recommend] …` segment it had previously written into `Notes`, so
  the column stays clean for human commentary.
- **No FHIR PHI.** The example IDs you may see in
  `Cerner/mirth-fhir/README.md` (e.g. `100000006`) are illustrative
  and not real patient identifiers.

---

## `Cerner/mirth-fhir/`

A read-only reference checkout of an open-source Mirth Connect FHIR
implementation. It exposes helpers like `getPatientFull(id)`,
`getPatientDemographics(id)`, and `getPatientResourceList(id, type)`
from a custom JAR (`mirth-fhir1.0.0.jar`) plus a set of importable
channels (`*.xml`). It is **not** invoked by the Python tooling and is
included purely as documentation for teams wiring Cerner ↔ FHIR through
Mirth. See `Cerner/mirth-fhir/README.md` for setup instructions.
=======
# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact
>>>>>>> 713a0a9 (Initial commit)
