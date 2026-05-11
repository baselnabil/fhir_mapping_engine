# Cerner Mapping Recommendation System

## Overview

The `recommend_cerner_mapping.py` script uses a **hybrid scoring system** to recommend Cerner database source columns for FHIR (Fast Healthcare Interoperability Resources) resource elements. It combines multiple machine learning and text-matching techniques to identify the most relevant columns from a Cerner database catalog.

The system operates in two phases:

1. **Table Suggestion**: Rank candidate Cerner tables for a given FHIR resource
2. **Column Recommendation**: For each FHIR element, rank and suggest the best matching columns from the candidate tables

---

## Machine Learning Techniques Used

### 1. **Name Similarity Scoring**

Combines two complementary approaches:

#### Jaccard Token Overlap
- Tokenizes both FHIR and Cerner names with CamelCase splitting
- Expands tokens using abbreviation aliases (e.g., `cd` → `{code}`, `dt` → `{date}`)
- Calculates Jaccard similarity: `|intersection| / |union|`
- Robust for matching abbreviated Cerner names to full FHIR names

#### Fuzzy String Matching (RapidFuzz)
- Uses token set ratio from RapidFuzz library
- Provides partial matching when exact token matching fails
- Normalized to [0, 1] range

**Combined Formula:**
```
name_score = 0.5 × jaccard + 0.5 × token_set_ratio
```

**Use Case:** Matches `birthdate` (FHIR) with `DOB` or `DT_BIRTH` (Cerner)

---

### 2. **Description Similarity Scoring**

Compares semantic meaning of FHIR element definitions with Cerner column definitions using text vectorization.

#### Option A: TF-IDF (Default, Fast)
- **TfidfVectorizer** from scikit-learn
- Configuration:
  - Unigrams + bigrams (`ngram_range=(1, 2)`)
  - Minimum document frequency: 1 (`min_df=1`)
  - Sublinear term frequency scaling (`sublinear_tf=True`)
- Computes cosine similarity between FHIR and Cerner definition vectors
- Fast, works well with short medical texts

#### Option B: Sentence Transformers (Optional, More Accurate)
- Uses pre-trained `sentence-transformers/all-MiniLM-L6-v2` model
- Generates normalized embeddings for FHIR and Cerner definitions
- Computes cosine similarity between embeddings
- Better semantic understanding but requires additional dependency
- ~10-30x slower than TF-IDF

**Enable with:** `--embeddings` flag

**Result:** Normalized cosine similarity in [0, 1]

---

### 3. **Type Compatibility Scoring**

Checks if the FHIR element data type matches the Cerner column data type.

**Type Match Rules:**

| FHIR Type | Cerner Data Type | Score |
|-----------|------------------|-------|
| `date`/`datetime`/`instant` | contains `date` | 1.0 (max) |
| `boolean` | contains `ind`, `indicator`, `flag` | 0.6 |
| `code` | contains `cd`, `code` | 0.6 |
| `integer`/`positiveint` | contains `number` | 0.4 |
| `decimal` | contains `number`/`float` | 0.4 |
| `string`/`uri`/`markdown` | contains `vc2`, `char`, `long` | 0.4 |
| (no match) | - | 0.0 |

**Implementation:** Raw scores mapped to [0, 1] via `min(raw / 5.0, 1.0)`

---

### 4. **Distance Score (Graph-Based)**

Scores candidate columns based on their FK (foreign key) relationship distance from seed tables.

**Formula:**
```
distance_score = 1.0 - (distance / (max_distance + 1))
```

- Seed tables have `distance = 0` → score = 1.0
- Distance increases through FK relationships
- Tables beyond `max_distance` hops are excluded
- Defaults to `max_distance=2`

**Why:** Cerner source tables are more reliable than distant related tables

---

### 5. **Hybrid Column-Level Scoring**

**Final column score:** Weighted combination of all components

```
total_score = w_name × name_score 
            + w_desc × desc_score 
            + w_type × type_score 
            + w_dist × distance_score
```

Default weights (normalized to sum to 1):
- `name`: 0.45 (45%)
- `desc`: 0.45 (45%)
- `type`: 0.5 (50%)
- `dist`: 0.5 (50%)

**Note:** All weights are renormalized to sum to 1.0 after parsing

---

### 6. **Table-Level Scoring** (For Auto-Suggestion)

When seed tables are not provided, the system auto-suggests them by scoring every Cerner table.

**Three Sub-Scores:**

#### Name Score
- Jaccard + RapidFuzz of resource name vs. table name
- Same as column name scoring

#### Content Score
- For each FHIR element, find the best-matching column in the table
- Average the best scores across all FHIR elements
- Reflects how well the table's columns cover the resource

#### Coverage Score
- Fraction of FHIR elements with at least one column scoring ≥ `coverage_threshold`
- Reflects breadth of coverage

**Table-Level Formula:**
```
table_score = tw_name × name_score 
            + tw_content × content_score 
            + tw_coverage × coverage_score
```

Default table weights:
- `name`: 0.40 (40%)
- `content`: 0.40 (40%)
- `coverage`: 0.20 (20%)

**Pre-Filtering:** Before scoring all tables, name similarity is computed and top `--suggest-prefilter` tables (default 80) are kept. This dramatically speeds up full scoring without losing quality.

---

### 7. **Semantic Enrichment**

#### Token Aliases
Maps Cerner abbreviations to FHIR-compatible tokens (lines 112–136):
```python
"cd": {"code"}
"dt": {"date"}
"id": {"identifier"}
"ind": {"indicator", "boolean"}
```

#### FHIR Synonyms
Maps FHIR terms to Cerner vocabulary (lines 139–158):
```python
"birthdate": {"birth", "born", "dob", "date"}
"gender": {"sex"}
"telecom": {"phone", "email", "contact", "fax"}
```

---

## Workflow

### Phase 1: Table Suggestion (if no `--tables` provided)

1. Load all Cerner tables from `columns.json`
2. Pre-filter by name similarity: keep top `--suggest-prefilter` tables
3. Fetch FHIR element definitions from hl7.org (or use cache)
4. Build description similarity matrix (TF-IDF or embeddings)
5. For each pre-filtered table, compute:
   - Name score (cheap)
   - Best column-element scores (using full hybrid scoring)
   - Coverage score
6. Combine into table-level score
7. Sort and return top `--suggest-tables` (default 5)
8. If `--suggest-only`: print and exit; otherwise use them as seeds

### Phase 2: Column Recommendation

1. Load seed tables (explicit or auto-suggested)
2. Build FK relationship adjacency graph from `relationships.json`
3. Expand seed tables via BFS up to `max_distance` hops
4. Load Cerner columns from expanded table set
5. Fetch FHIR element definitions (cached)
6. Build description similarity matrix
7. **For each FHIR element:**
   - Compute all component scores (name, desc, type, dist)
   - Rank columns by total score
   - Keep top `--top-k` candidates (default 5)
   - Filter by `--min-score` threshold (default 0.05)
8. Format and write results to CSV

---

## Tuning Parameters

### Column-Level Scoring Weights

Adjust the relative importance of different scoring components:

```bash
--weights "name=0.45,desc=0.45,type=0.5,dist=0.5"
```

**Examples:**

- **Prefer name matching over descriptions:**
  ```bash
  --weights "name=0.7,desc=0.1,type=0.1,dist=0.1"
  ```
  Use when: Column names closely mirror FHIR terms (common in well-designed schemas)

- **Prefer description matching:**
  ```bash
  --weights "name=0.2,desc=0.7,type=0.05,dist=0.05"
  ```
  Use when: Column names are cryptic, definitions are detailed

- **Strict type matching:**
  ```bash
  --weights "name=0.3,desc=0.3,type=0.35,dist=0.05"
  ```
  Use when: Data type mismatches cause problems downstream

- **Favor close tables (distance):**
  ```bash
  --weights "name=0.25,desc=0.25,type=0.25,dist=0.25"
  ```
  Use when: You prefer direct FK relationships over transitive ones

### Table Suggestion Weights

Adjust importance for table auto-suggestion:

```bash
--table-weights "name=0.4,content=0.4,coverage=0.2"
```

- **Prefer large, comprehensive tables:**
  ```bash
  --table-weights "name=0.1,content=0.6,coverage=0.3"
  ```

- **Prefer tables that cover all FHIR elements:**
  ```bash
  --table-weights "name=0.2,content=0.2,coverage=0.6"
  ```

### Candidate Selection

- **`--max-distance N`** (default: 2)
  - Limits FK relationship hops from seed tables
  - 0 = only seed tables, no FK expansion
  - Higher = more candidates but slower, potentially lower quality
  - Use 0 for strict schema conformance, 2-3 for flexibility

- **`--top-k N`** (default: 5)
  - Number of top candidates to return per FHIR element
  - 0 = keep all above `--min-score`
  - Higher = more options for manual review, but CSV gets cluttered

- **`--min-score THRESHOLD`** (default: 0.05)
  - Drop candidates with combined score below this
  - Range: [0.0, 1.0]
  - Higher = only high-confidence matches, fewer false positives

### Table Suggestion Refinement

- **`--suggest-tables N`** (default: 5)
  - How many seed tables to auto-suggest

- **`--suggest-prefilter N`** (default: 80)
  - Name-filter to this many tables before expensive content scoring
  - Lower = faster but risks missing good tables
  - Higher = thorough but slower (O(N×M) where N=tables, M=FHIR elements)

- **`--coverage-threshold SCORE`** (default: 0.30)
  - Per-element column score threshold for coverage calculation
  - A FHIR element counts as "covered" if any column scores ≥ this
  - Lower = tables appear more complete

### Description Similarity

- **`--embeddings`**
  - Use sentence-transformers instead of TF-IDF
  - Slower (~10-30x) but more semantically aware
  - Better for: Vague/generic definitions, multi-language content
  - Requires: `pip install sentence-transformers`
  - Trade-off: Speed vs. semantic accuracy

---

## Tuning Guide

### Scenario 1: Getting Started

Use defaults; they're balanced for most cases:

```bash
python recommend_cerner_mapping.py --resource Person
```

### Scenario 2: Too Many False Positives

Increase `--min-score` and shift weights to trusted components:

```bash
python recommend_cerner_mapping.py --resource Person \
  --min-score 0.2 \
  --weights "name=0.6,desc=0.2,type=0.1,dist=0.1"
```

### Scenario 3: Missing Good Matches

Lower `--min-score`, increase `--max-distance`, use embeddings:

```bash
python recommend_cerner_mapping.py --resource Person \
  --min-score 0.02 \
  --max-distance 3 \
  --embeddings
```

### Scenario 4: Slow Performance

Reduce computational cost:

```bash
python recommend_cerner_mapping.py --resource Person \
  --suggest-prefilter 40 \
  --top-k 3
```

(Skips embeddings, speeds up suggestion)

### Scenario 5: Prefer Tables With Complete Coverage

```bash
python recommend_cerner_mapping.py --resource Person \
  --table-weights "name=0.1,content=0.3,coverage=0.6" \
  --coverage-threshold 0.5
```

---

## Usage Examples

### Auto-Suggest Tables & Recommend Columns

```bash
python recommend_cerner_mapping.py --resource Patient
```

Output: Suggests 5 seed tables, fills `Source Fields` in `Patient_mapping.csv`

### Use Explicit Seed Tables

```bash
python recommend_cerner_mapping.py \
  --resource Patient \
  --tables PATIENT PERSON_PATIENT PERSON
```

### Print Suggestions Without Modifying CSV

```bash
python recommend_cerner_mapping.py \
  --resource Patient \
  --suggest-only
```

Output: Shows top 5 suggested tables with scores, exits

### Customize All Settings

```bash
python recommend_cerner_mapping.py \
  --resource Patient \
  --max-distance 3 \
  --top-k 8 \
  --min-score 0.1 \
  --weights "name=0.5,desc=0.3,type=0.1,dist=0.1" \
  --table-weights "name=0.3,content=0.5,coverage=0.2" \
  --coverage-threshold 0.4 \
  --suggest-prefilter 100 \
  --embeddings
```

---

## Performance Characteristics

### Time Complexity

| Phase | Operation | Complexity | Notes |
|-------|-----------|-----------|-------|
| Pre-filter | Name scoring all tables | O(T) | T = total tables (~1K) |
| Content | Score top tables × FHIR elements | O(K×M×C) | K=prefilter (80), M=FHIR elems, C=cols per table |
| TF-IDF | Vectorize texts | O(V×C×M) | V=vocabulary, C=cols, M=elements |
| Embeddings | Encode texts | O(M+C) | M=FHIR elems, C=total cols |
| BFS | Expand seed tables | O(T+E) | E=foreign key edges |
| Ranking | Score all columns | O(N×M) | N=candidate cols, M=FHIR elems |

### Memory Usage

- Columns matrix: ~10 KB per 1000 columns
- Description matrix: ~1 KB per element-column pair
- Embeddings: ~20 MB per model (cached)

### Typical Runtime (100 FHIR elements, ~10K columns)

- **TF-IDF (default):** 5-15 seconds
- **With embeddings:** 30-60 seconds
- **Table suggestion only:** 10-20 seconds

---

## Customization Points

### Adding New Token Aliases

Edit `TOKEN_ALIASES` dict (line 112):

```python
TOKEN_ALIASES: dict[str, set[str]] = {
    "cd": {"code"},
    "your_abbr": {"expanded", "forms"},
}
```

### Adding FHIR Synonyms

Edit `FHIR_SYNONYMS` dict (line 139):

```python
FHIR_SYNONYMS: dict[str, set[str]] = {
    "yourfield": {"cerner_term1", "cerner_term2"},
}
```

### Custom Type Rules

Edit `type_compat()` function (line 386):

```python
elif "your_type" in ftype and "your_cerner_type" in dtype:
    raw = 3.0
```

---

## Output Interpretation

### CSV Output Format

- **Cerner Source Fields**: `TABLE.COLUMN | TABLE.COLUMN | ...`
  - Multiple candidates separated by ` | ` (pipe)
  - Ordered by descending score
  - Top `--top-k` candidates above `--min-score`

### Score Breakdown

When using `--suggest-only`, see table scores:

```
  1. PERSON_PATIENT           score=0.658  name=0.82 content=0.68 coverage=0.45 cols=89
  2. PERSON                   score=0.542  name=0.71 content=0.52 coverage=0.30 cols=45
  3. PATIENT_ACCOUNT          score=0.428  name=0.55 content=0.41 coverage=0.20 cols=67
```

- **name**: Table name vs. resource name similarity [0, 1]
- **content**: Average best column scores for FHIR elements [0, 1]
- **coverage**: Fraction of FHIR elements with ≥ coverage_threshold score [0, 1]
- **cols**: Number of columns in table

---

## Troubleshooting

### No Suggestions Returned

1. **Check seed tables:** `--suggest-prefilter` might be too small
   ```bash
   --suggest-prefilter 100
   ```

2. **Check pre-filter ratio:** If catalog is large, increase prefilter:
   ```bash
   --suggest-prefilter $(python -c "print(max(5, int(TOTAL_TABLES * 0.1)))")
   ```

### Poor Quality Recommendations

1. **Try embeddings for semantic understanding:**
   ```bash
   --embeddings
   ```

2. **Adjust description weight:**
   ```bash
   --weights "name=0.3,desc=0.6,type=0.05,dist=0.05"
   ```

3. **Lower min-score to see more candidates:**
   ```bash
   --min-score 0.02
   ```

### Slow Performance

1. **Reduce table pre-filter:**
   ```bash
   --suggest-prefilter 40
   ```

2. **Disable embeddings:**
   (Remove `--embeddings` flag)

3. **Reduce max-distance:**
   ```bash
   --max-distance 1
   ```

### FK Expansion Includes Irrelevant Tables

1. **Reduce max-distance:**
   ```bash
   --max-distance 0  # Seed tables only
   ```

2. **Add more seed tables explicitly:**
   ```bash
   --tables TABLE1 TABLE2 TABLE3
   ```

---

## Architecture Decisions

### Why Hybrid Scoring?

- **Name alone:** Fails when Cerner columns are abbreviated
- **Description alone:** Requires good documentation (not always available)
- **Hybrid:** Combines strengths—abbreviation-aware name matching + semantic description understanding

### Why BFS for FK Expansion?

- Prioritizes nearby tables (distance-based scoring)
- Bounded by `max_distance` for quality control
- Standard in schema navigation

### Why Caching FHIR Definitions?

- hl7.org can be slow/unreliable
- Definitions rarely change
- Cache invalidation: manual (delete cache file)

### Why Token Aliases & Synonyms?

- Maps domain terminology (Cerner ↔ FHIR)
- Improves matching without retraining models
- Easily extensible for new domains

---

## References

- **RapidFuzz:** https://github.com/maxbachmann/RapidFuzz
- **scikit-learn TF-IDF:** https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- **Sentence Transformers:** https://www.sbert.net/
- **FHIR R4:** https://www.hl7.org/fhir/r4/
- **Jaccard Similarity:** https://en.wikipedia.org/wiki/Jaccard_index
- **Cosine Similarity:** https://en.wikipedia.org/wiki/Cosine_similarity
