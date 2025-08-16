## HomeMatch: Semantic Real Estate Matching

HomeMatch is a small, end-to-end example of building a retrieval-driven, generative AI experience for real estate discovery. It:

- Generates or loads mock listings
- Indexes them into two vectorstores (raw structured text vs. LLM-enhanced prose)
- Collects user preferences
- Performs semantic retrieval to surface the most relevant listings
- Optionally personalizes listing descriptions using the LLM

This project is intentionally compact and heavily documented to help you learn the moving parts and avoid common pitfalls we encountered while building it.

### Contents

- `HomeMatch.py`: Main pipeline (indexing, preference parsing, retrieval, optional augmentation)
- `GenerateListings.py`: Utility to synthesize `listings.json` with mock listings using an LLM
- `GenerateEnhancedListings.py`: Utility to precompute LLM-enhanced listing prose (more efficient; avoids repeated calls during the main run)
- `listings.json`: Generated dataset (created by `GenerateListings.py`)
- `semantic_enhanced_listings.txt`: Optional artifact created by `GenerateEnhancedListings.py` containing one enhanced line per listing

## Prerequisites

- Python 3.10+ (3.11/3.12 also fine)
- An OpenAI-compatible endpoint and API key
  - Environment variables: `OPENAI_API_KEY`, `OPENAI_API_BASE`

### Python dependencies

Install these packages into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install langchain-openai langchain-chroma chromadb python-dotenv tiktoken
```

If your editor shows unresolved import warnings, ensure it’s using this same virtual environment.

### Environment variables

The scripts read environment variables (and call `load_dotenv()`, so a `.env` file is supported). You can create a `.env` file like this:

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=your_key_here
OPENAI_API_BASE=https://api.openai.com/v1
EOF
```

Note: The scripts also set these variables in code for the Vocareum environment. Your env vars will override if set before running.

## How to run

### 1) Generate mock listings

This produces `listings.json` used by the main pipeline.

```bash
python GenerateListings.py
```

You can change the generated count by editing the `generate_listings(amount=...)` call at the bottom of `GenerateListings.py`.

### 2) (Optional, faster) Precompute semantic enhancements

Precompute the enhanced, human-friendly listing descriptions once to avoid repeating the LLM step inside `HomeMatch.py`:

```bash
python GenerateEnhancedListings.py
```

This writes `semantic_enhanced_listings.txt`, one enhanced description per listing (aligned to the order in `listings.json`). You can then adapt `HomeMatch.py` to read these lines and build the semantic vectorstore without calling the LLM at runtime.

### 3) Run the HomeMatch pipeline

```bash
python HomeMatch.py
```

You should see:

- Number of listings/documents
- A flattened “preferences” query built from the demo Q&A
- Two retrieval outputs:
  - From the raw index (listing JSON embedded as text)
  - From the semantic-enhanced index (LLM-generated prose embedded). If you precomputed enhancements, you can embed those instead for faster runs.
- Optional LLM “augmentation” that rewrites a listing to better match the user preferences

## How it works

### Dual indexes

- Raw vectorstore: embeds the full listing JSON (minus `id`) as the `page_content`. Good for structured text signals.
- Semantic vectorstore: embeds readable, LLM-enhanced descriptive prose. Good for “marketing” style similarity.

Both are stored in separate Chroma collections to keep results independent:

```python
raw_vectorstore = Chroma.from_documents(..., collection_name="raw_listings")
semantic_enhanced_vectorstore = Chroma.from_documents(..., collection_name="semantic_listings")
```

You can also persist to disk if desired by adding `persist_directory` per collection.

### Querying

We convert a structured preferences dict/Q&A into a single natural-language query string, then call similarity search:

```python
results = vectorstore.similarity_search(preferences_query, k=5)
```

You can also use:

- `similarity_search_with_score(query, k)` for scores
- `max_marginal_relevance_search(query, k, fetch_k)` for diverse results
- `filter={...}` for metadata filters (e.g., `{"neighborhood": "Sunnyvale"}`)

### Personalization

After retrieving candidates, we optionally rewrite the listing copy with the LLM to emphasize aspects that match user preferences. The prompt is careful to avoid changing factual details.

If you precomputed enhanced listings, you can still do a lightweight personalization pass at the end, or skip it for maximum speed.

## Troubleshooting and gotchas (from our build)

- Document type for indexing

  - Symptom: `AttributeError: 'dict' object has no attribute 'page_content'`
  - Cause: Passing a list of dicts into `Chroma.from_documents(...)`.
  - Fix: Create `langchain_core.documents.Document` objects with `page_content` (text to embed) and optional `metadata`.

- Query type for retrieval

  - Symptom: `TypeError: argument 'text': 'dict' object cannot be converted to 'PyString'`
  - Cause: Passing a dict to `similarity_search(...)`.
  - Fix: Always pass a string as the query. Flatten your preferences into a single text query.

- No `semantic_search` method

  - Symptom: `AttributeError: 'Chroma' object has no attribute 'semantic_search'`
  - Fix: Use `similarity_search`, `similarity_search_with_score`, or MMR methods on the vectorstore.

- Two indexes returning identical results

  - Symptom: Same results from `raw_vectorstore` and `semantic_enhanced_vectorstore`.
  - Causes:
    - Both used the default Chroma collection (`"langchain"`), so they wrote to/read from the same collection.
    - The embedded texts were very similar across both indexes.
  - Fixes:
    - Set distinct `collection_name`s (and optionally different `persist_directory`s).
    - Ensure each index embeds meaningfully different text (e.g., raw JSON vs. LLM-enhanced prose only).
    - If you precompute enhanced descriptions, verify you are embedding the precomputed content rather than regenerating the same text on the fly.

- Prompt templating

  - Symptom: Confusing string formatting or wrong values inserted into prompts.
  - Gotcha: Don’t mix f-strings and `.format()` on the same template. Choose one.
  - Tip: When inserting a listing dict into a prompt, `json.dumps(listing)` ensures a clean string.

- Code fences in LLM output

  - Symptom: JSON parsing fails when responses include ```json fences.
  - Fix: Strip code fences before `json.loads(...)`.

- Python list of strings separated by commas

  - Symptom: Missing commas cause implicit concatenation of adjacent string literals.
  - Fix: Ensure commas between each string item in lists.

- Import resolution warnings
  - Symptom: Editor shows `Import "langchain_*" could not be resolved`.
  - Cause: Editor not using your virtual environment.
  - Fix: Activate the venv and configure your editor to use it.

## Extending the project

- Metadata filters: store numeric fields (e.g., `price_num`, `bedrooms_num`) in `metadata` and use `filter={...}` at query time.
- Reranking: fetch more candidates with `similarity_search_with_score` or MMR, then apply a custom scorer (price/beds/amenities alignment) before presenting results.
- Field-focused indexing: embed only specific fields for targeted similarity; create multiple docs per listing (e.g., one per field) and filter by a `field` tag.

## Efficiency tips (based on our experience)

- **Precompute enhancements**: Use `GenerateEnhancedListings.py` to avoid repeated LLM calls every time you run the pipeline.
- **Separate collections**: Give unique `collection_name`s (and optional `persist_directory`) per index to avoid accidental reuse.
- **Batch runs**: Generate and index in batches (e.g., 50–200 listings) to reduce per-run overhead.
- **Cache or store embeddings**: Persist Chroma collections to disk to reuse embeddings across runs.

## Example commands

```bash
# Create env and install deps
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install langchain-openai langchain-chroma chromadb python-dotenv tiktoken

# Configure environment (or use a .env file)
export OPENAI_API_KEY=your_key_here
export OPENAI_API_BASE=https://api.openai.com/v1

# Generate data and run pipeline
python GenerateListings.py
# Optional: precompute enhanced descriptions to speed up the main run
python GenerateEnhancedListings.py
python HomeMatch.py
```

## Notes

- The code includes inline comments and docstrings for clarity. Read through `HomeMatch.py` to see where documents are created, vectorstores are built, and search is executed.
- API usage may incur cost. Use modest `k` values and sample sizes while testing.
