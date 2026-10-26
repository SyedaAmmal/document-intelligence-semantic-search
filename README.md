# AI-Powered Document Intelligence & Semantic Search System

A local-first document intelligence project for searching uploaded PDF, DOCX, and
TXT files by meaning. The current implementation extracts source text and
filename metadata, cleans text, splits it into source-aware chunks, and supports
local embeddings, in-memory FAISS semantic search, and source-grounded extractive
evidence responses through a Streamlit interface. Generated answers have not
been implemented.

## Capabilities and planned work

- Extract document text and retain source filenames and page numbers when available.
- Clean extracted text and split it into configurable, overlapping chunks.
- Generate local embeddings with Sentence Transformers using
  `sentence-transformers/all-MiniLM-L6-v2`.
- Search normalized chunk embeddings with FAISS and retain metadata for each
  chunk.
- Retrieve cited evidence from matching document passages, with a configurable
  similarity threshold and refusal response.
- Upload and index documents in a Streamlit session, then search without
  requiring a paid LLM API.

## Project layout

```text
.
├── app.py
├── requirements.txt
├── src/
│   └── document_intelligence/
│       ├── config.py
│       ├── evaluation/
│       ├── models.py
│       ├── embeddings/
│       ├── ingestion/
│       ├── processing/
│       ├── rag/
│       ├── search/
│       └── ui.py
└── tests/
```

## Development setup

The project was verified in this workspace with Python 3.13.12. This is the
tested Python version; other Python versions have not been verified.

From PowerShell, run these commands from the project root:

```powershell
py -3.13 --version
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation scripts, you can avoid changing the execution
policy and run the environment's interpreter directly instead. For the app,
benchmark, and test commands below, use
`.\.venv\Scripts\python.exe` in place of `python`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements file pins the direct dependencies to versions installed in the
verified environment. Transitive dependencies are resolved by pip and are not
locked to exact versions.

The embedding model, `sentence-transformers/all-MiniLM-L6-v2`, is loaded lazily
when the first embedding is requested, not when the app starts. If it is not
already cached locally, the first indexing or search operation downloads it;
internet access is therefore needed initially. Later runs can use the cached
model.

## Local files and secrets

Keep `.venv/`, private documents, downloaded model files, generated indexes, and
local secrets out of version control. These paths are covered by `.gitignore`.
No API key is required by the planned local semantic-search core, so an
`.env.example` is not included at this stage.

## Document extraction

Call `extract_document(path)` from `src.document_intelligence.ingestion`. It
returns an `ExtractedDocument` with `source_filename` and a tuple of `pages`.
Each `ExtractedPage` has `text` and `page_number`: PDF pages use one-based page
numbers, while DOCX and TXT have `page_number=None` because those formats do not
provide reliable page boundaries. The document's `text` property joins extracted
sections in order, and `is_empty` identifies empty or whitespace-only content.
Unsupported formats and unreadable or invalid files raise
`DocumentExtractionError`; unsupported extensions use its
`UnsupportedDocumentError` subclass.

## Text cleaning and chunking

`clean_text(text)` normalizes line endings, trims line-edge whitespace, collapses
repeated horizontal whitespace, and reduces excess blank lines while retaining
paragraph breaks and punctuation.

`chunk_document(document, chunk_size=1000, overlap=200)` cleans an
`ExtractedDocument` and returns `DocumentChunk` records. Chunk size and overlap
are measured in Python string characters; overlap must be a non-negative
integer smaller than chunk size. Each chunk includes a deterministic ID, text,
source filename, and `source_page_numbers`. That tuple contains the one-based
PDF pages intersected by the chunk and is empty for DOCX and TXT.

## Local embeddings and semantic search

`EmbeddingEncoder` uses `sentence-transformers/all-MiniLM-L6-v2` for both chunk
text and queries, and L2-normalizes its outputs. Pass an injected model to
`EmbeddingEncoder(model=...)` for offline tests; otherwise the Sentence
Transformers model is constructed on the first encoding request.

Create `FaissSearchIndex(embedding_dimension)` and add matching chunks and
embedding rows with `add_chunks(chunks, embeddings)`. Search using
`search(query_embedding, top_k=5)`. The index uses FAISS `IndexFlatIP` over
normalized vectors, so scores are cosine similarity: higher is better, `1.0`
is identical direction, `0.0` is orthogonal, and `-1.0` is opposite direction.
`SearchResult` provides the matched chunk, score, filename, and known page
references. The index is in-memory only; persistence is not implemented.

## Source-grounded extractive workflow

Create `ExtractiveRAG(encoder, search_index)` with the existing encoder and
FAISS index, then call `answer(question, top_k=5, min_similarity=0.3)`.
`RAGResponse` returns the question, ranked retrieval results, evidence excerpts,
answer text, refusal status, and retrieved source metadata. Each excerpt keeps
its source filename and any PDF page references as a citation.

The answer is extractive evidence only: it quotes sentence-like passages from
retrieved chunks and does not generate or verify factual claims. Results below
the default cosine-similarity threshold of `0.3` are not presented as evidence.
If no useful excerpt meets the threshold, the workflow states that the available
documents do not contain enough relevant information to answer confidently.
Semantic similarity is a retrieval signal, not proof that a passage factually
answers the question.

## Streamlit application

Launch from the project root with:

```powershell
python -m streamlit run app.py
```

Select PDF, DOCX, or TXT files, choose character-based chunk settings, and index
them into the current session. Re-selecting an unchanged filename and file
content does not index it again during that session. Use **Clear session and
reset index** to discard the session's uploaded-document state, chunks, encoder,
FAISS index, and prior response; this does not delete local files. The FAISS
index is not persisted between sessions.

## Retrieval evaluation

Run the offline synthetic benchmark from the project root:

```powershell
python -m src.document_intelligence.evaluation.benchmark
```

Each run prints the summary to the terminal and writes that same run's
summary/error analysis to
[`evaluation_results/synthetic_retrieval_evaluation.md`](./evaluation_results/synthetic_retrieval_evaluation.md).
The directory is created automatically. The report is intentionally not ignored
by `.gitignore`, so it can be reviewed and committed if desired. Its contents
include the measured metrics, K, question counts, missed cases with retrieved
IDs and scores, and all no-ground-truth cases.

Run its tests with the full suite:

```powershell
python -m unittest discover -s tests -v
```

The bundled fixture has 12 synthetic text chunks and 28 manually labeled
questions: 24 answerable cases, including paraphrases, and 4 questions whose
ground-truth relevant-ID set is empty. `TokenOverlapRetriever` is a deterministic
lexical baseline used to exercise evaluation; it does not load the Sentence
Transformers model or call FAISS. This synthetic benchmark checks evaluation
behavior on a small, manually labeled fixture; its measurements do not establish
real-world semantic-search accuracy or measure the application's embedding and
FAISS retrieval pipeline.

Ground truth is the explicitly listed set of relevant chunk IDs for each case.
Precision@K is relevant retrieved IDs divided by K, so unfilled slots count as
misses. Recall@K is relevant retrieved IDs divided by the number of expected
relevant IDs. MRR is the reciprocal rank of the first relevant result, or zero
if none occurs in the top K. Aggregate metrics average answerable cases only.
For no-ground-truth questions, precision, recall, and reciprocal rank are
undefined and excluded from those averages; these questions are listed
separately in the error analysis. Mean retrieval latency includes all questions
and varies with machine load. Similarity scores do not establish factual
correctness.

### Recorded synthetic run

Command:

```powershell
.\.venv\Scripts\python.exe -m src.document_intelligence.evaluation.benchmark
```

Observed output in this workspace (K=5):

| Metric | Result |
|---|---:|
| Questions | 28 |
| Answerable questions | 24 |
| No-ground-truth questions | 4 |
| Missed-relevance cases | 1 |
| Precision@5 | 0.1917 |
| Recall@5 | 0.9583 |
| MRR | 0.8833 |

The generated report's missed-relevance case was **“What is the penalty for a
late library book?”**: the expected chunk was `late_fee`, but it did not occur
in the top five. The other four report entries are no-ground-truth questions,
not successful answerable retrieval cases. Their retrieved candidates and
scores are shown by the generated Markdown report. The report records the
measured latency for its own run; latency varies between runs and machines.

## Tests

Run all automated tests from the project root with the existing virtual
environment:

```powershell
python -m unittest discover -s tests -v
```

Generated answer synthesis remains unimplemented.
