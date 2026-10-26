# Synthetic Retrieval Evaluation (K=5)

This benchmark uses synthetic data and a lexical token-overlap retrieval
baseline. It does not measure the application's Sentence Transformers
embeddings or FAISS retrieval path, and its results do not establish real-world
accuracy or performance.

- Questions: 28
- Answerable questions: 24
- No-ground-truth questions: 4
- Missed-relevance cases: 1
- Precision@5: 0.1917
- Recall@5: 0.9583
- MRR: 0.8833
- Mean retrieval latency: 0.050 ms

Metrics are averaged over answerable questions only. No-ground-truth questions have undefined per-query precision, recall, and reciprocal rank and are reported separately, never counted as successful hits.

## Missed relevance and no-evidence cases

Similarity scores rank retrieved chunks; they do not establish factual correctness.

| Case | Question | Retrieved chunk IDs (scores) | Expected relevant IDs |
|---|---|---|---|
| Missed relevance | What is the penalty for a late library book? | museum_admission (0.3750), library_hours (0.2500), loan_period (0.2500), museum_monday (0.2500), parking (0.2500) | late_fee |
| No evidence | What is the library's phone number? | library_hours (0.3333), museum_admission (0.3333), museum_monday (0.3333), parking (0.3333), accessible_entry (0.1667) | none (unanswerable) |
| No evidence | Does the cafe offer vegan pastries? | cafe_hours (0.3333), accessible_entry (0.1667), bus_route (0.1667), guest_wifi (0.1667), library_hours (0.1667) | none (unanswerable) |
| No evidence | What is the museum's postal address? | museum_admission (0.5000), museum_monday (0.5000), accessible_entry (0.3333), parking (0.3333), bus_route (0.1667) | none (unanswerable) |
| No evidence | Can visitors borrow museum artifacts? | guest_wifi (0.4000), accessible_entry (0.2000), museum_admission (0.2000), museum_monday (0.2000), renewals (0.2000) | none (unanswerable) |

Retrieval latency is a wall-clock measurement for this run and may vary between runs and machines.
