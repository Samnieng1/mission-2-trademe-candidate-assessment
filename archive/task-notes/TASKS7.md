Add a new automatic fuzzy evidence matching metric without removing or
changing the existing evidence_match_accuracy calculation.

Rename the current metric in the output and UI to:

evidence_exact_match_accuracy

Add a new metric:

evidence_fuzzy_match_accuracy

Do not add manual review, embeddings, another LLM, a vector database or any
external API.

1. Exact evidence metric

Preserve the existing exact evidence comparison behaviour.

Expose it as:

evidence_exact_match_accuracy

This metric should remain available so existing experiment results can still
be interpreted.

2. Fuzzy evidence metric

Verify each model-produced candidate_evidence against the uploaded CV text.

Do not compare candidate_evidence with benchmark source_evidence, because
source_evidence refers to the job advertisement.

3. Text normalisation

Create a shared helper that:

- converts text to lowercase;
- removes punctuation;
- collapses repeated whitespace;
- trims whitespace;
- safely handles null values;
- joins list-valued evidence into one string.

4. Evidence comparison

For each requirement match whose status is:

- matched;
- partially_matched;
- uncertain;

perform the following:

a. Direct containment check

If the normalised candidate evidence appears in the normalised CV, treat it as
supported with similarity 1.0.

b. Fuzzy comparison

If direct containment fails, split the CV into meaningful segments:

- paragraphs;
- bullet points;
- sentences.

Use Python's standard-library difflib.SequenceMatcher to compare the
normalised candidate evidence with each CV segment.

Keep the highest similarity score.

5. Threshold

Add a configurable setting:

EVIDENCE_FUZZY_THRESHOLD=0.70

If the highest similarity is greater than or equal to the threshold, treat the
candidate evidence as supported.

6. Metric calculation

For evidence_fuzzy_match_accuracy:

- supported evidence = 1.0;
- unsupported or missing evidence = 0.0.

Include only requirement matches that claim evidence:

- matched;
- partially_matched;
- uncertain.

Exclude not_matched requirements with no candidate evidence.

If no requirement claims evidence, return null rather than 0.0.

Formula:

evidence_fuzzy_match_accuracy =
supported claimed evidence items /
total claimed evidence items

7. Add an optional aggregate similarity metric

Also calculate:

mean_evidence_similarity

This should be the mean highest similarity across the evidence items included
in the fuzzy calculation.

Do not confuse this with accuracy.

8. Debug details

For each requirement, retain:

- requirement_id;
- candidate_evidence;
- direct_match;
- highest_similarity;
- matched_cv_segment;
- fuzzy_supported.

Display this only in a Streamlit debug expander.

9. Aggregates

For repeated runs, aggregate:

- evidence_exact_match_accuracy__mean;
- evidence_exact_match_accuracy__stdev;
- evidence_fuzzy_match_accuracy__mean;
- evidence_fuzzy_match_accuracy__stdev;
- mean_evidence_similarity__mean;
- mean_evidence_similarity__stdev.

Ensure only numeric scalar values are passed to statistics functions.

10. UI

Display both metrics with clear labels:

Exact evidence agreement
Fuzzy evidence support accuracy
Mean evidence similarity

Add helper text:

"Exact evidence agreement requires near-verbatim wording."

"Fuzzy evidence support accuracy checks whether the model evidence is
sufficiently similar to text in the uploaded CV."

11. Tests

Add tests for:

- exact quotation passes both metrics;
- punctuation and capitalisation differences pass fuzzy matching;
- close paraphrase passes fuzzy matching but fails exact matching;
- unrelated evidence fails both;
- null evidence for matched status counts as unsupported;
- null evidence for not_matched status is excluded;
- list-valued evidence is normalised;
- all seven supported evidence items produce fuzzy accuracy 1.0;
- no claimed evidence returns null;
- similarity values are flat numeric values;
- repeated-run aggregation handles null values safely.

12. Backward compatibility

If older saved result files contain only evidence_match_accuracy, treat that
field as evidence_exact_match_accuracy when loading them.

Do not overwrite old result files.
A useful final display would be:
Exact evidence agreement: 0.00
Fuzzy evidence support accuracy: 1.00
Mean evidence similarity: 0.86