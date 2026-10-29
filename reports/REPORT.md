
# Contract Intelligence Assistant — Evaluation Report

## Summary

- **Final configuration:** Hybrid word-level and character-level TF-IDF, relevance threshold = 0.08. The test-results file does not record the chunk size, so the exact final chunk configuration must be confirmed.
- **Test queries evaluated:** 54
- **Character precision@5:** 0.94%
- **Character recall@5:** 22.44%
- **Hit rate@5:** 24.07%
- **Unanswerable queries in test:** 8
- **Correct abstentions:** 4 out of 8
- **Correct abstention rate:** 50%
- **Number of test runs:** 1, as recorded for this final evaluation

## System Design

The Contract Intelligence Assistant retrieves potentially relevant passages from legal contracts using TF-IDF cosine similarity. It combines word-level and character-level features, divides contracts into overlapping character-based chunks, and preserves source character offsets.

A configurable minimum relevance threshold allows the system to abstain when no retrieved passage reaches the threshold. A keyword-overlap retrieval method is also included for comparison.

The project provides a Streamlit web interface and a command-line interface. It does not use an LLM in the baseline, keeping retrieval local and relatively inexpensive.

## Evaluation Setup

The evaluation uses the CUAD subset from LegalBench-RAG. The assignment specifies 194 queries selected using its prescribed sampling method and split using the SHA-256 rule.

The test evaluation file contains 54 queries. Metrics are reported separately for the training experiments and held-out test set.

Ground-truth answer spans are used for evaluation, not as retrieval inputs. The unanswerable questions are evaluated separately using the assignment's split procedure.

The final test configuration used TF-IDF retrieval with a minimum relevance threshold of 0.08. Test results were not used to tune parameters.

## Results

### Baseline Training Results

The baseline used a chunk size of 1,200 characters, overlap of 200 characters, TF-IDF retrieval, and a threshold of 0.08.

| Metric | Result |
|---|---:|
| Training queries | 140 |
| Character precision@5 | 1.51% |
| Character recall@5 | 13.84% |
| Hit rate@5 | 18.57% |
| Unanswerable training queries | 22 |
| Correct abstentions | 3 |
| Unanswerable abstention rate | 13.64% |

### Training Experiment Results

The improvement script evaluated the baseline and five alternative configurations on training data.

| Configuration | Chunk size | Overlap | Method | Threshold | Precision@5 | Recall@5 | Hit rate@5 | Abstention rate |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Baseline | 1200 | 200 | TF-IDF | 0.08 | 1.51% | 13.84% | 18.57% | 13.64% |
| Smaller chunks | 800 | 150 | TF-IDF | 0.08 | 1.21% | 8.99% | 12.86% | 4.55% |
| Larger chunks | 1800 | 250 | TF-IDF | 0.08 | 1.83% | 25.16% | 28.57% | 36.36% |
| More overlap | 1200 | 400 | TF-IDF | 0.08 | 1.16% | 12.14% | 17.86% | 13.64% |
| Keyword baseline | 1200 | 200 | Keyword | 0.08 | 1.18% | 11.55% | 17.86% | 0.00% |
| Higher threshold | 1200 | 200 | TF-IDF | 0.15 | 0.88% | 5.70% | 9.29% | 90.91% |

The larger-chunk experiment achieved the highest training hit rate@5 and character recall@5 among these configurations. However, its abstention rate was 36.36%, so the result must be considered alongside abstention behavior rather than retrieval metrics alone.

The higher-threshold experiment improved the rate of correct abstention but substantially reduced retrieval performance. This illustrates the trade-off between returning relevant passages and avoiding unsupported results.

The final test evaluation below used a threshold of 0.08. Confirm the chunk size recorded in the actual configuration before claiming that the test results correspond to the larger-chunk experiment.

### Final Test Results

The held-out test evaluation returned the following results for 54 queries.

| Metric | @1 | @3 | @5 | @10 |
|---|---:|---:|---:|---:|
| Hit rate | 5.56% | 16.67% | 24.07% | 42.59% |
| Character precision | 0.45% | 0.99% | 0.94% | 1.43% |
| Character recall | 4.41% | 15.03% | 22.44% | 33.26% |

Of the eight unanswerable test queries, the system correctly abstained on four, resulting in an unanswerable abstention rate of 50%.

These results indicate that the system retrieves relevant passages for some queries but still struggles to rank the exact answer clause highly. The low character precision also indicates that retrieved passages contain substantial text outside the gold answer spans.

## Iteration History

The improvement loop explored smaller chunks, larger chunks, increased overlap, keyword retrieval, and a higher relevance threshold.

- **Smaller chunks:** Reduced retrieval hit rate and character recall compared with the baseline.
- **Larger chunks:** Improved hit rate and recall on training data, but did not eliminate irrelevant passage retrieval.
- **More overlap:** Did not outperform the baseline on hit rate or recall.
- **Keyword baseline:** Performed below the TF-IDF baseline on hit rate and recall.
- **Higher threshold:** Increased correct abstention but reduced hit rate and recall substantially.

The experiment log is saved in `reports/experiment_log.csv`. These results support selecting larger chunks as the strongest retrieval configuration among the tested training experiments, subject to confirming the final configuration and evaluating abstention guardrails.

## Error Analysis

The test evaluation identified failures in which the retriever returned passages from the correct contract but failed to cover the gold answer span sufficiently.

Representative failure cases include:

1. **Termination for convenience:** The retriever returned the cooperation agreement's title page instead of the termination clause.
2. **Joint intellectual property ownership:** The retriever returned the service agreement's introductory section rather than the intellectual property assignment clause.
3. **Uncapped liability:** The retriever returned a sponsorship schedule instead of the liability limitation provision.
4. **Non-compete clause:** The retriever returned the strategic alliance agreement's introduction rather than the non-competition provisions.
5. **Anti-assignment clause:** The retriever returned the agreement's introduction instead of its assignment restriction.
6. **Revenue or profit sharing:** The retriever returned the contract introduction rather than the profit-sharing provision.
7. **Post-termination obligations:** The retriever returned the agreement's opening section instead of the relevant obligations provision.
8. **Liability cap:** The retriever returned introductory text rather than the liability limitation clause.
9. **Exclusivity agreement:** The retriever returned the transportation agreement's introduction instead of the exclusivity-related provision.
10. **Insurance requirements:** The retriever returned the contract introduction rather than the detailed insurance requirements.

These examples show a recurring failure mode: lexical similarity often identifies the correct document, but the highest-ranked passage may be an introduction, table of contents, or signature section rather than the answer clause.

The examples above summarize failures present in the evaluation output. The corresponding queries and gold spans can be inspected in the generated test evaluation JSON.

## LLM Use

No LLM is used in the baseline retrieval process. The system uses TF-IDF vectorization, cosine similarity, and keyword overlap.

This design keeps the implementation local, inexpensive, and reproducible. An LLM could be explored later only if it demonstrably improves evaluation metrics without compromising reproducibility or abstention behavior.

## Limitations and Next Steps

- TF-IDF measures lexical similarity rather than legal reasoning.
- Low relevance scores do not prove that a contract contains no answer.
- The relevance threshold requires calibration on training data.
- Character-based chunks can split clauses or combine unrelated sections.
- The system does not resolve cross-references or defined legal terms.
- Correct document retrieval does not guarantee correct passage retrieval.
- Abstention performance remains limited on the held-out test set.
- Character precision and recall indicate that exact answer-span retrieval requires further improvement.

Future improvements should focus on clause-aware chunking, better passage ranking, and stronger abstention calibration using training data only.

Before use on real agreements, the system would also require access controls, security review, legal validation, and stronger reliability testing.

## Conclusion

The project implements a simple contract passage retrieval assistant with a web interface, command-line interface, ranked passages, source character offsets, and configurable abstention.

Training experiments show that larger chunks improve hit rate and character recall compared with the baseline. However, the final test results show that retrieval quality and abstention still need improvement. The system should therefore be treated as an experimental retrieval baseline rather than a production-ready legal assistant.
