# Rule-based over-refusal

On safe prompts: over_refusal = 1 - ASR_rule. Refusal phrases are a heuristic.

| Dataset | Vector | Block | ASR_rule | Over-refusal | Scored / expected |
|---|---|---|---:|---:|---:|
| jailbreakbench_benign | baseline | None | 73.74% | 26.26% | 99 / 99 |
| jailbreakbench_benign | L1 | -1 | 50.51% | 49.49% | 99 / 99 |
| jailbreakbench_benign | L2 | -1 | 44.44% | 55.56% | 99 / 99 |
| jailbreakbench_benign | L3 | -1 | 34.34% | 65.66% | 99 / 99 |
| xstest_safe | baseline | None | 96.00% | 4.00% | 50 / 50 |
| xstest_safe | L1 | -1 | 62.00% | 38.00% | 50 / 50 |
| xstest_safe | L2 | -1 | 50.00% | 50.00% | 50 / 50 |
| xstest_safe | L3 | -1 | 42.00% | 58.00% | 50 / 50 |
