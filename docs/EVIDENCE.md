# Evidence map

| Report evidence | Repository location |
|---|---|
| Data cleaning and overlap audits | `multimodal-fake-news-detection/results/metrics/fakeddit_*.json` and related CSVs |
| Linear ablation | `multimodal-fake-news-detection/results/metrics/linear_fusion_ablation/` |
| Ordinary late-fusion selection | `multimodal-fake-news-detection/results/metrics/late_probability_fusion/` |
| Frozen ordinary holdout | `multimodal-fake-news-detection/results/metrics/final_holdout/` |
| Final predictions and metrics | `multimodal-fake-news-detection/results/metrics/final_evaluation/` |
| Paired tests and calibration | `multimodal-fake-news-detection/results/metrics/statistical_comparison/` and `calibration_analysis/`; final tests also in final results |
| SHAP, Grad-CAM and selected cases | `multimodal-fake-news-detection/results/shap/`, `gradcam/`, `case_studies/` |
| Later methodological diagnostics | `v2/outputs/` |
| Source-disjoint model selection, test and statistical evidence | `v2/experiments/source_disjoint/` |

The report review's evidence package called the first directory `original/`; in this repository it is `multimodal-fake-news-detection/`. The `v2/` paths are unchanged. Use this mapping if linking from Appendix E.

`evidence_manifest.csv` fingerprints copied results, diagnostic outputs and manifests. Historical files are not rewritten to make their workstation paths look current. Figures and attribution examples are selected illustrations, not independent proof of factual reasoning.

## Consolidated notebook

[The complete project notebook](../Notebook.ipynb) contains readable implementations and a frozen copy of the evidence above. Its displayed verification output comes from recomputing the saved predictions, not retraining. The notebook's earlier local variants are provenance copies; use the canonical repository scripts for reproduction.
