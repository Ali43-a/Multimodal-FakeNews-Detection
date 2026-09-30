# Original experiment pipeline

This folder contains the original Fakeddit development experiments and ordinary final holdout evaluation. See the [repository overview](../README.md) and [reproduction instructions](../docs/REPRODUCIBILITY.md).

Run its scripts from this folder. Saved results are under `results/metrics/`; the principal final files are `final_evaluation/final_results.json` and `final_evaluation/final_predictions.csv`. Development results are exploratory and distinct from the 5,000-case locked holdout.

The `src/fake_news` directories are package scaffolding; the implemented pipeline is in `scripts/`. The existing environment tests only check basic PyTorch availability. The root verification tool separately checks the saved scientific results.
