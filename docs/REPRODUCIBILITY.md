# Reproduction and environment

## Inspect results without training

Run `python tools/verify_saved_results.py` from the repository root. This requires no third-party packages and does not change any result files. Open the CSV/JSON results listed in EVIDENCE.md for detailed evaluation.

## Consolidated notebook

See [the notebook guide](NOTEBOOK.md). Its source-registration cells and embedded evidence checks can run without the external dataset or GPU. The experiment code is displayed in full but executed only through explicit reproduction steps. This is different from the original `v2/methodology_gap_analysis_v2.ipynb`, whose code cells run analyses and write outputs when executed.

`python tools/verify_notebook.py` checks notebook/source/evidence consistency using only the standard library.

## Full experiment environment

The original `requirements.txt` and `requirements-desktop.txt` are preserved historical environment snapshots, not independently tested cross-platform installation specifications. The desktop snapshot omits PyTorch packages. The source-disjoint experiment configuration records Python 3.12.3, torch 2.13.0+cu126, torchvision 0.28.0+cu126 and CUDA 12.6 on an NVIDIA GeForce RTX 2070 SUPER. Select a mutually compatible PyTorch/torchvision build for your hardware before installing the other dependencies. Clean-environment installation and GPU retraining were not performed during repository packaging.

Create your own virtual environment; do not commit it. Training needs the Fakeddit release, processed manifests, downloaded images and pretrained model weights. Reproducing exact inference also needs the saved trained checkpoints. These assets are excluded from Git. Obtain data from its original source and arrange separate access to trained weights where needed.

Original scripts use paths relative to `multimodal-fake-news-detection/`. The sequence is preparation/audit, development subset and image cleaning, unimodal training, feature/probability extraction, fusion comparisons, holdout construction/freezing and final evaluation. Inspect script constants before execution; these scripts are experiment entry points, not a uniform command-line application. Many write to the existing results directory, so reproduce in a separate working copy.

## Source-disjoint runner

`v2/experiments/source_disjoint/run_source_disjoint.py` exposes these stages in order:

```text
audit → smoke → train-roberta → train-resnet → lock → test
```

Example from the repository root, after restoring data and installing dependencies:

```sh
python v2/experiments/source_disjoint/run_source_disjoint.py audit --experiment-root runs/source-disjoint
python v2/experiments/source_disjoint/run_source_disjoint.py smoke --experiment-root runs/source-disjoint
```

Use the same new experiment root for later stages. `lock` selects fusion on validation; `test` consumes the locked settings. Never tune from the test scores. `FAKEDDIT_PROJECT_ROOT` optionally points the runner at a different original-project data directory. The runner resolves old image paths beneath that directory without changing archived manifests.

Launch the methodology notebook with the working directory set to the repository root or `v2/`. It needs the omitted raw and processed data. Some historical diagnostic records/manifests retain original workstation paths; they are provenance, not portable asset downloads. Inspect and resolve those references in a separate copy when reproducing notebook diagnostics. Do not rewrite archived fingerprints and then claim the original freeze still applies.

The chronological manifest alone is not evidence of a chronological model evaluation.
