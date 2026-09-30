# Complete project notebook

Open [Notebook.ipynb](../Notebook.ipynb) in JupyterLab or VS Code with a Python notebook kernel. Download the file if the GitHub preview is unavailable; the notebook contains embedded evidence and is approximately 10 MB.

## Inspection environment

Use Python 3.12 in a separate virtual environment:

```sh
python -m venv .venv
# Activate that environment using the command for your operating system.
python -m pip install -r requirements-notebook.txt
python -m jupyterlab
```

Open the notebook and select that environment's kernel, then restart the kernel and Run All. The notebook stores the source, checks its hashes and syntax, extracts evidence to a temporary directory and verifies saved metrics. It does not download data, train models or overwrite the repository's evidence. Successful verification output is already saved in the delivered notebook.

## Why source cells use a magic

Cells starting with `%%project_source` register the source under its original filename. They do not execute the training script. This preserves script entry points and independent namespaces while making all the source visible. The registration helper must run first; running an isolated source cell before it will fail with an unknown-magic error. Restart and Run All after editing source. The integrity check will flag deliberate source changes until the recorded hashes are updated; retain an unchanged archival notebook and use an exported working copy for new experiments.

## What is included

- Data preparation, image downloads, integrity checks, placeholder investigation and clean-manifest freezing.
- RoBERTa and ResNet-50 training, feature extraction, learned and linear fusion, probability export and validation-only late fusion.
- Error analysis, calibration, missing-modality checks, SHAP, Grad-CAM and case studies.
- Final holdout preparation/evaluation, methodology diagnostics and source-disjoint retraining.
- Verification code, configuration snapshots, dependency records and historical local variants.
- Saved prediction/evidence artifacts needed to recompute the reported results.

## Reproduce experiments

The last section exports a **new** directory and supplies explicit launch helpers. Restore the external dataset, images, checkpoint/feature assets and appropriate ML dependencies before selecting a stage. The original scripts have individual path constants and command-line interfaces; this notebook does not turn them into a single automatic training pipeline. Historical variants retain workstation paths and are blocked by the launch helper.

The source-disjoint stages run in order: audit, smoke, train-roberta, train-resnet, lock, test. Keep the same experiment directory and choose weights/checkpoints from validation. See [reproduction guidance](REPRODUCIBILITY.md) and [known limitations](PACKAGING.md).

## Snapshot boundaries

Embedded documentation records the notebook's assembly snapshot; repository-level documentation may be newer. Scientific results and canonical experiment implementations are checked for consistency by `python tools/verify_notebook.py`. Raw corpora, trained weights, credentials, virtual environments and document/video editing utilities are not bundled.
