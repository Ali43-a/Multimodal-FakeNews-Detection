# Multimodal fake-news detection

MSc Data Science project by **Ali Abdulzahra**, supervised by **Professor Jialie Shen**.

This repository accompanies the dissertation and compares RoBERTa, ResNet-50, feature fusion, linear probes and late probability fusion on Fakeddit. It includes the original experiments and later source-disjoint retraining. The task predicts dataset labels; it is not an independent fact-checking system.

## Main finding

Late fusion improves the ordinary locked holdout result, but its advantage does not transfer to the source-disjoint experiment. The latter selects zero text weight on validation, making fusion equal to ResNet-50.

| Evaluation | Cases | RoBERTa macro F1 | ResNet-50 macro F1 | Late fusion macro F1 | Text weight |
|---|---:|---:|---:|---:|---:|
| Ordinary locked holdout | 5,000 | 0.8000 | 0.7275 | 0.8188 | 0.45 |
| Source-disjoint test | 2,739 | 0.7596 | 0.6771 | 0.6771 | 0.00 |

Weights and checkpoints were selected using the corresponding validation partition. These designs use different training/test populations and models, so their absolute score difference is not an isolated causal effect of source separation. Class 1 is true-labelled content; class 0 is fake/misleading content. PR AUC in the saved metrics denotes average precision.

## Start here

Start with the [complete project notebook](Notebook.ipynb). It combines 42 Python source files, the methodology notebook cells, explanations, historical variants and embedded saved evidence. **Run All performs inspection and saved-result verification; it does not launch training or downloads.** See the [notebook guide](docs/NOTEBOOK.md) for setup, source-cell behaviour and explicit reproduction steps.

The accompanying [video presentation slides](presentation/Ali%20Abdulzahra%20Project%20Presentation.pptx) cover the methods, code demonstration and findings. The narration script is kept separately from this repository.

From this repository root, with Python 3.12 or later:

```sh
python tools/verify_saved_results.py
python tools/verify_notebook.py
```

This read-only check uses only the Python standard library. It recomputes accuracy, macro F1 and confusion matrices from both sets of saved test predictions, checks IDs and fusion arithmetic, and verifies the fingerprints of packaged results. It needs no GPU, checkpoints or downloaded images. The second command checks notebook structure, source consistency and embedded evidence without executing the notebook or requiring Jupyter.

## Repository layout

```text
Notebook.ipynb  Consolidated source and evidence
multimodal-fake-news-detection/  Original scripts, configuration, tests and results
v2/                            Later diagnostics and source-disjoint experiment
docs/                          Reproduction, evidence map and packaging notes
tools/                         Read-only result and notebook verification
presentation/                  Accompanying PowerPoint slides
evidence_manifest.csv           SHA-256 fingerprints of packaged result artifacts
```

The two experiment folders retain their original sibling layout. Run original scripts with `multimodal-fake-news-detection` as the working directory. See [reproduction instructions](docs/REPRODUCIBILITY.md), [evidence map](docs/EVIDENCE.md) and [packaging notes and limitations](docs/PACKAGING.md).

## Scope and limitations

Included: source code, selected manifests, saved probabilities, metric summaries, figures, explanation exports and training logs. Excluded: credentials, virtual environments, raw image/data corpora, model checkpoints, feature tensors and dissertation drafts. Full training/inference therefore requires external assets; this is not a standalone deployment.

Completed work includes source-disjoint retraining, calibration, statistical comparisons and explainability. Chronological retraining, CLIP, cross-dataset evaluation and retrieval-assisted verification are not completed experiments in this repository.