# Later diagnostics and source-disjoint evaluation

The methodology notebook investigates label semantics, image-retention bias, calibration controls, source shortcuts, duplicate candidates and stronger split designs. It expects the original project as its sibling folder.

**Source-disjoint retraining is complete.** See `experiments/source_disjoint/reports/source_disjoint_results.md`, saved metrics, predictions and logs. Both models were retrained from pretrained weights on source-disjoint training data. Validation selected text weight 0.00. The chronological manifest remains an unevaluated extension.

The notebook has been cleared of embedded outputs for review; saved outputs remain under `outputs/`. Running the full notebook needs the original external dataset and local image assets. Do not execute it merely to inspect results: it writes diagnostics and manifests. Use a separate working copy to reproduce analyses.

See [reproduction and path guidance](../docs/REPRODUCIBILITY.md).
