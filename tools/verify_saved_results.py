"""Read-only verification of the packaged dissertation evidence (stdlib only)."""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_models(rows, label, specs, scores, macro_key):
    for model, prediction, probability in specs:
        cm = [[0, 0], [0, 0]]
        for row in rows:
            y, pred, prob = int(row[label]), int(row[prediction]), float(row[probability])
            require(y in (0, 1) and pred in (0, 1), "Invalid binary label")
            require(math.isfinite(prob) and 0 <= prob <= 1, "Invalid probability")
            require(pred == int(prob >= 0.5), f"{model}: prediction/threshold mismatch")
            cm[y][pred] += 1
        accuracy = (cm[0][0] + cm[1][1]) / len(rows)
        macro = sum(2 * cm[k][k] / (2 * cm[k][k] + cm[0][1] + cm[1][0]) for k in (0, 1)) / 2
        require(cm == scores[model]["confusion_matrix"], f"{model}: confusion matrix mismatch")
        require(abs(accuracy - scores[model]["accuracy"]) < 1e-10, f"{model}: accuracy mismatch")
        require(abs(macro - scores[model][macro_key]) < 1e-10, f"{model}: macro F1 mismatch")
        print(f"  {model:12s} n={len(rows):4d} accuracy={accuracy:.4f} macro_F1={macro:.4f}")


def main():
    manifest = load_csv(ROOT / "evidence_manifest.csv")
    for row in manifest:
        path = ROOT / row["path"]
        require(path.is_file(), f"Missing evidence: {row['path']}")
        require(path.stat().st_size == int(row["bytes"]), f"Size changed: {row['path']}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], f"Fingerprint changed: {row['path']}")
    print(f"Verified {len(manifest)} evidence fingerprints.")

    ordinary = ROOT / "multimodal-fake-news-detection/results/metrics/final_evaluation"
    rows = load_csv(ordinary / "final_predictions.csv")
    saved = json.loads((ordinary / "final_results.json").read_text())
    ids = [r["id"] for r in rows]
    require(len(ids) == len(set(ids)) == saved["holdout"]["samples"] == 5000, "Ordinary ID/size mismatch")
    require(hashlib.sha256("\n".join(ids).encode()).hexdigest() == saved["holdout"]["ordered_id_sha256"], "Ordinary ordered-ID freeze mismatch")
    alpha = saved["locked_specification"]["alpha_text"]
    require(alpha == 0.45, "Unexpected ordinary fusion weight")
    for r in rows:
        expected = alpha * float(r["probability_roberta"]) + (1-alpha) * float(r["probability_resnet"])
        require(abs(expected-float(r["probability_fusion"])) < 1e-10, "Ordinary fusion arithmetic mismatch")
    print("Ordinary locked holdout:")
    verify_models(rows, "label", [(m, f"prediction_{col}", f"probability_{col}") for m,col in [("roberta","roberta"),("resnet","resnet"),("late_fusion","fusion")]], saved["metrics"], "f1_macro")

    source = ROOT / "v2/experiments/source_disjoint"
    rows = load_csv(source / "predictions/test_predictions.csv")
    saved = json.loads((source / "metrics/summary.json").read_text())
    require(len(rows) == len({r["sample_id"] for r in rows}) == saved["test_samples"] == 2739, "Source ID/size mismatch")
    require(saved["selected_alpha_text"] == 0, "Unexpected source-disjoint weight")
    for r in rows:
        require(abs(float(r["fusion_probability_class1"])-float(r["resnet_probability_class1"])) < 1e-10, "Source fusion arithmetic mismatch")
    print("Source-disjoint test:")
    verify_models(rows, "true_label", [(m, f"{col}_prediction", f"{col}_probability_class1") for m,col in [("roberta","roberta"),("resnet50","resnet"),("late_fusion","fusion")]], saved["models"], "macro_f1")
    print("PASS: packaged results agree with saved predictions. No experiments were rerun.")


if __name__ == "__main__":
    main()
