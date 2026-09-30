from pathlib import Path

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


RESULT_ROOT = Path("results/metrics/roberta_baseline")
FIGURE_ROOT = Path("results/figures/roberta_baseline")

FIGURE_ROOT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Training history
# ------------------------------------------------------------

history = pd.read_csv(
    RESULT_ROOT / "training_history.csv"
)

plt.figure()

plt.plot(
    history["epoch"],
    history["train_loss"],
    marker="o",
    label="Training loss"
)

plt.plot(
    history["epoch"],
    history["validation_loss"],
    marker="o",
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RoBERTa training and validation loss")
plt.xticks(history["epoch"])
plt.legend()

plt.tight_layout()
plt.savefig(
    FIGURE_ROOT / "training_validation_loss.png",
    dpi=300
)
plt.close()


# ------------------------------------------------------------
# Validation Macro F1
# ------------------------------------------------------------

plt.figure()

plt.plot(
    history["epoch"],
    history["validation_f1_macro"],
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title("RoBERTa validation Macro F1")
plt.xticks(history["epoch"])

plt.tight_layout()
plt.savefig(
    FIGURE_ROOT / "validation_macro_f1.png",
    dpi=300
)
plt.close()


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

with (
    RESULT_ROOT / "test_metrics.json"
).open("r", encoding="utf-8") as f:
    metrics = json.load(f)

matrix = np.array(
    metrics["confusion_matrix"]
)

plt.figure()

plt.imshow(matrix)

plt.title("RoBERTa test confusion matrix")
plt.xlabel("Predicted class")
plt.ylabel("True class")

plt.xticks([0, 1], ["0", "1"])
plt.yticks([0, 1], ["0", "1"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(matrix[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()
plt.savefig(
    FIGURE_ROOT / "test_confusion_matrix.png",
    dpi=300
)
plt.close()


print("RoBERTa baseline figures created.")

for path in sorted(FIGURE_ROOT.glob("*.png")):
    print(" -", path)
