from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEV_ROOT = Path("data/processed/fakeddit/development")
CLEAN_ROOT = Path("data/processed/fakeddit/development_clean")
FIGURE_ROOT = Path("results/figures/eda")
METRIC_ROOT = Path("results/metrics")

FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
METRIC_ROOT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]

summary_rows = []


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Load datasets
# ============================================================

original = {}
status = {}
clean = {}

for split in SPLITS:
    original[split] = pd.read_parquet(
        DEV_ROOT / f"{split}.parquet"
    )

    status[split] = pd.read_parquet(
        CLEAN_ROOT / f"{split}_with_image_status.parquet"
    )

    clean[split] = pd.read_parquet(
        CLEAN_ROOT / f"{split}.parquet"
    )


# ============================================================
# 1. Original vs cleaned class distribution
# ============================================================

rows = []

for split in SPLITS:
    for dataset_name, df in [
        ("Original", original[split]),
        ("Cleaned", clean[split]),
    ]:
        counts = df["label"].value_counts().sort_index()

        for label, count in counts.items():
            rows.append({
                "split": split,
                "dataset": dataset_name,
                "label": int(label),
                "count": int(count),
                "proportion": count / len(df),
            })

class_df = pd.DataFrame(rows)

for split in SPLITS:
    subset = class_df[class_df["split"] == split]

    pivot = subset.pivot(
        index="label",
        columns="dataset",
        values="proportion"
    )

    ax = pivot.plot(kind="bar")

    ax.set_title(
        f"{split.capitalize()} class distribution: original vs cleaned"
    )
    ax.set_xlabel("Class label")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.legend(title="Dataset")

    plt.xticks(rotation=0)

    save_figure(
        f"{split}_class_distribution_original_vs_cleaned.png"
    )


# ============================================================
# 2. Image removal rate by class
# ============================================================

removal_rows = []

for split in SPLITS:
    df = status[split]

    removal = (
        df.groupby("label")["image_usable"]
        .apply(lambda x: 1 - x.mean())
    )

    for label, rate in removal.items():
        removal_rows.append({
            "split": split,
            "label": int(label),
            "removal_rate": float(rate),
        })

removal_df = pd.DataFrame(removal_rows)

pivot = removal_df.pivot(
    index="split",
    columns="label",
    values="removal_rate"
)

ax = pivot.plot(kind="bar")

ax.set_title("Unusable image rate by class and split")
ax.set_xlabel("Dataset split")
ax.set_ylabel("Removal rate")
ax.set_ylim(0, 1)
ax.legend(title="Class label")

plt.xticks(rotation=0)

save_figure("image_removal_rate_by_class.png")


# ============================================================
# 3. Overall usable image rate by split
# ============================================================

availability_rows = []

for split in SPLITS:
    df = status[split]

    usable = int(df["image_usable"].sum())
    total = len(df)

    availability_rows.append({
        "split": split,
        "usable": usable,
        "unusable": total - usable,
        "usable_rate": usable / total,
    })

availability_df = pd.DataFrame(availability_rows)

ax = availability_df.set_index("split")[
    ["usable", "unusable"]
].plot(
    kind="bar",
    stacked=True
)

ax.set_title("Usable and unusable images by split")
ax.set_xlabel("Dataset split")
ax.set_ylabel("Number of samples")

plt.xticks(rotation=0)

save_figure("usable_images_by_split.png")


# ============================================================
# 4. Text-length distribution
# ============================================================

combined_clean = pd.concat(
    [
        clean[split].assign(dataset_split=split)
        for split in SPLITS
    ],
    ignore_index=True
)

if "text_length_words" in combined_clean.columns:
    text_lengths = combined_clean["text_length_words"]
else:
    text_lengths = (
        combined_clean["text"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

plt.figure()

plt.hist(
    text_lengths.clip(upper=text_lengths.quantile(0.995)),
    bins=50
)

plt.title("Text length distribution in cleaned multimodal dataset")
plt.xlabel("Number of words")
plt.ylabel("Number of samples")

save_figure("cleaned_text_length_distribution.png")


# ============================================================
# 5. Image width distribution
# ============================================================

image_rows = []

for split in SPLITS:
    df = status[split]

    usable = df[df["image_usable"] == True].copy()
    usable["dataset_split"] = split

    image_rows.append(usable)

usable_images = pd.concat(
    image_rows,
    ignore_index=True
)

width_limit = usable_images["width"].quantile(0.99)

plt.figure()

plt.hist(
    usable_images["width"].clip(upper=width_limit),
    bins=50
)

plt.title("Image width distribution")
plt.xlabel("Image width (pixels)")
plt.ylabel("Number of images")

save_figure("image_width_distribution.png")


# ============================================================
# 6. Image height distribution
# ============================================================

height_limit = usable_images["height"].quantile(0.99)

plt.figure()

plt.hist(
    usable_images["height"].clip(upper=height_limit),
    bins=50
)

plt.title("Image height distribution")
plt.xlabel("Image height (pixels)")
plt.ylabel("Number of images")

save_figure("image_height_distribution.png")


# ============================================================
# 7. Placeholder artefact counts
# ============================================================

placeholder_rows = []

for split in SPLITS:
    df = status[split]

    count = int(df["is_placeholder"].sum())

    placeholder_rows.append({
        "split": split,
        "placeholder_count": count,
    })

placeholder_df = pd.DataFrame(placeholder_rows)

ax = placeholder_df.set_index("split")[
    "placeholder_count"
].plot(kind="bar")

ax.set_title(
    "Repeated regional-unavailable placeholder images"
)
ax.set_xlabel("Dataset split")
ax.set_ylabel("Number of placeholder images")

plt.xticks(rotation=0)

save_figure("placeholder_images_by_split.png")


# ============================================================
# 8. Save numerical EDA summary
# ============================================================

for split in SPLITS:
    original_df = original[split]
    status_df = status[split]
    clean_df = clean[split]

    for label in sorted(original_df["label"].unique()):
        original_label = original_df[
            original_df["label"] == label
        ]

        status_label = status_df[
            status_df["label"] == label
        ]

        clean_label = clean_df[
            clean_df["label"] == label
        ]

        summary_rows.append({
            "split": split,
            "label": int(label),

            "original_count": len(original_label),
            "clean_count": len(clean_label),

            "original_proportion":
                len(original_label) / len(original_df),

            "clean_proportion":
                len(clean_label) / len(clean_df),

            "image_removal_rate":
                1 - status_label["image_usable"].mean(),

            "placeholder_count":
                int(status_label["is_placeholder"].sum()),
        })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    METRIC_ROOT / "fakeddit_eda_summary.csv",
    index=False
)


# ============================================================
# Console summary
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print("\nClean dataset sizes:")

for split in SPLITS:
    print(
        f"{split.capitalize():12s}: "
        f"{len(clean[split]):,}"
    )

print("\nFigures created:")

for figure in sorted(FIGURE_ROOT.glob("*.png")):
    print(f" - {figure}")

print(
    "\nSummary saved to:"
    "\nresults/metrics/fakeddit_eda_summary.csv"
)
