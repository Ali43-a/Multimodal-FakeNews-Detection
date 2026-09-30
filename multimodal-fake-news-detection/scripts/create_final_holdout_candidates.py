import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 2026
CANDIDATE_SIZE = 10000


FULL_TEST_PATH = Path(
    "data/processed/fakeddit/test.parquet"
)

DEV_ROOT = Path(
    "data/processed/fakeddit/development"
)

OUTPUT_ROOT = Path(
    "data/processed/fakeddit/final_holdout"
)

METRICS_ROOT = Path(
    "results/metrics/final_holdout"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


def resolve_column(
    dataframe,
    candidates,
    name
):

    for column in candidates:

        if column in dataframe.columns:
            return column

    raise ValueError(
        f"Could not find {name}. "
        f"Available columns: "
        f"{list(dataframe.columns)}"
    )


def normalise_string(series):

    return (
        series
        .astype("string")
        .fillna("")
        .str.strip()
    )


print("=" * 70)
print("CREATE FINAL HOLDOUT CANDIDATES")
print("=" * 70)


# ============================================================
# Load full official public test pool
# ============================================================

full = pd.read_parquet(
    FULL_TEST_PATH
)

print(
    f"\nFull public test rows: "
    f"{len(full):,}"
)

print(
    "\nColumns:"
)

print(
    list(full.columns)
)


id_col = resolve_column(
    full,
    ["id"],
    "ID column"
)

label_col = resolve_column(
    full,
    [
        "label",
        "2_way_label",
    ],
    "binary label column"
)

text_col = resolve_column(
    full,
    [
        "text",
        "clean_title",
        "title",
    ],
    "text column"
)

image_url_col = resolve_column(
    full,
    [
        "image_url",
    ],
    "image URL column"
)


# ============================================================
# Load ALL development manifests
#
# These include every sample we have previously used:
# train + validation + development test.
# ============================================================

development_frames = []

for split in [
    "train",
    "validation",
    "test",
]:

    path = (
        DEV_ROOT
        / f"{split}.parquet"
    )

    frame = pd.read_parquet(
        path
    )

    frame["_development_split"] = (
        split
    )

    development_frames.append(
        frame
    )


development = pd.concat(
    development_frames,
    ignore_index=True
)


print(
    f"\nPreviously used development rows: "
    f"{len(development):,}"
)


dev_id_col = resolve_column(
    development,
    ["id"],
    "development ID column"
)

dev_text_col = resolve_column(
    development,
    [
        "text",
        "clean_title",
        "title",
    ],
    "development text column"
)

dev_image_url_col = resolve_column(
    development,
    ["image_url"],
    "development image URL column"
)


# ============================================================
# Canonical values for overlap checks
# ============================================================

full_ids = normalise_string(
    full[id_col]
)

full_text = normalise_string(
    full[text_col]
)

full_urls = normalise_string(
    full[image_url_col]
)


development_ids = set(
    normalise_string(
        development[dev_id_col]
    )
)

development_text = set(
    value
    for value in normalise_string(
        development[dev_text_col]
    )
    if value != ""
)

development_urls = set(
    value
    for value in normalise_string(
        development[dev_image_url_col]
    )
    if value != ""
)


# ============================================================
# Leakage exclusion
#
# Final holdout cannot share:
# - ID
# - exact processed text
# - exact image URL
#
# with anything previously used during development.
# ============================================================

id_overlap = (
    full_ids.isin(
        development_ids
    )
)

text_overlap = (
    full_text.ne("")
    &
    full_text.isin(
        development_text
    )
)

url_overlap = (
    full_urls.ne("")
    &
    full_urls.isin(
        development_urls
    )
)


any_overlap = (
    id_overlap
    |
    text_overlap
    |
    url_overlap
)


print(
    "\nOverlap with development data:"
)

print(
    f"ID overlap:        "
    f"{int(id_overlap.sum()):,}"
)

print(
    f"Exact text overlap:"
    f" {int(text_overlap.sum()):,}"
)

print(
    f"Image URL overlap: "
    f"{int(url_overlap.sum()):,}"
)

print(
    f"Excluded by union: "
    f"{int(any_overlap.sum()):,}"
)


eligible = (
    full.loc[
        ~any_overlap
    ]
    .copy()
    .reset_index(drop=True)
)


print(
    f"\nEligible untouched pool: "
    f"{len(eligible):,}"
)


if len(eligible) < CANDIDATE_SIZE:

    raise ValueError(
        "Eligible pool is smaller than "
        f"{CANDIDATE_SIZE:,}."
    )


# ============================================================
# Select 10,000 candidates
#
# Stratification preserves the binary-label distribution
# of the untouched eligible pool.
# ============================================================

candidates, _ = train_test_split(
    eligible,
    train_size=CANDIDATE_SIZE,
    random_state=SEED,
    stratify=eligible[label_col],
)


candidates = (
    candidates
    .sample(
        frac=1.0,
        random_state=SEED
    )
    .reset_index(drop=True)
)


# Standard aliases for downstream scripts.
candidates["id"] = (
    candidates[id_col]
    .astype(str)
)

candidates["label"] = (
    candidates[label_col]
    .astype(int)
)

candidates["text"] = (
    candidates[text_col]
    .astype("string")
    .fillna("")
)

candidates["image_url"] = (
    candidates[image_url_col]
    .astype("string")
    .fillna("")
)


output_path = (
    OUTPUT_ROOT
    / "candidates.parquet"
)

candidates.to_parquet(
    output_path,
    index=False
)


# ============================================================
# Reproducibility metadata
# ============================================================

label_counts = {
    str(key): int(value)
    for key, value
    in candidates[
        "label"
    ]
    .value_counts()
    .sort_index()
    .items()
}


metadata = {
    "purpose":
        "unseen final evaluation holdout",

    "source":
        str(FULL_TEST_PATH),

    "candidate_seed":
        SEED,

    "candidate_size":
        CANDIDATE_SIZE,

    "full_public_test_rows":
        int(len(full)),

    "previous_development_rows":
        int(len(development)),

    "overlap_counts": {
        "id":
            int(id_overlap.sum()),

        "exact_text":
            int(text_overlap.sum()),

        "image_url":
            int(url_overlap.sum()),

        "union_excluded":
            int(any_overlap.sum()),
    },

    "eligible_pool_rows":
        int(len(eligible)),

    "candidate_label_counts":
        label_counts,

    "model_predictions_used":
        False,

    "test_metrics_used_for_selection":
        False,
}


with (
    METRICS_ROOT
    / "candidate_selection.json"
).open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


print(
    "\nCandidate label counts:"
)

print(
    candidates[
        "label"
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nCandidate manifest saved to:"
)

print(
    output_path
)


print(
    "\nIMPORTANT:"
)

print(
    "No model predictions or model "
    "performance were used to select "
    "these samples."
)


print(
    "\n" + "=" * 70
)

print(
    "FINAL HOLDOUT CANDIDATE "
    "SELECTION COMPLETE"
)

print("=" * 70)
