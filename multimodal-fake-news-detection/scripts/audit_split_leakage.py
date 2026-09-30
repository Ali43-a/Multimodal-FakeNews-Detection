from pathlib import Path
import json
import pandas as pd

BASE = Path("data/raw/fakeddit/multimodal_only_samples")

FILES = {
    "train": BASE / "multimodal_train.tsv",
    "validation": BASE / "multimodal_validate.tsv",
    "test": BASE / "multimodal_test_public.tsv",
}

OUTPUT = Path("results/metrics/fakeddit_split_leakage.json")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

USE_COLUMNS = [
    "id",
    "clean_title",
    "image_url",
    "2_way_label",
]

splits = {}

for name, path in FILES.items():
    print(f"Loading {name}...")
    splits[name] = pd.read_csv(
        path,
        sep="\t",
        usecols=USE_COLUMNS,
        low_memory=False
    )

pairs = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]

report = {}

for left_name, right_name in pairs:
    left = splits[left_name]
    right = splits[right_name]

    print("\n" + "=" * 70)
    print(f"{left_name.upper()} vs {right_name.upper()}")
    print("=" * 70)

    pair_key = f"{left_name}_vs_{right_name}"
    report[pair_key] = {}

    # ID overlap
    left_ids = set(left["id"].dropna())
    right_ids = set(right["id"].dropna())
    id_overlap = left_ids.intersection(right_ids)

    print(f"ID overlap: {len(id_overlap)}")

    report[pair_key]["id_overlap"] = len(id_overlap)

    # Exact clean-title overlap
    left_titles = set(left["clean_title"].dropna())
    right_titles = set(right["clean_title"].dropna())
    title_overlap = left_titles.intersection(right_titles)

    print(f"Exact clean_title overlap: {len(title_overlap)}")

    report[pair_key]["clean_title_overlap"] = len(title_overlap)

    # Image URL overlap
    left_images = set(left["image_url"].dropna())
    right_images = set(right["image_url"].dropna())
    image_overlap = left_images.intersection(right_images)

    print(f"Image URL overlap: {len(image_overlap)}")

    report[pair_key]["image_url_overlap"] = len(image_overlap)

    # Exact text + image pair overlap
    left_pairs = set(
        zip(
            left["clean_title"].fillna(""),
            left["image_url"].fillna("")
        )
    )

    right_pairs = set(
        zip(
            right["clean_title"].fillna(""),
            right["image_url"].fillna("")
        )
    )

    pair_overlap = left_pairs.intersection(right_pairs)

    print(f"Exact text-image pair overlap: {len(pair_overlap)}")

    report[pair_key]["text_image_pair_overlap"] = len(pair_overlap)

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nAudit complete.")
print(f"Saved to: {OUTPUT}")
