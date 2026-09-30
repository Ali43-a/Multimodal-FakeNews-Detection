from pathlib import Path
import pandas as pd

RAW_ROOT = Path("data/raw/fakeddit/multimodal_only_samples")
CLEAN_ROOT = Path("data/processed/fakeddit/development_clean")
OUTPUT = Path("results/metrics/fakeddit_placeholder_source_analysis.csv")

FILES = {
    "train": RAW_ROOT / "multimodal_train.tsv",
    "validation": RAW_ROOT / "multimodal_validate.tsv",
    "test": RAW_ROOT / "multimodal_test_public.tsv",
}

records = []

for split, raw_path in FILES.items():
    print("\n" + "=" * 70)
    print(split.upper())
    print("=" * 70)

    clean = pd.read_parquet(
        CLEAN_ROOT / f"{split}_with_image_status.parquet"
    )

    placeholder_ids = set(
        clean.loc[clean["is_placeholder"] == True, "id"].astype(str)
    )

    raw = pd.read_csv(
        raw_path,
        sep="\t",
        usecols=["id", "domain", "subreddit", "image_url"],
        low_memory=False
    )

    raw["id"] = raw["id"].astype(str)

    affected = raw[raw["id"].isin(placeholder_ids)].copy()

    print(f"Placeholder samples: {len(affected):,}")

    print("\nTop domains:")
    print(
        affected["domain"]
        .fillna("<missing>")
        .value_counts()
        .head(20)
        .to_string()
    )

    print("\nTop subreddits:")
    print(
        affected["subreddit"]
        .fillna("<missing>")
        .value_counts()
        .head(20)
        .to_string()
    )

    for domain, count in affected["domain"].fillna("<missing>").value_counts().items():
        records.append({
            "split": split,
            "domain": domain,
            "count": int(count)
        })

pd.DataFrame(records).to_csv(OUTPUT, index=False)

print(f"\nSaved: {OUTPUT}")
