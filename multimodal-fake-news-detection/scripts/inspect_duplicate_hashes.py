from pathlib import Path
import pandas as pd

ROOT = Path("results/metrics")

for split in ["train", "validation", "test"]:
    print("\n" + "=" * 70)
    print(split.upper())
    print("=" * 70)

    path = ROOT / f"fakeddit_{split}_image_integrity.csv"
    df = pd.read_csv(path)

    valid = df[df["valid"] == True].copy()

    counts = (
        valid["sha256"]
        .value_counts()
        .reset_index()
    )

    counts.columns = ["sha256", "count"]

    repeated = counts[counts["count"] > 1]

    print("\nTop repeated hashes:")
    print(repeated.head(20).to_string(index=False))

    output = ROOT / f"fakeddit_{split}_duplicate_hashes.csv"
    repeated.to_csv(output, index=False)

    print(f"\nSaved: {output}")
