from pathlib import Path
import hashlib
import json

import pandas as pd
from PIL import Image, UnidentifiedImageError


IMAGE_ROOT = Path("data/raw/fakeddit/development_images")
MANIFEST_ROOT = Path("data/processed/fakeddit/development")
OUTPUT_DIR = Path("results/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]

summary = {}

for split in SPLITS:
    print("\n" + "=" * 70)
    print(split.upper())
    print("=" * 70)

    image_dir = IMAGE_ROOT / split
    manifest_path = MANIFEST_ROOT / f"{split}.parquet"

    manifest = pd.read_parquet(manifest_path)

    results = []
    hash_counts = {}

    for row in manifest.itertuples(index=False):
        sample_id = str(row.id)
        image_path = image_dir / f"{sample_id}.jpg"

        record = {
            "id": sample_id,
            "exists": image_path.exists(),
            "valid": False,
            "width": None,
            "height": None,
            "mode": None,
            "format": None,
            "size_bytes": None,
            "sha256": None,
            "error": None,
        }

        if not image_path.exists():
            record["error"] = "missing_file"
            results.append(record)
            continue

        try:
            record["size_bytes"] = image_path.stat().st_size

            with image_path.open("rb") as f:
                file_bytes = f.read()

            sha256 = hashlib.sha256(file_bytes).hexdigest()
            record["sha256"] = sha256

            hash_counts[sha256] = hash_counts.get(sha256, 0) + 1

            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                img.load()

                record["width"] = img.width
                record["height"] = img.height
                record["mode"] = img.mode
                record["format"] = img.format

                if img.width <= 0 or img.height <= 0:
                    record["error"] = "invalid_dimensions"
                else:
                    record["valid"] = True

        except UnidentifiedImageError:
            record["error"] = "unidentified_image"

        except Exception as exc:
            record["error"] = str(exc)

        results.append(record)

    result_df = pd.DataFrame(results)

    duplicate_hashes = {
        h: count
        for h, count in hash_counts.items()
        if count > 1
    }

    output_csv = OUTPUT_DIR / f"fakeddit_{split}_image_integrity.csv"
    result_df.to_csv(output_csv, index=False)

    valid_count = int(result_df["valid"].sum())
    missing_count = int((~result_df["exists"]).sum())
    invalid_count = int(
        ((result_df["exists"]) & (~result_df["valid"])).sum()
    )

    suspicious_34641 = int(
        (result_df["size_bytes"] == 34641).sum()
    )

    duplicate_file_count = sum(
        count for count in duplicate_hashes.values()
    )

    print(f"Manifest rows: {len(result_df):,}")
    print(f"Valid images: {valid_count:,}")
    print(f"Missing files: {missing_count:,}")
    print(f"Invalid/corrupt images: {invalid_count:,}")
    print(f"Files exactly 34,641 bytes: {suspicious_34641:,}")
    print(f"Files involved in duplicate hashes: {duplicate_file_count:,}")
    print(f"Unique duplicated hashes: {len(duplicate_hashes):,}")

    summary[split] = {
        "manifest_rows": int(len(result_df)),
        "valid_images": valid_count,
        "missing_files": missing_count,
        "invalid_images": invalid_count,
        "files_34641_bytes": suspicious_34641,
        "duplicate_hash_groups": int(len(duplicate_hashes)),
        "files_in_duplicate_hash_groups": int(duplicate_file_count),
    }

with (OUTPUT_DIR / "fakeddit_image_integrity_summary.json").open(
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\nImage integrity audit complete.")
