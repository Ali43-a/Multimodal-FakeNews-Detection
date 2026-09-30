from pathlib import Path
import time

import pandas as pd
import requests
from tqdm import tqdm


DATA_DIR = Path("data/processed/fakeddit/development")
IMAGE_ROOT = Path("data/raw/fakeddit/development_images")
LOG_DIR = Path("results/metrics")

IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]

TIMEOUT = 15
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36"
    )
}


def download_image(session, url, destination):
    """
    Download a single image with basic retry handling.
    """

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                url,
                timeout=TIMEOUT,
                headers=HEADERS,
                stream=True
            )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type",
                ""
            ).lower()

            if "image" not in content_type:
                return False, f"non-image content: {content_type}"

            with destination.open("wb") as f:
                for chunk in response.iter_content(
                    chunk_size=8192
                ):
                    if chunk:
                        f.write(chunk)

            if destination.stat().st_size == 0:
                destination.unlink(missing_ok=True)
                return False, "zero-byte image"

            return True, ""

        except Exception as exc:
            if attempt == MAX_RETRIES:
                return False, str(exc)

            time.sleep(attempt * 1.5)

    return False, "unknown failure"


for split in SPLITS:

    print("\n" + "=" * 70)
    print(split.upper())
    print("=" * 70)

    manifest = pd.read_parquet(
        DATA_DIR / f"{split}.parquet"
    )

    image_dir = IMAGE_ROOT / split
    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    failures = []
    skipped_missing_url = 0
    already_present = 0
    downloaded = 0

    with requests.Session() as session:

        for row in tqdm(
            manifest.itertuples(index=False),
            total=len(manifest),
            desc=f"Downloading {split}"
        ):

            sample_id = str(row.id)
            url = row.image_url

            destination = image_dir / f"{sample_id}.jpg"

            if destination.exists() and destination.stat().st_size > 0:
                already_present += 1
                continue

            if pd.isna(url) or not str(url).strip():
                skipped_missing_url += 1
                failures.append({
                    "id": sample_id,
                    "url": "",
                    "reason": "missing image URL"
                })
                continue

            success, reason = download_image(
                session,
                str(url),
                destination
            )

            if success:
                downloaded += 1
            else:
                destination.unlink(missing_ok=True)

                failures.append({
                    "id": sample_id,
                    "url": str(url),
                    "reason": reason
                })

    failure_df = pd.DataFrame(failures)

    failure_path = (
        LOG_DIR /
        f"fakeddit_{split}_image_download_failures.csv"
    )

    failure_df.to_csv(
        failure_path,
        index=False
    )

    print()
    print(f"Samples: {len(manifest):,}")
    print(f"Downloaded: {downloaded:,}")
    print(f"Already present: {already_present:,}")
    print(f"Missing URLs: {skipped_missing_url:,}")
    print(f"Failures: {len(failures):,}")
    print(f"Failure log: {failure_path}")

print("\nDevelopment image download complete.")
