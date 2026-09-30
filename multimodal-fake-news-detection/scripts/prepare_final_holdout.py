import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image


CANDIDATE_PATH = Path(
    "data/processed/fakeddit/final_holdout/candidates.parquet"
)

IMAGE_ROOT = Path(
    "data/raw/fakeddit/final_holdout_images"
)

DEVELOPMENT_IMAGE_ROOT = Path(
    "data/raw/fakeddit/development_images"
)

OUTPUT_ROOT = Path(
    "data/processed/fakeddit/final_holdout"
)

METRICS_ROOT = Path(
    "results/metrics/final_holdout"
)


FINAL_SIZE = 5000

MAX_WORKERS = 16
TIMEOUT_SECONDS = 20
MAX_ATTEMPTS = 3


# Known invalid "not available in your region" placeholder.
PLACEHOLDER_HASH = (
    "faa24ec881e6040655c187a681d6dc496eb8aa41e1bd0652a180b3a40b457187"
)


IMAGE_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


HEADERS = {
    "User-Agent":
        "Mozilla/5.0 multimodal-fake-news-research"
}


def sha256_bytes(data):

    return hashlib.sha256(
        data
    ).hexdigest()


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_image_bytes(data):

    try:

        with Image.open(
            BytesIO(data)
        ) as image:

            image.verify()

        # Open a second time because verify()
        # invalidates the first PIL object.
        with Image.open(
            BytesIO(data)
        ) as image:

            image = image.convert("RGB")

            width, height = image.size

        return (
            True,
            int(width),
            int(height),
            None,
        )

    except Exception as exc:

        return (
            False,
            None,
            None,
            str(exc),
        )


def validate_existing_file(path):

    try:

        data = path.read_bytes()

    except Exception as exc:

        return {
            "valid":
                False,

            "sha256":
                None,

            "width":
                None,

            "height":
                None,

            "error":
                str(exc),
        }


    valid, width, height, error = (
        validate_image_bytes(data)
    )

    return {
        "valid":
            valid,

        "sha256":
            (
                sha256_bytes(data)
                if valid
                else None
            ),

        "width":
            width,

        "height":
            height,

        "error":
            error,
    }


def download_one(index, row):

    sample_id = str(
        row["id"]
    )

    url = str(
        row["image_url"]
    ).strip()

    output_path = (
        IMAGE_ROOT
        / f"{sample_id}.jpg"
    )


    result = {
        "candidate_order":
            int(index),

        "id":
            sample_id,

        "image_path":
            str(output_path),

        "download_status":
            None,

        "image_valid":
            False,

        "image_sha256":
            None,

        "image_width":
            None,

        "image_height":
            None,

        "download_error":
            None,
    }


    if (
        not url
        or url.lower() == "nan"
    ):

        result[
            "download_status"
        ] = "missing_url"

        return result


    # --------------------------------------------------------
    # Reuse an existing downloaded file.
    # --------------------------------------------------------

    if output_path.exists():

        checked = (
            validate_existing_file(
                output_path
            )
        )

        if checked["valid"]:

            result.update({
                "download_status":
                    "already_present",

                "image_valid":
                    True,

                "image_sha256":
                    checked["sha256"],

                "image_width":
                    checked["width"],

                "image_height":
                    checked["height"],
            })

            return result

        else:

            # Remove broken partial download
            # and try again.
            try:
                output_path.unlink()
            except Exception:
                pass


    # --------------------------------------------------------
    # Download with retry.
    # --------------------------------------------------------

    last_error = None


    for attempt in range(
        1,
        MAX_ATTEMPTS + 1
    ):

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=TIMEOUT_SECONDS
            )

            response.raise_for_status()

            data = response.content


            valid, width, height, error = (
                validate_image_bytes(
                    data
                )
            )


            if not valid:

                last_error = (
                    f"invalid_image: {error}"
                )

                continue


            output_path.write_bytes(
                data
            )


            result.update({
                "download_status":
                    "downloaded",

                "image_valid":
                    True,

                "image_sha256":
                    sha256_bytes(data),

                "image_width":
                    width,

                "image_height":
                    height,

                "download_error":
                    None,
            })


            return result


        except Exception as exc:

            last_error = str(exc)

            if attempt < MAX_ATTEMPTS:

                time.sleep(
                    0.5 * attempt
                )


    result[
        "download_status"
    ] = "failed"

    result[
        "download_error"
    ] = last_error

    return result


print("=" * 70)
print("PREPARE FINAL UNSEEN HOLDOUT")
print("=" * 70)


# ============================================================
# Load frozen candidate manifest
# ============================================================

candidates = pd.read_parquet(
    CANDIDATE_PATH
)

candidates["id"] = (
    candidates["id"]
    .astype(str)
)

candidates["text"] = (
    candidates["text"]
    .astype("string")
    .fillna("")
)

candidates["image_url"] = (
    candidates["image_url"]
    .astype("string")
    .fillna("")
)


print(
    f"\nFrozen candidates: "
    f"{len(candidates):,}"
)

print(
    f"Target final holdout: "
    f"{FINAL_SIZE:,}"
)


# ============================================================
# Hash all EXISTING development images
#
# This detects visual duplicates even when URLs differ.
# ============================================================

print(
    "\nHashing existing development images..."
)


development_hashes = set()

development_files = [
    path
    for path
    in DEVELOPMENT_IMAGE_ROOT.rglob("*")
    if path.is_file()
]


for i, path in enumerate(
    development_files,
    start=1
):

    try:

        development_hashes.add(
            sha256_file(path)
        )

    except Exception:
        pass


    if i % 5000 == 0:

        print(
            f"  Hashed {i:,} "
            "development files"
        )


print(
    f"Development files scanned: "
    f"{len(development_files):,}"
)

print(
    f"Unique development image hashes: "
    f"{len(development_hashes):,}"
)


# ============================================================
# Download candidate images
# ============================================================

print(
    "\nDownloading / validating "
    "candidate images..."
)

print(
    f"Workers: {MAX_WORKERS}"
)


records = []


with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    futures = {
        executor.submit(
            download_one,
            index,
            row
        ): index

        for index, row
        in candidates.iterrows()
    }


    completed = 0


    for future in as_completed(
        futures
    ):

        records.append(
            future.result()
        )

        completed += 1


        if (
            completed % 500 == 0
            or completed
            == len(candidates)
        ):

            print(
                f"  Completed "
                f"{completed:,} / "
                f"{len(candidates):,}"
            )


status = pd.DataFrame(
    records
)


# Restore deterministic candidate order.
status = status.sort_values(
    "candidate_order"
).reset_index(
    drop=True
)


manifest = candidates.merge(
    status,
    on="id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# Cleaning flags
# ============================================================

manifest[
    "is_known_placeholder"
] = (
    manifest["image_sha256"]
    == PLACEHOLDER_HASH
)


manifest[
    "image_seen_in_development"
] = (
    manifest["image_sha256"]
    .fillna("")
    .isin(
        development_hashes
    )
)


# Exact duplicate text within final candidate pool.
#
# Keep the earliest sample in the already-frozen
# candidate order.
normalised_text = (
    manifest["text"]
    .astype("string")
    .fillna("")
    .str.strip()
)

manifest[
    "duplicate_text_within_candidates"
] = (
    normalised_text.ne("")
    &
    normalised_text.duplicated(
        keep="first"
    )
)


# Exact image duplicate inside final candidate pool.
#
# Again keep first occurrence only.
hash_values = (
    manifest[
        "image_sha256"
    ]
    .fillna("")
)

manifest[
    "duplicate_image_within_candidates"
] = (
    hash_values.ne("")
    &
    hash_values.duplicated(
        keep="first"
    )
)


manifest[
    "usable_final"
] = (
    manifest[
        "image_valid"
    ].fillna(False)
    &
    ~manifest[
        "is_known_placeholder"
    ]
    &
    ~manifest[
        "image_seen_in_development"
    ]
    &
    ~manifest[
        "duplicate_text_within_candidates"
    ]
    &
    ~manifest[
        "duplicate_image_within_candidates"
    ]
)


# ============================================================
# Diagnostics
# ============================================================

print(
    "\nDownload status:"
)

print(
    manifest[
        "download_status"
    ]
    .value_counts(
        dropna=False
    )
)


print(
    "\nCleaning exclusions:"
)

print(
    "Invalid / unavailable image:",
    int(
        (
            ~manifest[
                "image_valid"
            ].fillna(False)
        ).sum()
    )
)

print(
    "Known placeholder:",
    int(
        manifest[
            "is_known_placeholder"
        ].sum()
    )
)

print(
    "Image seen in development:",
    int(
        manifest[
            "image_seen_in_development"
        ].sum()
    )
)

print(
    "Duplicate text within candidates:",
    int(
        manifest[
            "duplicate_text_within_candidates"
        ].sum()
    )
)

print(
    "Duplicate image within candidates:",
    int(
        manifest[
            "duplicate_image_within_candidates"
        ].sum()
    )
)


usable = manifest[
    manifest[
        "usable_final"
    ]
].copy()


print(
    f"\nUsable leakage-reduced samples: "
    f"{len(usable):,}"
)


if len(usable) < FINAL_SIZE:

    raise RuntimeError(
        "Fewer than "
        f"{FINAL_SIZE:,} usable samples. "
        "Do not change model settings. "
        "Increase the candidate pool instead."
    )


# ============================================================
# Freeze EXACTLY first 5,000 usable samples
#
# Candidate ordering was fixed before any image download
# or model evaluation.
# ============================================================

final = (
    usable
    .head(
        FINAL_SIZE
    )
    .copy()
    .reset_index(drop=True)
)


# ============================================================
# Final overlap assertions
# ============================================================

assert (
    final["id"]
    .nunique()
    == len(final)
)

assert not final[
    "is_known_placeholder"
].any()

assert not final[
    "image_seen_in_development"
].any()

assert not final[
    "duplicate_text_within_candidates"
].any()

assert not final[
    "duplicate_image_within_candidates"
].any()


# ============================================================
# Holdout fingerprint
# ============================================================

ordered_ids = "\n".join(
    final["id"]
    .astype(str)
    .tolist()
)

id_sha256 = hashlib.sha256(
    ordered_ids.encode("utf-8")
).hexdigest()


# ============================================================
# Save manifests
# ============================================================

manifest.to_parquet(
    OUTPUT_ROOT
    / "candidate_image_status.parquet",
    index=False
)

final.to_parquet(
    OUTPUT_ROOT
    / "final_5000.parquet",
    index=False
)


label_counts = {
    str(key): int(value)
    for key, value
    in final[
        "label"
    ]
    .value_counts()
    .sort_index()
    .items()
}


metadata = {
    "purpose":
        "final unseen multimodal evaluation",

    "candidate_count":
        int(len(candidates)),

    "target_final_count":
        FINAL_SIZE,

    "usable_before_final_selection":
        int(len(usable)),

    "final_count":
        int(len(final)),

    "final_label_counts":
        label_counts,

    "ordered_id_sha256":
        id_sha256,

    "known_placeholder_hash":
        PLACEHOLDER_HASH,

    "cleaning": {
        "require_readable_image":
            True,

        "exclude_known_placeholder":
            True,

        "exclude_development_image_hashes":
            True,

        "deduplicate_exact_text_within_candidates":
            True,

        "deduplicate_exact_image_hash_within_candidates":
            True,
    },

    "fusion_model_locked": {
        "text_model":
            "roberta-base fine-tuned checkpoint",

        "image_model":
            "resnet50 fine-tuned checkpoint",

        "alpha_text":
            0.45,

        "alpha_image":
            0.55,

        "decision_threshold":
            0.5,
    },

    "model_predictions_used_for_selection":
        False,

    "model_metrics_used_for_selection":
        False,
}


with (
    METRICS_ROOT
    / "final_holdout_freeze.json"
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
    "\n" + "=" * 70
)

print(
    "FINAL HOLDOUT FROZEN"
)

print("=" * 70)


print(
    f"\nFinal samples: "
    f"{len(final):,}"
)

print(
    "\nFinal label counts:"
)

print(
    final[
        "label"
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nOrdered ID SHA256:"
)

print(
    id_sha256
)


print(
    "\nFinal manifest:"
)

print(
    OUTPUT_ROOT
    / "final_5000.parquet"
)


print(
    "\nIMPORTANT:"
)

print(
    "The final holdout is now frozen."
)

print(
    "Do not alter models, fusion weights, "
    "thresholds, or cleaning rules after "
    "viewing final performance."
)
