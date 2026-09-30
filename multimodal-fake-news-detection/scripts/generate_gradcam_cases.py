from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
)

import torch
import torch.nn as nn

from torchvision.models import resnet50
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
)


SEED = 42
N_PER_GROUP = 8

ERROR_PATH = Path(
    "results/metrics/error_analysis/"
    "paired_test_error_analysis.csv"
)

CHECKPOINT_PATH = Path(
    "checkpoints/resnet50_baseline/best_model.pt"
)

OUTPUT_ROOT = Path(
    "results/gradcam"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Load ResNet
# ============================================================

model = resnet50(
    weights=None
)

features = model.fc.in_features

model.fc = nn.Linear(
    features,
    2
)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


# Last convolutional block.
target_layers = [
    model.layer4[-1]
]


transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# Load paired analysis
# ============================================================

df = pd.read_csv(
    ERROR_PATH
)

df["id"] = (
    df["id"]
    .astype(str)
)


groups = [
    "fusion_only_correct",
    "roberta_only_correct",
    "both_correct",
    "both_wrong",
]


cam = GradCAM(
    model=model,
    target_layers=target_layers
)


summary_rows = []


for group in groups:

    subset = df[
        df["comparison_group"]
        == group
    ].copy()

    if len(subset) == 0:
        continue

    sample_n = min(
        N_PER_GROUP,
        len(subset)
    )

    sampled = subset.sample(
        n=sample_n,
        random_state=SEED
    )

    group_root = (
        OUTPUT_ROOT / group
    )

    group_root.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        "\n" + "=" * 70
    )

    print(
        group.upper()
    )

    print("=" * 70)


    for _, row in sampled.iterrows():

        image_path = Path(
            str(row["image_path"])
        )

        if not image_path.exists():

            print(
                "Missing:",
                image_path
            )

            continue


        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            display_image = (
                image.resize(
                    (224, 224)
                )
            )

            rgb = (
                np.asarray(
                    display_image
                )
                .astype(np.float32)
                / 255.0
            )

            input_tensor = (
                transform(image)
                .unsqueeze(0)
                .to(device)
            )


        # Get ResNet's predicted class.
        with torch.no_grad():

            logits = model(
                input_tensor
            )

            predicted_class = int(
                torch.argmax(
                    logits,
                    dim=1
                ).item()
            )


        # Explain the predicted class.
        targets = [
            ClassifierOutputTarget(
                predicted_class
            )
        ]


        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]


        visualization = (
            show_cam_on_image(
                rgb,
                grayscale_cam,
                use_rgb=True
            )
        )


        output_path = (
            group_root
            / f"{row['id']}.jpg"
        )


        Image.fromarray(
            visualization
        ).save(
            output_path
        )


        summary_rows.append({
            "id":
                row["id"],

            "comparison_group":
                group,

            "label":
                int(row["label"]),

            "prediction_roberta":
                int(
                    row[
                        "prediction_roberta"
                    ]
                ),

            "prediction_resnet":
                int(
                    row[
                        "prediction_resnet"
                    ]
                ),

            "prediction_fusion":
                int(
                    row[
                        "prediction_fusion"
                    ]
                ),

            "probability_roberta":
                float(
                    row[
                        "probability_roberta"
                    ]
                ),

            "probability_resnet":
                float(
                    row[
                        "probability_resnet"
                    ]
                ),

            "probability_fused":
                float(
                    row[
                        "probability_fused"
                    ]
                ),

            "text":
                str(row["text"]),

            "image_path":
                str(image_path),

            "gradcam_path":
                str(output_path),
        })


        print(
            f"{row['id']} | "
            f"label={int(row['label'])} | "
            f"text={int(row['prediction_roberta'])} | "
            f"image={int(row['prediction_resnet'])} | "
            f"fusion={int(row['prediction_fusion'])}"
        )
        
summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUTPUT_ROOT
    / "gradcam_case_summary.csv",
    index=False
)


print(
    "\n" + "=" * 70
)

print(
    "GRAD-CAM COMPLETE"
)

print("=" * 70)

print(
    f"\nGenerated {len(summary):,} "
    "case visualisations."
)

print(
    "Saved to:",
    OUTPUT_ROOT
)
