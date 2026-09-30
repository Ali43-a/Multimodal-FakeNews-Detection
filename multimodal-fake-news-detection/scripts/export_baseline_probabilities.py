from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from torchvision.models import resnet50
from torchvision import transforms


DATA_ROOT = Path(
    "data/processed/fakeddit/development_clean"
)

ROBERTA_CHECKPOINT = Path(
    "checkpoints/roberta_baseline"
)

RESNET_CHECKPOINT = Path(
    "checkpoints/resnet50_baseline/best_model.pt"
)

OUTPUT_ROOT = Path(
    "results/metrics/baseline_probabilities"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

MAX_LENGTH = 64
BATCH_SIZE_TEXT = 16
BATCH_SIZE_IMAGE = 32
NUM_WORKERS = 4


class TextDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        tokens = self.tokenizer(
            str(row["text"]),
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "id": str(row["id"]),

            "label": int(
                row["label"]
            ),

            "input_ids":
                tokens["input_ids"]
                .squeeze(0),

            "attention_mask":
                tokens["attention_mask"]
                .squeeze(0),
        }


class ImageDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = Path(
            str(row["image_path"])
        )

        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            image = self.transform(
                image
            )

        return {
            "id": str(row["id"]),

            "label": int(
                row["label"]
            ),

            "image": image,
        }


@torch.no_grad()
def export_roberta(
    dataframe,
    split,
    model,
    tokenizer,
    device
):

    dataset = TextDataset(
        dataframe,
        tokenizer
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_TEXT,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        )
    )

    ids = []
    labels = []
    probabilities = []
    predictions = []

    model.eval()

    for batch in loader:

        input_ids = (
            batch["input_ids"]
            .to(
                device,
                non_blocking=True
            )
        )

        attention_mask = (
            batch["attention_mask"]
            .to(
                device,
                non_blocking=True
            )
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                device.type == "cuda"
            )
        ):

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits

        probs = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        preds = torch.argmax(
            logits,
            dim=1
        )

        ids.extend(
            batch["id"]
        )

        labels.extend(
            batch["label"]
            .numpy()
            .tolist()
        )

        probabilities.extend(
            probs
            .cpu()
            .numpy()
            .tolist()
        )

        predictions.extend(
            preds
            .cpu()
            .numpy()
            .tolist()
        )

    output = pd.DataFrame({
        "id": ids,
        "label": labels,
        "prediction": predictions,
        "probability_class_1":
            probabilities,
    })

    path = (
        OUTPUT_ROOT
        / f"roberta_{split}.csv"
    )

    output.to_csv(
        path,
        index=False
    )

    print(
        f"RoBERTa {split}: "
        f"{len(output):,} rows -> {path}"
    )


@torch.no_grad()
def export_resnet(
    dataframe,
    split,
    model,
    transform,
    device
):

    dataset = ImageDataset(
        dataframe,
        transform
    )

    loader_kwargs = {
        "batch_size":
            BATCH_SIZE_IMAGE,

        "shuffle":
            False,

        "num_workers":
            NUM_WORKERS,

        "pin_memory":
            device.type == "cuda",
    }

    if NUM_WORKERS > 0:
        loader_kwargs[
            "persistent_workers"
        ] = True

    loader = DataLoader(
        dataset,
        **loader_kwargs
    )

    ids = []
    labels = []
    probabilities = []
    predictions = []

    model.eval()

    for batch in loader:

        images = (
            batch["image"]
            .to(
                device,
                non_blocking=True
            )
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                device.type == "cuda"
            )
        ):

            logits = model(
                images
            )

        probs = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        preds = torch.argmax(
            logits,
            dim=1
        )

        ids.extend(
            batch["id"]
        )

        labels.extend(
            batch["label"]
            .numpy()
            .tolist()
        )

        probabilities.extend(
            probs
            .cpu()
            .numpy()
            .tolist()
        )

        predictions.extend(
            preds
            .cpu()
            .numpy()
            .tolist()
        )

    output = pd.DataFrame({
        "id": ids,
        "label": labels,
        "prediction": predictions,
        "probability_class_1":
            probabilities,
    })

    path = (
        OUTPUT_ROOT
        / f"resnet_{split}.csv"
    )

    output.to_csv(
        path,
        index=False
    )

    print(
        f"ResNet {split}: "
        f"{len(output):,} rows -> {path}"
    )


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("EXPORT BASELINE PROBABILITIES")
    print("=" * 70)

    print(
        f"\nDevice: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # --------------------------------------------------------
    # RoBERTa
    # --------------------------------------------------------

    print(
        "\nLoading RoBERTa checkpoint..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            ROBERTA_CHECKPOINT
        )
    )

    roberta = (
        AutoModelForSequenceClassification
        .from_pretrained(
            ROBERTA_CHECKPOINT
        )
    )

    roberta.to(device)


    # --------------------------------------------------------
    # ResNet
    # --------------------------------------------------------

    print(
        "Loading ResNet checkpoint..."
    )

    resnet_checkpoint = torch.load(
        RESNET_CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    resnet = resnet50(
        weights=None
    )

    features = (
        resnet.fc.in_features
    )

    resnet.fc = nn.Linear(
        features,
        2
    )

    resnet.load_state_dict(
        resnet_checkpoint[
            "model_state_dict"
        ]
    )

    resnet.to(device)


    image_transform = transforms.Compose([
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


    # --------------------------------------------------------
    # Export VALIDATION and TEST
    # --------------------------------------------------------

    for split in [
        "validation",
        "test"
    ]:

        print(
            "\n" + "=" * 70
        )

        print(
            split.upper()
        )

        print("=" * 70)

        dataframe = pd.read_parquet(
            DATA_ROOT
            / f"{split}.parquet"
        )

        export_roberta(
            dataframe,
            split,
            roberta,
            tokenizer,
            device
        )

        export_resnet(
            dataframe,
            split,
            resnet,
            image_transform,
            device
        )


    print(
        "\nBaseline probability "
        "export complete."
    )


if __name__ == "__main__":
    main()
