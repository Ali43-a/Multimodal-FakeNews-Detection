import argparse
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


SEED = 42
MAX_LENGTH = 64

DATA_ROOT = Path("data/processed/fakeddit/development_clean")

ROBERTA_CHECKPOINT = Path("checkpoints/roberta_baseline")
RESNET_CHECKPOINT = Path("checkpoints/resnet50_baseline/best_model.pt")

OUTPUT_ROOT = Path("data/processed/fakeddit/multimodal_features")


class MultimodalDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        image_transform
    ):
        self.df = dataframe.reset_index(drop=True)

        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        sample_id = str(row["id"])

        text = str(row["text"])

        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )

        tokens = {
            key: value.squeeze(0)
            for key, value in tokens.items()
        }

        image_path = Path(
            str(row["image_path"])
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.image_transform(image)

        label = int(row["label"])

        return {
            "id": sample_id,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "image": image,
            "label": label,
        }


def load_roberta(device):

    print("Loading fine-tuned RoBERTa...")

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            ROBERTA_CHECKPOINT
        )
    )

    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


def load_resnet(device):

    print("Loading fine-tuned ResNet-50...")

    checkpoint = torch.load(
        RESNET_CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    model = resnet50(
        weights=None
    )

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        2
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Remove classifier so output becomes
    # the 2048-dimensional visual feature vector.
    model.fc = nn.Identity()

    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


@torch.no_grad()
def extract_split(
    split,
    dataframe,
    tokenizer,
    roberta,
    resnet,
    image_transform,
    device,
    batch_size,
    num_workers,
    output_root
):

    print("\n" + "=" * 70)
    print(split.upper())
    print("=" * 70)

    dataset = MultimodalDataset(
        dataframe,
        tokenizer,
        image_transform
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs[
            "persistent_workers"
        ] = True

    loader = DataLoader(
        dataset,
        **loader_kwargs
    )

    all_ids = []
    all_labels = []

    all_text_features = []
    all_image_features = []

    for step, batch in enumerate(
        loader,
        start=1
    ):

        input_ids = batch[
            "input_ids"
        ].to(
            device,
            non_blocking=True
        )

        attention_mask = batch[
            "attention_mask"
        ].to(
            device,
            non_blocking=True
        )

        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda"
        ):

            roberta_output = (
                roberta.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            )

            # First-token representation
            text_features = (
                roberta_output
                .last_hidden_state[:, 0, :]
            )

            image_features = resnet(
                images
            )

        all_text_features.append(
            text_features.float().cpu()
        )

        all_image_features.append(
            image_features.float().cpu()
        )

        all_labels.append(
            batch["label"].long()
        )

        all_ids.extend(
            batch["id"]
        )

        if step % 100 == 0:
            print(
                f"  Batch {step:4d}/{len(loader)}"
            )

    text_features = torch.cat(
        all_text_features,
        dim=0
    )

    image_features = torch.cat(
        all_image_features,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    if text_features.shape[0] != len(dataframe):
        raise ValueError(
            "Text feature row count mismatch."
        )

    if image_features.shape[0] != len(dataframe):
        raise ValueError(
            "Image feature row count mismatch."
        )

    print(
        f"\nSamples:        {len(all_ids):,}"
    )

    print(
        "Text features: ",
        tuple(text_features.shape)
    )

    print(
        "Image features:",
        tuple(image_features.shape)
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_root
        / f"{split}.pt"
    )

    torch.save(
        {
            "ids": all_ids,
            "labels": labels,
            "text_features":
                text_features,
            "image_features":
                image_features,
        },
        output_path
    )

    print(
        f"Saved: {output_path}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true"
    )

    args = parser.parse_args()

    torch.manual_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("MULTIMODAL FEATURE EXTRACTION")
    print("=" * 70)

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            ROBERTA_CHECKPOINT
        )
    )

    roberta = load_roberta(
        device
    )

    resnet = load_resnet(
        device
    )

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
        ),
    ])

    output_root = OUTPUT_ROOT

    if args.smoke_test:

        print("\nSMOKE TEST MODE")

        output_root = Path(
            "data/processed/fakeddit/"
            "multimodal_features_smoke"
        )

    for split in [
        "train",
        "validation",
        "test"
    ]:

        df = pd.read_parquet(
            DATA_ROOT / f"{split}.parquet"
        )

        if args.smoke_test:

            df = df.sample(
                n=min(256, len(df)),
                random_state=SEED
            ).reset_index(drop=True)

        extract_split(
            split=split,
            dataframe=df,
            tokenizer=tokenizer,
            roberta=roberta,
            resnet=resnet,
            image_transform=image_transform,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_root=output_root
        )

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
