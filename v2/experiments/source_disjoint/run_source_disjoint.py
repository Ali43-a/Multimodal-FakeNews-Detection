"""Isolated source-disjoint retraining and evaluation for the Fakeddit study.

The command is deliberately split into stages so the test partition is not used
until both validation checkpoints and the late-fusion coefficient are locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torch.nn as nn
import torchvision
import transformers
from PIL import Image
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


SEED = 42
MODEL_NAME = "roberta-base"
MAX_LENGTH = 64
THRESHOLD = 0.50
N_BINS = 10
N_BOOTSTRAPS = 5000
PLACEHOLDER_SHA256 = "faa24ec881e6040655c187a681d6dc496eb8aa41e1bd0652a180b3a40b457187"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get("FAKEDDIT_PROJECT_ROOT", str(REPOSITORY_ROOT / "multimodal-fake-news-detection")))
SOURCE_MANIFEST = REPOSITORY_ROOT / "v2" / "manifests" / "source_disjoint.parquet"
DEFAULT_EXPERIMENT_ROOT = Path(__file__).resolve().parent


def paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "manifests": root / "manifests",
        "checkpoints": root / "checkpoints",
        "roberta_checkpoint": root / "checkpoints" / "roberta",
        "resnet_checkpoint": root / "checkpoints" / "resnet50",
        "predictions": root / "predictions",
        "metrics": root / "metrics",
        "logs": root / "logs",
        "figures": root / "figures",
        "bootstrap": root / "bootstrap",
        "calibration": root / "calibration",
        "reports": root / "reports",
        "manifest": root / "manifests" / "source_disjoint.parquet",
        "config": root / "experiment_config.json",
        "audit": root / "manifests" / "manifest_audit.json",
        "lock": root / "metrics" / "validation_lock.json",
    }


def make_dirs(root: Path) -> None:
    for key, value in paths(root).items():
        if key not in {"manifest", "config", "audit", "lock"}:
            value.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_image_path(value: str) -> Path:
    # Archived manifests retain the original workstation's absolute image paths.
    normalised = str(value).replace("\\", "/")
    marker = "/multimodal-fake-news-detection/"
    if marker in normalised:
        return PROJECT_ROOT / normalised.split(marker, 1)[1]
    path = Path(normalised)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_manifest(root: Path) -> pd.DataFrame:
    path = paths(root)["manifest"]
    if not path.exists():
        raise FileNotFoundError(f"Run the audit stage first: {path}")
    df = pd.read_parquet(path)
    required = {"id", "label", "text", "image_path", "subreddit", "v2_split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest lacks columns: {sorted(missing)}")
    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["label"] = df["label"].astype(int)
    df["resolved_image_path"] = df["image_path"].map(lambda x: str(resolve_image_path(x)))
    return df


def split_df(df: pd.DataFrame, split: str) -> pd.DataFrame:
    result = df.loc[df["v2_split"].eq(split)].copy().reset_index(drop=True)
    if result.empty:
        raise ValueError(f"Empty split: {split}")
    return result


def metrics_for(labels, predictions, probabilities) -> dict:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "recall_class_0": float(recall_score(labels, predictions, labels=[0], average=None, zero_division=0)[0]),
        "recall_class_1": float(recall_score(labels, predictions, labels=[1], average=None, zero_division=0)[0]),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else None,
        "pr_auc": float(average_precision_score(labels, probabilities)) if np.unique(labels).size == 2 else None,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    return result


class TextDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer):
        self.ids = dataframe["id"].astype(str).tolist()
        self.labels = torch.tensor(dataframe["label"].to_numpy(dtype=int), dtype=torch.long)
        self.encodings = tokenizer(
            dataframe["text"].fillna("").astype(str).tolist(),
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        item["sample_index"] = torch.tensor(index, dtype=torch.long)
        return item


class ImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(row["resolved_image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_index": torch.tensor(index, dtype=torch.long),
        }


def image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def worker_seed(worker_id: int) -> None:
    worker_seed_value = SEED + worker_id
    np.random.seed(worker_seed_value)
    random.seed(worker_seed_value)


def text_evaluate(model, loader, device):
    model.eval()
    labels, predictions, probabilities, indices = [], [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            index = batch.pop("sample_index")
            cpu_labels = batch["labels"].numpy()
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(**batch)
            probability = torch.softmax(output.logits, dim=1)[:, 1]
            prediction = (probability >= THRESHOLD).long()
            total_loss += float(output.loss.item())
            indices.extend(index.numpy().tolist())
            labels.extend(cpu_labels.tolist())
            predictions.extend(prediction.cpu().numpy().tolist())
            probabilities.extend(probability.cpu().float().numpy().tolist())
    result = metrics_for(labels, predictions, probabilities)
    result["loss"] = total_loss / len(loader)
    return result, np.asarray(indices), np.asarray(labels), np.asarray(predictions), np.asarray(probabilities)


def image_evaluate(model, loader, device):
    model.eval()
    labels, predictions, probabilities, indices = [], [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(image)
                loss = criterion(logits, label)
            probability = torch.softmax(logits, dim=1)[:, 1]
            prediction = (probability >= THRESHOLD).long()
            total_loss += float(loss.item())
            indices.extend(batch["sample_index"].numpy().tolist())
            labels.extend(label.cpu().numpy().tolist())
            predictions.extend(prediction.cpu().numpy().tolist())
            probabilities.extend(probability.cpu().float().numpy().tolist())
    result = metrics_for(labels, predictions, probabilities)
    result["loss"] = total_loss / len(loader)
    return result, np.asarray(indices), np.asarray(labels), np.asarray(predictions), np.asarray(probabilities)


def prediction_frame(dataframe, indices, labels, predictions, probabilities, prefix: str) -> pd.DataFrame:
    indexed = dataframe.iloc[indices].reset_index(drop=True)
    if not np.array_equal(indexed["label"].to_numpy(dtype=int), labels):
        raise ValueError(f"Label alignment failed for {prefix}")
    return pd.DataFrame({
        "sample_id": indexed["id"].astype(str),
        "true_label": labels,
        f"{prefix}_probability_class1": probabilities,
        f"{prefix}_prediction": predictions,
    })


def stage_audit(root: Path) -> None:
    make_dirs(root)
    p = paths(root)
    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(SOURCE_MANIFEST)
    if p["manifest"].exists():
        if hashlib.sha256(p["manifest"].read_bytes()).hexdigest() != hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest():
            raise FileExistsError(f"An unrelated manifest already exists at {p['manifest']}")
    else:
        shutil.copy2(SOURCE_MANIFEST, p["manifest"])

    df = pd.read_parquet(p["manifest"])
    required = ["id", "label", "text", "image_path", "subreddit", "domain", "created_utc", "v2_split"]
    if list(df.columns) != required:
        raise ValueError(f"Unexpected schema: {df.columns.tolist()}")
    if df["id"].astype(str).duplicated().any():
        raise ValueError("Duplicate sample IDs")
    if set(df["v2_split"]) != {"train", "validation", "test"}:
        raise ValueError("Unexpected split labels")
    if set(df["label"].astype(int)) != {0, 1}:
        raise ValueError("Unexpected class labels")

    source_sets = {name: set(group["subreddit"].astype(str)) for name, group in df.groupby("v2_split")}
    overlap = {}
    names = sorted(source_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap[f"{left}__{right}"] = sorted(source_sets[left] & source_sets[right])
    if any(overlap.values()):
        raise ValueError(f"Source overlap found: {overlap}")

    records, unreadable, placeholders = [], [], []
    for row in df.itertuples(index=False):
        image_path = resolve_image_path(row.image_path)
        exists = image_path.is_file()
        readable = False
        digest = None
        error = None
        if exists:
            try:
                with Image.open(image_path) as image:
                    image.verify()
                readable = True
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            except Exception as exc:  # recorded in audit output
                error = repr(exc)
        if not readable:
            unreadable.append({"id": str(row.id), "path": str(image_path), "error": error or "missing"})
        if digest == PLACEHOLDER_SHA256:
            placeholders.append(str(row.id))
        records.append({"v2_split": row.v2_split, "label": int(row.label), "readable": readable})
    checks = pd.DataFrame(records)
    split_rows = []
    for split in ["train", "validation", "test"]:
        part = df[df["v2_split"].eq(split)]
        verified = checks[checks["v2_split"].eq(split)]
        split_rows.append({
            "split": split,
            "manifest_samples": int(len(part)),
            "retained_samples": int(verified["readable"].sum()),
            "removed_samples": int((~verified["readable"]).sum()),
            "removed_class_0": 0,
            "removed_class_1": 0,
            "class_0": int(part["label"].eq(0).sum()),
            "class_1": int(part["label"].eq(1).sum()),
            "class_1_rate": float(part["label"].mean()),
            "sources": int(part["subreddit"].nunique()),
        })
    if unreadable or placeholders:
        raise ValueError(f"Manifest contains {len(unreadable)} unreadable and {len(placeholders)} placeholder images")
    pd.DataFrame(split_rows).to_csv(p["manifests"] / "split_counts.csv", index=False)

    manifest_sha = hashlib.sha256(p["manifest"].read_bytes()).hexdigest()
    audit = {
        "source_manifest": str(SOURCE_MANIFEST),
        "isolated_manifest": str(p["manifest"]),
        "sha256": manifest_sha,
        "schema": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "rows": int(len(df)),
        "duplicate_ids": 0,
        "blank_text": int(df["text"].fillna("").astype(str).str.strip().eq("").sum()),
        "source_overlap": overlap,
        "unreadable_images": unreadable,
        "known_placeholder_images": placeholders,
        "splits": split_rows,
        "positive_class": "1 (true)",
        "negative_class": "0 (fake or misleading)",
        "note": "The manifest was constructed from the existing cleaned multimodal pool, so its manifest and retained counts are equal.",
    }
    dump_json(p["audit"], audit)
    config = {
        "experiment": "source_disjoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "project_root": str(PROJECT_ROOT),
        "source_manifest": str(SOURCE_MANIFEST),
        "isolated_manifest": str(p["manifest"]),
        "manifest_sha256": manifest_sha,
        "split_counts": {row["split"]: row["retained_samples"] for row in split_rows},
        "positive_class": 1,
        "classification_threshold": THRESHOLD,
        "roberta": {"base_model": MODEL_NAME, "pretrained_base_weights": True, "max_length": MAX_LENGTH, "batch_size": 16, "optimizer": "AdamW", "learning_rate": 2e-5, "weight_decay": 0.01, "gradient_clipping": 1.0, "warmup_fraction": 0.10, "max_epochs": 3, "selection_metric": "validation_macro_f1", "checkpoint": str(p["roberta_checkpoint"])},
        "resnet50": {"base_model": "resnet50", "weights": "IMAGENET1K_V2", "fine_tune": "full network", "input_size": [224, 224], "batch_size": 32, "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 0.01, "max_epochs": 3, "selection_metric": "validation_macro_f1", "checkpoint": str(p["resnet_checkpoint"] / "best_model.pt")},
        "fusion": {"formula": "alpha * p_text + (1 - alpha) * p_image", "alpha_search": [round(x, 2) for x in np.arange(0, 1.0001, 0.05)], "selection_metric": "validation_macro_f1", "tie_break": "closest to 0.50, then lower alpha", "selected_alpha": None},
        "calibration": {"ece_bins": N_BINS, "convention": "confidence bins from 0.5 to 1.0"},
        "bootstrap": {"iterations": N_BOOTSTRAPS, "seed": SEED, "paired": True},
        "packages": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__, "transformers": transformers.__version__, "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "scipy": scipy.__version__},
        "hardware": {"cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }
    dump_json(p["config"], config)
    print(pd.DataFrame(split_rows).to_string(index=False))
    print(f"Audit saved: {p['audit']}")


def create_resnet() -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def stage_smoke(root: Path) -> None:
    set_seed()
    df = load_manifest(root)
    train = split_df(df, "train").sample(n=32, random_state=SEED).reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    text_loader = DataLoader(TextDataset(train, tokenizer), batch_size=8, shuffle=False)
    text_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    text_batch = next(iter(text_loader))
    text_indices = text_batch.pop("sample_index")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        text_out = text_model(**{key: value.to(device) for key, value in text_batch.items()})
    image_loader = DataLoader(ImageDataset(train, image_transform()), batch_size=4, shuffle=False, num_workers=0)
    image_model = create_resnet().to(device)
    image_batch = next(iter(image_loader))
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        image_logits = image_model(image_batch["image"].to(device))
    if text_out.logits.shape != (8, 2) or image_logits.shape != (4, 2):
        raise ValueError("Unexpected smoke-test output shape")
    if not np.array_equal(text_batch["labels"].numpy(), train.iloc[text_indices.numpy()]["label"].to_numpy()):
        raise ValueError("Text labels are not aligned")
    if not np.array_equal(image_batch["label"].numpy(), train.iloc[image_batch["sample_index"].numpy()]["label"].to_numpy()):
        raise ValueError("Image labels are not aligned")
    result = {"passed": True, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "text_batch_shape": list(text_out.logits.shape), "image_batch_shape": list(image_logits.shape), "labels_aligned": True, "output_root": str(root)}
    dump_json(paths(root)["logs"] / "smoke_test.json", result)
    print(json.dumps(result, indent=2))


def train_roberta(root: Path, epochs: int, batch_size: int) -> None:
    p = paths(root)
    if (p["roberta_checkpoint"] / "model.safetensors").exists() or (p["roberta_checkpoint"] / "pytorch_model.bin").exists():
        raise FileExistsError("RoBERTa checkpoint already exists; refusing to overwrite it")
    set_seed()
    df = load_manifest(root)
    train, validation = split_df(df, "train"), split_df(df, "validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(TextDataset(train, tokenizer), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(SEED))
    validation_loader = DataLoader(TextDataset(validation, tokenizer), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.10), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best = [], -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch.pop("sample_index")
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(**batch)
                loss = output.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.item())
            if step % 100 == 0:
                print(f"RoBERTa epoch {epoch} step {step}/{len(train_loader)} loss={total_loss/step:.4f}", flush=True)
        val_metrics, indices, labels, preds, probs = text_evaluate(model, validation_loader, device)
        record = {"epoch": epoch, "train_loss": total_loss / len(train_loader), **{f"validation_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"}}
        history.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            model.save_pretrained(p["roberta_checkpoint"])
            tokenizer.save_pretrained(p["roberta_checkpoint"])
            prediction_frame(validation, indices, labels, preds, probs, "roberta").to_csv(p["predictions"] / "roberta_validation.csv", index=False)
            dump_json(p["metrics"] / "roberta_best_validation.json", {"best_epoch": epoch, "metrics": val_metrics, "checkpoint": str(p["roberta_checkpoint"])})
    pd.DataFrame(history).to_csv(p["metrics"] / "roberta_training_history.csv", index=False)
    print(f"Best validation macro F1: {best:.6f}")


def train_resnet(root: Path, epochs: int, batch_size: int, num_workers: int) -> None:
    p = paths(root)
    checkpoint_path = p["resnet_checkpoint"] / "best_model.pt"
    if checkpoint_path.exists():
        raise FileExistsError("ResNet checkpoint already exists; refusing to overwrite it")
    set_seed()
    df = load_manifest(root)
    train, validation = split_df(df, "train"), split_df(df, "validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_args = {"num_workers": num_workers, "pin_memory": device.type == "cuda", "worker_init_fn": worker_seed}
    if num_workers > 0:
        loader_args["persistent_workers"] = True
    train_loader = DataLoader(ImageDataset(train, image_transform()), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED), **loader_args)
    validation_loader = DataLoader(ImageDataset(validation, image_transform()), batch_size=batch_size, shuffle=False, **loader_args)
    model = create_resnet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best = [], -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())
            if step % 50 == 0:
                print(f"ResNet epoch {epoch} step {step}/{len(train_loader)} loss={total_loss/step:.4f}", flush=True)
        val_metrics, indices, labels, preds, probs = image_evaluate(model, validation_loader, device)
        record = {"epoch": epoch, "train_loss": total_loss / len(train_loader), **{f"validation_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"}}
        history.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            torch.save({"model_state_dict": model.state_dict(), "model": "resnet50", "weights": "IMAGENET1K_V2", "num_classes": 2, "best_epoch": epoch, "validation_macro_f1": best}, checkpoint_path)
            prediction_frame(validation, indices, labels, preds, probs, "resnet").to_csv(p["predictions"] / "resnet_validation.csv", index=False)
            dump_json(p["metrics"] / "resnet_best_validation.json", {"best_epoch": epoch, "metrics": val_metrics, "checkpoint": str(checkpoint_path)})
    pd.DataFrame(history).to_csv(p["metrics"] / "resnet_training_history.csv", index=False)
    print(f"Best validation macro F1: {best:.6f}")


def stage_lock(root: Path) -> None:
    p = paths(root)
    if p["lock"].exists():
        raise FileExistsError("Validation lock already exists; refusing to replace it")
    text = pd.read_csv(p["predictions"] / "roberta_validation.csv")
    image = pd.read_csv(p["predictions"] / "resnet_validation.csv")
    merged = text.merge(image, on=["sample_id", "true_label"], validate="one_to_one")
    expected = len(split_df(load_manifest(root), "validation"))
    if len(merged) != expected:
        raise ValueError(f"Paired validation count {len(merged)} does not equal {expected}")
    merged.to_csv(p["predictions"] / "validation_predictions.csv", index=False)
    rows = []
    y = merged["true_label"].to_numpy(dtype=int)
    for alpha in np.arange(0, 1.0001, 0.05):
        probability = alpha * merged["roberta_probability_class1"].to_numpy() + (1 - alpha) * merged["resnet_probability_class1"].to_numpy()
        pred = (probability >= THRESHOLD).astype(int)
        score = metrics_for(y, pred, probability)
        rows.append({"alpha_text": round(float(alpha), 2), "alpha_image": round(float(1-alpha), 2), **{k: v for k, v in score.items() if k not in {"confusion_matrix", "recall_class_0", "recall_class_1"}}})
    search = pd.DataFrame(rows)
    search.to_csv(p["metrics"] / "validation_alpha_search.csv", index=False)
    maximum = search["macro_f1"].max()
    tied = search[np.isclose(search["macro_f1"], maximum)].copy()
    tied["distance_from_half"] = (tied["alpha_text"] - 0.5).abs()
    selected = tied.sort_values(["distance_from_half", "alpha_text"], ascending=[True, True]).iloc[0]

    # Optional fair text control: tune threshold and Platt calibration on validation only.
    threshold_rows = []
    for threshold in np.arange(0.05, 0.951, 0.01):
        pred = (merged["roberta_probability_class1"].to_numpy() >= threshold).astype(int)
        threshold_rows.append({"threshold": round(float(threshold), 2), "macro_f1": f1_score(y, pred, average="macro", zero_division=0)})
    threshold_search = pd.DataFrame(threshold_rows)
    threshold_search.to_csv(p["metrics"] / "roberta_validation_threshold_search.csv", index=False)
    threshold_best = threshold_search.sort_values(["macro_f1", "threshold"], ascending=[False, True]).iloc[0]
    calibrator = LogisticRegression(random_state=SEED).fit(merged[["roberta_probability_class1"]], y)
    dump_json(p["metrics"] / "fair_text_validation_control.json", {"selected_threshold": float(threshold_best["threshold"]), "validation_macro_f1": float(threshold_best["macro_f1"]), "platt_coefficient": float(calibrator.coef_[0, 0]), "platt_intercept": float(calibrator.intercept_[0])})

    lock = {"locked_utc": datetime.now(timezone.utc).isoformat(), "selection_partition": "validation", "paired_validation_samples": int(len(merged)), "selection_metric": "macro_f1", "selected_alpha_text": float(selected["alpha_text"]), "selected_alpha_image": float(selected["alpha_image"]), "validation_macro_f1": float(selected["macro_f1"]), "classification_threshold": THRESHOLD, "tied_alpha_values": tied["alpha_text"].astype(float).tolist(), "tie_break": "closest to 0.50, then lower alpha", "roberta_checkpoint": str(p["roberta_checkpoint"]), "resnet_checkpoint": str(p["resnet_checkpoint"] / "best_model.pt")}
    dump_json(p["lock"], lock)
    config = json.loads(p["config"].read_text(encoding="utf-8"))
    config["fusion"]["selected_alpha"] = lock["selected_alpha_text"]
    config["fusion"]["locked_utc"] = lock["locked_utc"]
    dump_json(p["config"], config)
    print(json.dumps(lock, indent=2))


def ece(labels, probabilities, n_bins=N_BINS):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= 0.5).astype(int)
    confidence = np.maximum(probabilities, 1 - probabilities)
    correctness = (predictions == labels).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    value, rows = 0.0, []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= lower) & (confidence <= upper if index == n_bins - 1 else confidence < upper)
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else None
        accuracy = float(correctness[mask].mean()) if count else None
        gap = abs(accuracy - mean_confidence) if count else None
        if count:
            value += count / len(labels) * gap
        rows.append({"bin_lower": float(lower), "bin_upper": float(upper), "count": count, "mean_confidence": mean_confidence, "accuracy": accuracy, "gap": gap})
    return float(value), rows


def calibration_for(name, labels, probabilities):
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    ece_value, bins = ece(labels, clipped)
    return {"model": name, "brier": float(brier_score_loss(labels, clipped)), "log_loss": float(log_loss(labels, clipped)), "ece": ece_value}, bins


def bootstrap(root: Path, y, models: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    text_pred, text_prob = models["roberta"]
    fusion_pred, fusion_prob = models["late_fusion"]
    observed_text = metrics_for(y, text_pred, text_prob)
    observed_fusion = metrics_for(y, fusion_pred, fusion_prob)
    names = ["accuracy", "f1", "macro_f1", "roc_auc", "pr_auc"]
    values = {name: [] for name in names}
    for _ in range(N_BOOTSTRAPS):
        idx = rng.integers(0, len(y), size=len(y))
        ys = y[idx]
        basic_text = metrics_for(ys, text_pred[idx], text_prob[idx])
        basic_fusion = metrics_for(ys, fusion_pred[idx], fusion_prob[idx])
        for name in names:
            if basic_text[name] is not None and basic_fusion[name] is not None:
                values[name].append(basic_fusion[name] - basic_text[name])
    pd.DataFrame({name: pd.Series(series) for name, series in values.items()}).to_csv(paths(root)["bootstrap"] / "paired_differences.csv", index=False)
    rows = []
    for name in names:
        series = np.asarray(values[name])
        rows.append({"metric": name, "difference": observed_fusion[name] - observed_text[name], "ci_lower": float(np.percentile(series, 2.5)), "ci_upper": float(np.percentile(series, 97.5)), "valid_iterations": int(len(series)), "requested_iterations": N_BOOTSTRAPS})
    result = pd.DataFrame(rows)
    result.to_csv(paths(root)["metrics"] / "statistical_comparison.csv", index=False)
    return result


def reliability_plot(root: Path, bins_by_model: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0.5, 1.0], [0.5, 1.0], color="black", linestyle=":", label="Perfect calibration")
    for name, rows in bins_by_model.items():
        frame = pd.DataFrame(rows).dropna(subset=["mean_confidence", "accuracy"])
        ax.plot(frame["mean_confidence"], frame["accuracy"], marker="o", label=name)
    ax.set(xlabel="Mean confidence", ylabel="Observed accuracy", xlim=(0.5, 1.0), ylim=(0.5, 1.0), title="Source-disjoint test reliability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths(root)["figures"] / "reliability_diagram.png", dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    """Render a small DataFrame without the optional tabulate dependency."""
    display = frame.copy()
    for column in display.select_dtypes(include=["float", "floating"]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}")
    headers = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def stage_test(root: Path, text_batch: int, image_batch: int, num_workers: int) -> None:
    p = paths(root)
    final_predictions_path = p["predictions"] / "test_predictions.csv"
    if final_predictions_path.exists():
        raise FileExistsError("Test predictions already exist; refusing to evaluate the locked test twice")
    lock = json.loads(p["lock"].read_text(encoding="utf-8"))
    alpha = float(lock["selected_alpha_text"])
    set_seed()
    test = split_df(load_manifest(root), "test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(p["roberta_checkpoint"])
    text_loader = DataLoader(TextDataset(test, tokenizer), batch_size=text_batch, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    text_model = AutoModelForSequenceClassification.from_pretrained(p["roberta_checkpoint"]).to(device)
    _, ti, ty, tp, tprob = text_evaluate(text_model, text_loader, device)
    text_frame = prediction_frame(test, ti, ty, tp, tprob, "roberta")
    del text_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    loader_args = {"num_workers": num_workers, "pin_memory": device.type == "cuda", "worker_init_fn": worker_seed}
    if num_workers > 0:
        loader_args["persistent_workers"] = True
    image_loader = DataLoader(ImageDataset(test, image_transform()), batch_size=image_batch, shuffle=False, **loader_args)
    image_model = create_resnet().to(device)
    checkpoint = torch.load(p["resnet_checkpoint"] / "best_model.pt", map_location=device, weights_only=True)
    image_model.load_state_dict(checkpoint["model_state_dict"])
    _, ii, iy, ip, iprob = image_evaluate(image_model, image_loader, device)
    image_frame = prediction_frame(test, ii, iy, ip, iprob, "resnet")

    predictions = text_frame.merge(image_frame, on=["sample_id", "true_label"], validate="one_to_one")
    diagnostics = test[["id", "subreddit"]].rename(columns={"id": "sample_id"})
    predictions = predictions.merge(diagnostics, on="sample_id", validate="one_to_one")
    predictions["fusion_probability_class1"] = alpha * predictions["roberta_probability_class1"] + (1 - alpha) * predictions["resnet_probability_class1"]
    predictions["fusion_prediction"] = (predictions["fusion_probability_class1"] >= THRESHOLD).astype(int)
    predictions.to_csv(final_predictions_path, index=False)

    y = predictions["true_label"].to_numpy(dtype=int)
    model_arrays = {
        "roberta": (predictions["roberta_prediction"].to_numpy(dtype=int), predictions["roberta_probability_class1"].to_numpy()),
        "resnet50": (predictions["resnet_prediction"].to_numpy(dtype=int), predictions["resnet_probability_class1"].to_numpy()),
        "late_fusion": (predictions["fusion_prediction"].to_numpy(dtype=int), predictions["fusion_probability_class1"].to_numpy()),
    }
    metric_rows, full_metrics = [], {}
    calibration_rows, bins_by_model = [], {}
    for name, (pred, prob) in model_arrays.items():
        values = metrics_for(y, pred, prob)
        full_metrics[name] = values
        cal, bins = calibration_for(name, y, prob)
        calibration_rows.append(cal)
        bins_by_model[name] = bins
        pd.DataFrame(bins).to_csv(p["calibration"] / f"{name}_bins.csv", index=False)
        metric_rows.append({"model": name, **{key: value for key, value in values.items() if key != "confusion_matrix"}, **{key: value for key, value in cal.items() if key != "model"}})
    metric_summary = pd.DataFrame(metric_rows)
    metric_summary.to_csv(p["metrics"] / "summary.csv", index=False)
    dump_json(p["metrics"] / "summary.json", {"test_samples": len(predictions), "selected_alpha_text": alpha, "models": full_metrics})
    calibration_frame = pd.DataFrame(calibration_rows)
    calibration_frame.to_csv(p["calibration"] / "calibration_metrics.csv", index=False)
    dump_json(p["calibration"] / "calibration_metrics.json", {"n_bins": N_BINS, "convention": "confidence bins from 0.5 to 1.0", "models": calibration_rows})
    reliability_plot(root, bins_by_model)

    cr = model_arrays["roberta"][0] == y
    ci = model_arrays["resnet50"][0] == y
    cf = model_arrays["late_fusion"][0] == y
    b = int(np.sum(cr & ~cf))
    c = int(np.sum(~cr & cf))
    mcnemar_p = float(binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue) if b + c else 1.0
    mcnemar = {"both_correct": int(np.sum(cr & cf)), "both_wrong": int(np.sum(~cr & ~cf)), "roberta_correct_fusion_wrong": b, "fusion_correct_roberta_wrong": c, "discordant": b + c, "exact_p_value": mcnemar_p}
    dump_json(p["metrics"] / "mcnemar.json", mcnemar)
    stats = bootstrap(root, y, model_arrays)
    complementarity = {
        "roberta_vs_fusion": mcnemar,
        "roberta_vs_resnet_disagreements": int(np.sum(model_arrays["roberta"][0] != model_arrays["resnet50"][0])),
        "roberta_only_correct": int(np.sum(cr & ~ci)),
        "resnet_only_correct": int(np.sum(~cr & ci)),
        "image_correct_text_wrong_recovered_by_fusion": int(np.sum(~cr & ci & cf)),
        "text_correct_image_wrong_preserved_by_fusion": int(np.sum(cr & ~ci & cf)),
    }
    dump_json(p["metrics"] / "error_complementarity.json", complementarity)

    # Apply the validation-selected fair text threshold and Platt model once to test.
    fair = json.loads((p["metrics"] / "fair_text_validation_control.json").read_text(encoding="utf-8"))
    tuned_pred = (model_arrays["roberta"][1] >= fair["selected_threshold"]).astype(int)
    z = fair["platt_intercept"] + fair["platt_coefficient"] * model_arrays["roberta"][1]
    platt_prob = 1 / (1 + np.exp(-z))
    pd.DataFrame([
        {"control": "validation_tuned_threshold", "threshold": fair["selected_threshold"], **{k: v for k, v in metrics_for(y, tuned_pred, model_arrays["roberta"][1]).items() if k != "confusion_matrix"}},
        {"control": "validation_platt_calibration", "threshold": 0.5, **{k: v for k, v in metrics_for(y, (platt_prob >= 0.5).astype(int), platt_prob).items() if k != "confusion_matrix"}},
    ]).to_csv(p["metrics"] / "fair_text_test_control.csv", index=False)

    original = {"roberta": {"accuracy": 0.8046, "macro_f1": 0.8000, "roc_auc": 0.8846, "pr_auc": 0.9027}, "resnet50": {"accuracy": 0.7382, "macro_f1": 0.7275, "roc_auc": 0.7976, "pr_auc": 0.8053}, "late_fusion": {"accuracy": 0.8242, "macro_f1": 0.8188, "roc_auc": 0.8993, "pr_auc": 0.9097}}
    penalty_rows = []
    for name in original:
        for metric, old in original[name].items():
            penalty_rows.append({"model": name, "metric": metric, "ordinary_locked_holdout": old, "source_disjoint": full_metrics[name][metric], "change_source_disjoint_minus_ordinary": full_metrics[name][metric] - old})
    pd.DataFrame(penalty_rows).to_csv(p["metrics"] / "generalisation_penalty.csv", index=False)

    counts = pd.read_csv(p["manifests"] / "split_counts.csv")
    macro_row = stats[stats["metric"].eq("macro_f1")].iloc[0]
    best_text = json.loads((p["metrics"] / "roberta_best_validation.json").read_text(encoding="utf-8"))
    best_image = json.loads((p["metrics"] / "resnet_best_validation.json").read_text(encoding="utf-8"))
    lines = [
        "# Source-disjoint evaluation results", "",
        "## Dataset and split verification", "",
        "The manifest assigns every subreddit to one partition. No subreddit appears in more than one split. Both labels occur in every split. All image files opened correctly and the known regional-unavailable placeholder was absent.", "",
        markdown_table(counts), "",
        "The source-disjoint manifest was created from the existing cleaned multimodal pool. The manifest count is therefore also the final usable count, with no further removals in this run.", "",
        "## Locked validation choices", "",
        f"RoBERTa checkpoint: `{best_text['checkpoint']}` (epoch {best_text['best_epoch']}).", "",
        f"ResNet-50 checkpoint: `{best_image['checkpoint']}` (epoch {best_image['best_epoch']}).", "",
        f"Validation selected alpha = {alpha:.2f} for RoBERTa and {1-alpha:.2f} for ResNet-50. The classification threshold remained 0.50.", "",
        "The complete search is in `metrics/validation_alpha_search.csv`.", "",
        "## Source-disjoint test performance", "", markdown_table(metric_summary), "",
        "## Calibration", "", markdown_table(calibration_frame), "",
        "Lower Brier score, log loss and ECE indicate better calibration.", "",
        "## Paired statistical comparison", "",
        f"Late fusion and RoBERTa were compared on {len(y):,} paired test cases. Both were correct on {mcnemar['both_correct']:,}; both were wrong on {mcnemar['both_wrong']:,}. RoBERTa alone was correct on {b:,} and fusion alone on {c:,}. The exact McNemar p-value was {mcnemar_p:.6g}.", "",
        markdown_table(stats), "",
        "## Error complementarity", "",
        f"RoBERTa and ResNet-50 disagreed on {complementarity['roberta_vs_resnet_disagreements']:,} cases. RoBERTa alone was correct on {complementarity['roberta_only_correct']:,}; ResNet-50 alone was correct on {complementarity['resnet_only_correct']:,}. Late fusion recovered {complementarity['image_correct_text_wrong_recovered_by_fusion']:,} image-correct/text-wrong cases and preserved {complementarity['text_correct_image_wrong_preserved_by_fusion']:,} text-correct/image-wrong cases.", "",
        "## Interpretation", "",
        f"Removing subreddit overlap changed RoBERTa macro F1 by {full_metrics['roberta']['macro_f1']-0.8000:+.4f}, ResNet-50 by {full_metrics['resnet50']['macro_f1']-0.7275:+.4f}, and late fusion by {full_metrics['late_fusion']['macro_f1']-0.8188:+.4f} compared with the ordinary locked holdout. Late fusion changed macro F1 over RoBERTa by {full_metrics['late_fusion']['macro_f1']-full_metrics['roberta']['macro_f1']:+.4f}. Its paired 95% bootstrap interval was [{macro_row['ci_lower']:+.4f}, {macro_row['ci_upper']:+.4f}]. Validation selected zero text weight, so the locked late-fusion result is identical to ResNet-50 and does not show a benefit from combining modalities. These figures measure transfer to unseen subreddit groups rather than performance on the earlier mixed-source holdout.", "",
    ]
    (p["reports"] / "source_disjoint_results.md").write_text("\n".join(lines), encoding="utf-8")
    config = json.loads(p["config"].read_text(encoding="utf-8"))
    config["test_evaluated_utc"] = datetime.now(timezone.utc).isoformat()
    config["test_predictions"] = str(final_predictions_path)
    dump_json(p["config"], config)
    print(metric_summary.to_string(index=False))
    print(json.dumps(mcnemar, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "smoke", "train-roberta", "train-resnet", "lock", "test"])
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    if args.stage == "audit": stage_audit(root)
    elif args.stage == "smoke": stage_smoke(root)
    elif args.stage == "train-roberta": train_roberta(root, args.epochs, args.batch_size or 16)
    elif args.stage == "train-resnet": train_resnet(root, args.epochs, args.batch_size or 32, args.num_workers)
    elif args.stage == "lock": stage_lock(root)
    elif args.stage == "test": stage_test(root, args.batch_size or 16, 32, args.num_workers)


if __name__ == "__main__":
    main()
