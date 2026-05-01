#!/usr/bin/env python3

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt

from models.fusion_model import FusionModel
from dataset.transforms import resnet_test, clip_test  # 👈 USE YOUR FILE

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "best_model.pth"
from configs.config import BENCHMARK_DIR
DATASET_PATH = BENCHMARK_DIR
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# CLASS MAPPING
# ==============================
MODEL_TO_BENCH = {
    0: "real",
    1: "dalle",
    2: "midjourney",
    3: "sd",
    4: "sd",
    5: "sd"
}

BENCH_CLASSES = ["real", "dalle", "midjourney", "sd"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(BENCH_CLASSES)}

# ==============================
# DATASET
# ==============================
class BenchmarkDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for label in BENCH_CLASSES:
            folder = os.path.join(root_dir, label)
            for fname in os.listdir(folder):
                path = os.path.join(folder, fname)
                self.samples.append((path, CLASS_TO_IDX[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        # 🔥 IMPORTANT: TWO TRANSFORMS
        resnet_img = resnet_test(image)
        clip_img = clip_test(image)

        return resnet_img, clip_img, label


# ==============================
# LOAD MODEL
# ==============================
def load_model():
    model = FusionModel(num_classes=6)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ==============================
# EVALUATE
# ==============================
def evaluate():
    dataset = BenchmarkDataset(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model()

    all_preds = []
    all_labels = []

    print(f"Total samples: {len(dataset)}")

    with torch.no_grad():
        for resnet_img, clip_img, labels in tqdm(loader, desc="Evaluating"):
            resnet_img = resnet_img.to(DEVICE)
            clip_img = clip_img.to(DEVICE)

            outputs = model(resnet_img, clip_img)

            preds = torch.argmax(outputs, dim=1)

            # 🔥 map to benchmark classes
            mapped_preds = [
                CLASS_TO_IDX[MODEL_TO_BENCH[p.item()]]
                for p in preds
            ]

            all_preds.extend(mapped_preds)
            all_labels.extend(labels.numpy())

    # ==============================
    # METRICS
    # ==============================
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print("\n===== RESULTS =====")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(all_labels, all_preds, target_names=BENCH_CLASSES))

    # ==============================
    # CONFUSION MATRIX
    # ==============================
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=BENCH_CLASSES,
                yticklabels=BENCH_CLASSES)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")
    plt.show()


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    evaluate()