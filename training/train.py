import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
import os
from pathlib import Path

from dataset.multiclass_dataset import MultiClassImageDataset
from dataset.transforms import *
from utils.split_dataset import stratified_split
from models.fusion_model import FusionModel
from engine.train import train_one_epoch
from engine.eval import evaluate

# ==============================
# CONFIG
# ==============================
from configs.config import DATASET_DIR
ROOT_DIR = DATASET_DIR

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# DATASET
# ==============================
print("📦 Loading dataset...")

dataset = MultiClassImageDataset(
    ROOT_DIR,
    resnet_transform=resnet_train,
    clip_transform=clip_train
)

train_set, val_set, test_set = stratified_split(dataset)

# 🔥 IMPORTANT: different transforms per split
train_set.dataset.resnet_transform = resnet_train
train_set.dataset.clip_transform = clip_train

val_set.dataset.resnet_transform = resnet_test
val_set.dataset.clip_transform = clip_test

test_set.dataset.resnet_transform = resnet_test
test_set.dataset.clip_transform = clip_test

# ==============================
# DATALOADERS
# ==============================
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# ==============================
# MODEL
# ==============================
print("🧠 Loading model...")

model = FusionModel().to(device)

# ==============================
# CLASS WEIGHTS (normalized)
# ==============================
labels = [label for _, label in dataset.samples]
counts = Counter(labels)

weights = torch.tensor([1.0 / counts[i] for i in range(6)], dtype=torch.float)
weights = weights / weights.sum()
weights = weights.to(device)

print("📊 Class Weights:", weights)

# ==============================
# LOSS + OPTIMIZER + SCHEDULER
# ==============================
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)

# ==============================
# TRAIN LOOP
# ==============================
best_f1 = 0

train_losses = []
train_accs = []
val_accs = []
val_f1s = []

print("\n🚀 Training started...\n")

for epoch in range(EPOCHS):
    print(f"\n📘 Epoch [{epoch+1}/{EPOCHS}]")

    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )

    acc, prec, rec, f1, cm = evaluate(model, val_loader, device)

    scheduler.step()

    # 🔹 Store metrics
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_accs.append(acc)
    val_f1s.append(f1)

    print(f"📉 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"✅ Val Acc: {acc:.4f}, F1: {f1:.4f}")
    print("📊 Confusion Matrix:\n", cm)

    # 🔥 Save BEST based on F1
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_model.pth")
        print("💾 Best model saved!")

# ==============================
# TEST
# ==============================
print("\n🧪 Testing best model...")

model.load_state_dict(torch.load("best_model.pth"))

acc, prec, rec, f1, cm = evaluate(model, test_loader, device)

print("\n===== FINAL RESULTS =====")
print(f"🎯 Test Accuracy: {acc:.4f}")
print(f"🎯 Test F1 Score: {f1:.4f}")
print("📊 Confusion Matrix:\n", cm)

# ==============================
# SAVE FINAL MODEL
# ==============================
torch.save(model.state_dict(), "final_model.pth")
print("\n💾 Final model saved!")

# ==============================
# 📈 PLOTS
# ==============================
epochs_range = range(1, EPOCHS + 1)

# Loss
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()

# Accuracy
plt.plot(epochs_range, train_accs, label="Train Acc")
plt.plot(epochs_range, val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.savefig("accuracy_curve.png")
plt.show()

# F1
plt.plot(epochs_range, val_f1s, label="Val F1")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("F1 Score Curve")
plt.legend()
plt.savefig("f1_curve.png")
plt.show()

# ==============================
# CONFUSION MATRIX PLOT
# ==============================
import seaborn as sns

classes = ["Real","DALLE3","Midjourney","SD21","SDXL","SD3"]

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes,
            yticklabels=classes)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()