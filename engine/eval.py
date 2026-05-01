import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc="🔵 Evaluating", leave=False)

    with torch.no_grad():
        for r_img, c_img, labels in loop:
            r_img = r_img.to(device)
            c_img = c_img.to(device)

            outputs = model(r_img, c_img)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # ✅ Metrics (safe version)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, prec, rec, f1, cm