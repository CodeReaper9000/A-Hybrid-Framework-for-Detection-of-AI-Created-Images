import torch
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="🟢 Training", leave=False)

    for r_img, c_img, labels in loop:
        try:
            r_img = r_img.to(device)
            c_img = c_img.to(device)
            labels = labels.to(device)

            # 🔥 Skip bad inputs
            if torch.isnan(r_img).any() or torch.isnan(c_img).any():
                print("⚠️ Skipping batch (NaN in input)")
                continue

            optimizer.zero_grad()

            outputs = model(r_img, c_img)

            # 🔥 Skip bad outputs
            if torch.isnan(outputs).any():
                print("⚠️ Skipping batch (NaN in output)")
                continue

            loss = criterion(outputs, labels)

            # 🔥 Skip NaN loss
            if torch.isnan(loss):
                print("⚠️ Skipping batch (NaN loss)")
                continue

            loss.backward()

            # 🔥 Gradient clipping (VERY IMPORTANT)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            acc = correct / total if total > 0 else 0

            loop.set_postfix(loss=loss.item(), acc=acc)

        except Exception as e:
            print(f"⚠️ Skipping batch due to error: {e}")
            continue

    avg_loss = total_loss / max(len(loader), 1)
    epoch_acc = correct / total if total > 0 else 0

    return avg_loss, epoch_acc