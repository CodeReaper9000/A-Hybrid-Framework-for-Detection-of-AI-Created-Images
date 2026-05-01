import os
from PIL import Image
from torch.utils.data import Dataset

class MultiClassImageDataset(Dataset):
    def __init__(self, root_dir, resnet_transform=None, clip_transform=None):
        self.samples = []
        self.resnet_transform = resnet_transform
        self.clip_transform = clip_transform

        # ✅ Merge multiple REAL folders into one class
        self.class_to_idx = {
            "SCIMD-6": 0,
            "images": 0,
            "RealWorld": 0,
            "realwhatsapp": 0,
            "DALLE3": 1,
            "Midjourney": 2,
            "SD21": 3,
            "SDXL": 4,
            "SD3": 5
        }

        self.idx_to_class = {
            0: "Real",
            1: "DALLE3",
            2: "Midjourney",
            3: "SD21",
            4: "SDXL",
            5: "SD3"
        }

        # 🔍 Load dataset
        for folder, label in self.class_to_idx.items():
            folder_path = os.path.join(root_dir, folder)

            if not os.path.exists(folder_path):
                print(f"⚠️ Skipping missing folder: {folder}")
                continue

            count = 0

            for file in os.listdir(folder_path):
                path = os.path.join(folder_path, file)

                if path.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((path, label))
                    count += 1

            print(f"📂 Loaded {count} images from {folder} → class {label}")

        print(f"\n✅ Total dataset size: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Error loading image: {path}")
            # fallback: return a blank image
            img = Image.new("RGB", (224, 224))

        # Apply transforms
        if self.resnet_transform:
            r_img = self.resnet_transform(img)
        else:
            r_img = img

        if self.clip_transform:
            c_img = self.clip_transform(img)
        else:
            c_img = img

        return r_img, c_img, label