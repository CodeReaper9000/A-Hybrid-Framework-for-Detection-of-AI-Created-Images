from torchvision import transforms
from PIL import Image
import torch
import io, random

# =========================
# 🔹 JPEG COMPRESSION
# =========================
def jpeg_compress(img):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=random.randint(20, 80))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


# =========================
# 🔹 ADD SAFE NOISE
# =========================
def add_noise(x):
    noise = 0.05 * torch.randn_like(x)
    x = x + noise
    return torch.clamp(x, 0.0, 1.0)  # 🔥 prevents NaN


# =========================
# 🔹 RESNET TRAIN
# =========================
resnet_train = transforms.Compose([
    transforms.Resize((256,256)),

    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),

    # 🔥 Blur (simulate camera)
    transforms.RandomApply([
        transforms.GaussianBlur(3)
    ], p=0.3),

    # 🔥 Lighting variation
    transforms.RandomApply([
        transforms.ColorJitter(0.3, 0.3, 0.3)
    ], p=0.5),

    # 🔥 Compression (VERY IMPORTANT)
    transforms.RandomApply([
        transforms.Lambda(jpeg_compress)
    ], p=0.7),

    transforms.ToTensor(),

    # 🔥 Noise (CRITICAL FIX)
    transforms.RandomApply([
        transforms.Lambda(add_noise)
    ], p=0.4),

    # 🔥 ImageNet normalization
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================
# 🔹 CLIP TRAIN (LIGHT AUG ONLY)
# =========================
clip_train = transforms.Compose([
    transforms.Resize((224,224)),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    # 🔥 MUST match CLIP expectations
    transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]
    )
])


# =========================
# 🔹 RESNET TEST
# =========================
resnet_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================
# 🔹 CLIP TEST
# =========================
clip_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]
    )
])