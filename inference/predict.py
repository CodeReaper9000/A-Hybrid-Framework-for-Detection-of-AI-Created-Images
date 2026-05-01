import torch
import torch.nn.functional as F
from PIL import Image
import os

from models.fusion_model import FusionModel
from dataset.transforms import resnet_test, clip_test

# =========================
# 🔹 CONFIG
# =========================
from configs.config import FINAL_MODEL, UPLOADS_DIR

MODEL_PATH = FINAL_MODEL
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Real",
    "DALLE3",
    "Midjourney",
    "SD21",
    "SDXL",
    "SD3"
]

# =========================
# 🔹 LOAD MODEL
# =========================
model = FusionModel(num_classes=6).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded successfully!")


# =========================
# 🔹 METADATA ANALYSIS
# =========================
def check_metadata(filepath):
    try:
        with open(filepath, "rb") as f:
            data = f.read()

        # 🔹 C2PA check
        if b'c2pa' in data:
            return "AI (C2PA detected)", 1.0

        # 🔹 XMP check
        if b"<x:xmpmeta" in data:
            xmp_str = data.decode("utf-8", errors="ignore").lower()
            tools = ["dall-e", "midjourney", "stable diffusion"]
            for t in tools:
                if t in xmp_str:
                    return f"AI ({t})", 1.0

    except:
        pass

    return None, None


# =========================
# 🔹 PREDICT SINGLE IMAGE
# =========================
def predict_image(image_path):
    print("\n" + "="*60)
    print(f"📷 IMAGE: {image_path}")

    # 🔹 Step 1: Metadata check
    meta_pred, meta_conf = check_metadata(image_path)

    if meta_pred is not None:
        print(f"⚡ Metadata Prediction: {meta_pred}")
        print("="*60)
        return

    # 🔹 Step 2: Load image
    try:
        img = Image.open(image_path).convert("RGB")
    except:
        print("❌ Error loading image")
        return

    # 🔹 Step 3: Apply transforms
    r_img = resnet_test(img).unsqueeze(0).to(DEVICE)
    c_img = clip_test(img).unsqueeze(0).to(DEVICE)

    # 🔹 Step 4: Model inference
    with torch.no_grad():
        outputs = model(r_img, c_img)

        # 🔥 Safety check
        if torch.isnan(outputs).any():
            print("❌ Model output is NaN — skipping")
            return

        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    # 🔹 Step 5: Output
    pred_class = CLASS_NAMES[pred.item()]
    confidence = conf.item()

    print(f"🧠 Model Prediction: {pred_class}")
    print(f"📊 Confidence: {confidence:.4f}")
    print("="*60)


# =========================
# 🔹 PREDICT FOLDER
# =========================
def predict_folder(folder_path):
    print(f"\n📂 Testing folder: {folder_path}")

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        predict_image(path)


# =========================
# 🔹 MAIN
# =========================
if __name__ == "__main__":
    # 🔥 CHANGE THIS

    IMAGE_PATH = r"C:\Users\swast\OneDrive\Desktop\purulia 25'\lr edit\IMG_8336-2.jpg"
    # FOLDER_PATH = "test_images/"

    if os.path.isfile(IMAGE_PATH):
        predict_image(IMAGE_PATH)
    else:
        print("❌ Image not found")

    # Uncomment for folder testing
    predict_folder(UPLOADS_DIR)