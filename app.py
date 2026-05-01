from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import os
from models.fusion_model import FusionModel
from dataset.transforms import resnet_test, clip_test
from utils.metadata_analyzer import analyze_metadata

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# LOAD MODEL
# ==========================
print("🔄 Loading model...")
model = FusionModel(num_classes=6).to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()
print("✅ Model loaded")

IDX_TO_CLASS = {
    0: "Real",
    1: "DALLE3",
    2: "Midjourney",
    3: "SD21",
    4: "SDXL",
    5: "SD3"
}

# ==========================
# ROUTES
# ==========================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("📥 Request received")

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"})

        file = request.files["image"]
        print("📁 File:", file.filename)

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        # =========================
        # 🔥 STEP 1: METADATA FIRST
        # =========================
        metadata = analyze_metadata(path)
        print("🧾 Metadata:", metadata)

        # =========================
        # 🔥 STEP 2: MODEL
        # =========================
        img = Image.open(path).convert("RGB")

        r_img = resnet_test(img).unsqueeze(0).to(DEVICE)
        c_img = clip_test(img).unsqueeze(0).to(DEVICE)

        print("✅ Transforms done")

        with torch.no_grad():
            outputs = model(r_img, c_img)

            # 🔥 Safety check
            if torch.isnan(outputs).any():
                return jsonify({"error": "Model produced NaN outputs"})

            probs = torch.softmax(outputs, dim=1)[0]

        print("✅ Model inference done")

        probs = probs.cpu().numpy()

        # =========================
        # 🔥 MULTICLASS
        # =========================
        multiclass = {
            IDX_TO_CLASS[i]: float(probs[i])
            for i in range(6)
        }

        # =========================
        # 🔥 BINARY
        # =========================
        real_prob = float(probs[0])
        ai_prob = float(probs[1:].sum())

        binary = "AI" if ai_prob > real_prob else "REAL"
        confidence = float(max(ai_prob, real_prob))

        # =========================
        # 🔥 RESPONSE
        # =========================
        return jsonify({
            "binary": binary,
            "confidence": round(confidence, 4),
            "multiclass": multiclass,
            "metadata": metadata   # 🔥 NEW FIELD
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)