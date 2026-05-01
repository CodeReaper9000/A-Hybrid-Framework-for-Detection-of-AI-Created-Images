function triggerFile() {
    document.getElementById("imageInput").click();
}

/* PREVIEW */
document.getElementById("imageInput").addEventListener("change", function () {
    const file = this.files[0];
    if (!file) return;

    const preview = document.getElementById("uploadPreview");
    const content = document.getElementById("uploadContent");

    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
    content.style.display = "none";
});

/* DRAG DROP */
const uploadBox = document.getElementById("uploadBox");

uploadBox.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadBox.style.background = "rgba(79,70,229,0.2)";
});

uploadBox.addEventListener("dragleave", () => {
    uploadBox.style.background = "transparent";
});

uploadBox.addEventListener("drop", (e) => {
    e.preventDefault();
    document.getElementById("imageInput").files = e.dataTransfer.files;
    document.getElementById("imageInput").dispatchEvent(new Event("change"));
});

/* PREDICT */
function uploadImage() {
    const file = document.getElementById("imageInput").files[0];

    if (!file) {
        alert("Upload an image first");
        return;
    }

    document.getElementById("landing").classList.add("hidden");
    document.getElementById("resultPage").classList.remove("hidden");

    document.getElementById("preview").innerHTML =
        `<img src="${URL.createObjectURL(file)}">`;

    const formData = new FormData();
    formData.append("image", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        // ==========================
        // 🔥 METADATA DISPLAY
        // ==========================
        const meta = data.metadata;

        let metaHTML = "";

        if (meta.c2pa_present) {
            metaHTML += `<div>✔ C2PA Signature Detected</div>`;
        }

        if (meta.ai_tool) {
            metaHTML += `<div>✔ AI Tool: ${meta.ai_tool}</div>`;
        }

        if (meta.camera) {
            metaHTML += `<div>📷 Camera: ${meta.camera}</div>`;
        }

        if (meta.software) {
            metaHTML += `<div>🛠 Software: ${meta.software}</div>`;
        }

        if (!metaHTML) {
            metaHTML = `<div>No strong metadata signals found</div>`;
        }

        document.getElementById("metadataBox").innerHTML = metaHTML;
        const realProb = data.multiclass["Real"];
        const aiProb = 1 - realProb;

        const aiPercent = (aiProb * 100).toFixed(1);
        const realPercent = (realProb * 100).toFixed(1);

        // ALERT
        const alertBox = document.getElementById("resultAlert");
        const isAI = aiProb > realProb;

        alertBox.className = "alert-box " + (isAI ? "alert-ai" : "alert-real");

        alertBox.innerText = isAI
            ? "This input is likely AI-generated"
            : "This input appears to be real";

        // BARS
        document.getElementById("aiScore").innerText = aiPercent + "%";
        document.getElementById("realScore").innerText = realPercent + "%";

        document.getElementById("aiBar").style.width = aiPercent + "%";
        document.getElementById("realBar").style.width = realPercent + "%";

        document.getElementById("aiTotal").innerText = aiPercent + "%";

        // SOURCES
        const sources = Object.entries(data.multiclass)
            .filter(([k]) => k !== "Real")
            .sort((a, b) => b[1] - a[1]);

        let tagsHTML = "";

        sources.slice(0, 4).forEach(([key, val]) => {
            tagsHTML += `<div class="tag">${key}: ${(val * 100).toFixed(1)}%</div>`;
        });

        document.getElementById("sourceTags").innerHTML = tagsHTML;
    });
}

/* RESET */
function resetApp() {
    location.reload();
}