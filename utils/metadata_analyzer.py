from PIL import Image
from PIL.ExifTags import TAGS


# =========================
# C2PA CHECK
# =========================
def check_c2pa(filepath):
    result = {"c2pa": False}

    try:
        with open(filepath, "rb") as f:
            data = f.read()

            signatures = [
                b'c2pa', b'jumb', b'jumd',
                b'c2pa.assertions', b'c2pa.claim'
            ]

            for sig in signatures:
                if sig in data:
                    result["c2pa"] = True
                    result["marker"] = sig.decode("utf-8", errors="ignore")
                    break
    except:
        pass

    return result


# =========================
# EXIF DATA
# =========================
def extract_exif(filepath):
    result = {}

    try:
        img = Image.open(filepath)
        exif_data = img._getexif()

        if exif_data:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                result[tag_name] = str(value)
    except:
        pass

    return result


# =========================
# XMP DATA
# =========================
def extract_xmp(filepath):
    result = {}

    try:
        with open(filepath, "rb") as f:
            data = f.read()

            if b"<x:xmpmeta" in data:
                result["has_xmp"] = True

                start = data.find(b"<x:xmpmeta")
                end = data.find(b"</x:xmpmeta>", start)

                if start != -1 and end != -1:
                    xmp = data[start:end+12].decode("utf-8", errors="ignore")

                    tools = [
                        "DALL-E", "Midjourney", "Stable Diffusion",
                        "Adobe", "Firefly", "Runway"
                    ]

                    for t in tools:
                        if t.lower() in xmp.lower():
                            result["ai_tool"] = t
                            break
    except:
        pass

    return result


# =========================
# MAIN ANALYZER
# =========================
def analyze_metadata(filepath):
    c2pa = check_c2pa(filepath)
    exif = extract_exif(filepath)
    xmp = extract_xmp(filepath)

    return {
        "c2pa_present": c2pa.get("c2pa", False),
        "c2pa_marker": c2pa.get("marker", None),
        "camera": exif.get("Model", None),
        "software": exif.get("Software", None),
        "xmp_present": xmp.get("has_xmp", False),
        "ai_tool": xmp.get("ai_tool", None)
    }