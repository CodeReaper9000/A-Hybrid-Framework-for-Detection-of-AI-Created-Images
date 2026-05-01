import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset (training data)
DATASET_DIR = os.path.join(BASE_DIR, "VISION+defactify")

# Benchmark dataset
BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmark_export")

# Models
BEST_MODEL = os.path.join(BASE_DIR, "best_model.pth")
FINAL_MODEL = os.path.join(BASE_DIR, "final_model.pth")

# Uploads folder
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")