#!/bin/bash
# Run the complete SOC Auto data, training, tuning, and evaluation pipeline.
set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "BƯỚC 1: Sinh dữ liệu"
echo "=========================================="
python scripts/generate_sample_data.py

echo "=========================================="
echo "BƯỚC 2: Feature engineering"
echo "=========================================="
python -m src.features

echo "=========================================="
echo "BƯỚC 3: Train model"
echo "=========================================="
python -m src.train

echo "=========================================="
echo "BƯỚC 4: Tune threshold"
echo "=========================================="
python scripts/tune_threshold.py

echo "=========================================="
echo "BƯỚC 5: Đánh giá"
echo "=========================================="
python -m src.evaluate

echo "=========================================="
echo "KẾT QUẢ ĐÁNH GIÁ"
echo "=========================================="
cat outputs/evaluation.json

echo "=========================================="
echo "PIPELINE HOÀN TẤT"
echo "=========================================="