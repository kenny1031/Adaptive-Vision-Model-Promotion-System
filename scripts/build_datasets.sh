#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# FiftyOne data cache directory
export FIFTYONE_DATA_DIR="${PROJECT_ROOT}/data/fiftyone"

# Python script path
PYTHON_SCRIPT="${PROJECT_ROOT}/scripts/build_openimages_dataset.py"
OUTPUT_ROOT="${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"
echo "FIFTYONE_DATA_DIR: ${FIFTYONE_DATA_DIR}"
echo "Python script: ${PYTHON_SCRIPT}"

mkdir -p "${FIFTYONE_DATA_DIR}"
mkdir -p "${PROJECT_ROOT}/data/raw"
mkdir -p "${PROJECT_ROOT}/data/metadata"

# train : val : test = 70 : 15 : 15 for now
echo "========== Building TRAIN dataset =========="
python "${PYTHON_SCRIPT}" \
  --split train \
  --output-root "${OUTPUT_ROOT}" \
  --max-samples 6000 \
  --shuffle \
  --overwrite \
  --dataset-name open-images-v7-train-content-safety

echo "========== Building VALIDATION dataset =========="
python "${PYTHON_SCRIPT}" \
  --split validation \
  --output-root "${OUTPUT_ROOT}" \
  --max-samples 900 \
  --shuffle \
  --overwrite \
  --dataset-name open-images-v7-validation-content-safety

echo "========== Building TEST dataset =========="
python "${PYTHON_SCRIPT}" \
  --split test \
  --output-root "${OUTPUT_ROOT}" \
  --max-samples 900 \
  --shuffle \
  --overwrite \
  --dataset-name open-images-v7-test-content-safety

echo "========== Done =========="
echo "Raw/remapped data should now be under:"
echo "  ${PROJECT_ROOT}/data/raw/openimages_v7/"
echo "Metadata should now be under:"
echo "  ${PROJECT_ROOT}/data/metadata/"
echo "FiftyOne cache should now be under:"
echo "  ${PROJECT_ROOT}/data/fiftyone/"