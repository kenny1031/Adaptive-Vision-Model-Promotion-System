#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_SCRIPT="${PROJECT_ROOT}/scripts/build_suspicious_candidates.py"

echo "Project root: ${PROJECT_ROOT}"
echo "Python script: ${PYTHON_SCRIPT}"

mkdir -p "${PROJECT_ROOT}/data/raw/openimages_v7/test/remapped/suspicious_candidates"
mkdir -p "${PROJECT_ROOT}/data/raw/openimages_v7/validation/remapped/suspicious_candidates"

echo "========== Building Train suspicious candidates =========="
python "${PYTHON_SCRIPT}" \
  --predictions-csv "${PROJECT_ROOT}/reports/error_analysis/train_predictions.csv" \
  --output-dir "${PROJECT_ROOT}/data/raw/openimages_v7/train/remapped/suspicious_candidates" \
  --include-fp \
  --include-low-confidence \
  --include-fn-below-conf 0.55 \
  --max-files 300

echo "========== Building TEST suspicious candidates =========="
python "${PYTHON_SCRIPT}" \
  --predictions-csv "${PROJECT_ROOT}/reports/error_analysis/test_predictions.csv" \
  --output-dir "${PROJECT_ROOT}/data/raw/openimages_v7/test/remapped/suspicious_candidates" \
  --include-fp \
  --include-low-confidence \
  --include-fn-below-conf 0.55 \
  --max-files 150

echo "========== Building VALIDATION suspicious candidates =========="
python "${PYTHON_SCRIPT}" \
  --predictions-csv "${PROJECT_ROOT}/reports/error_analysis/validation_predictions.csv" \
  --output-dir "${PROJECT_ROOT}/data/raw/openimages_v7/validation/remapped/suspicious_candidates" \
  --include-fp \
  --include-low-confidence \
  --include-fn-below-conf 0.55 \
  --max-files 120

echo "========== Done =========="
echo "Check candidate folders:"
echo "  ${PROJECT_ROOT}/data/raw/openimages_v7/test/remapped/suspicious_candidates"
echo "  ${PROJECT_ROOT}/data/raw/openimages_v7/validation/remapped/suspicious_candidates"