#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Resetting project data under: ${PROJECT_ROOT}"

rm -rf "${PROJECT_ROOT}/data/raw/openimages_v7"
rm -rf "${PROJECT_ROOT}/data/metadata"
rm -rf "${PROJECT_ROOT}/reports/error_analysis"
rm -f  "${PROJECT_ROOT}/reports/dataset_stats.csv"
rm -rf "${PROJECT_ROOT}/checkpoints/resnet18_binary_champion"

mkdir -p "${PROJECT_ROOT}/data/raw"
mkdir -p "${PROJECT_ROOT}/data/metadata"
mkdir -p "${PROJECT_ROOT}/reports"
mkdir -p "${PROJECT_ROOT}/checkpoints"

echo "Done."