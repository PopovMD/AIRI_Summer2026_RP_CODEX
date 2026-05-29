#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-codex_gears_py311}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "[INFO] Setting up conda environment: ${ENV_NAME} with Python ${PYTHON_VERSION}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not available in PATH."
  echo "Run:"
  echo "  source ~/miniconda3/etc/profile.d/conda.sh"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Conda environment '${ENV_NAME}' already exists."
  echo "[INFO] Reusing it. To recreate from scratch, run:"
  echo "       conda env remove -n ${ENV_NAME}"
else
  conda create -y -n "${ENV_NAME}" -c conda-forge "python=${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"

echo "[INFO] Installing scientific stack from conda-forge..."
# Use conda-forge for compiled/scientific packages on old Ubuntu/GCC.
# This avoids building NumPy/SciPy from source and also avoids the pip --only-binary/asciitree conflict.
conda install -y -c conda-forge \
  "numpy=1.26.4" \
  "pandas=2.2.2" \
  "scipy=1.14.1" \
  "scikit-learn=1.5.1" \
  "scanpy=1.10.2" \
  "anndata=0.10.9" \
  "zarr=2.18.7" \
  "numcodecs<0.16" \
  "networkx=3.3" \
  "tqdm=4.66.5" \
  "dcor=0.6"

python -m pip install --upgrade pip setuptools wheel

echo "[INFO] Installing PyTorch..."
# Override TORCH_INSTALL if you need a specific CUDA/CPU wheel.
# Examples:
#   TORCH_INSTALL='python -m pip install torch --index-url https://download.pytorch.org/whl/cu121' bash setup_codex_conda_py311_v2.sh
#   TORCH_INSTALL='python -m pip install torch --index-url https://download.pytorch.org/whl/cpu' bash setup_codex_conda_py311_v2.sh
TORCH_INSTALL="${TORCH_INSTALL:-python -m pip install torch}"
eval "${TORCH_INSTALL}"

echo "[INFO] Installing torch_geometric..."
python -m pip install torch_geometric

echo "[INFO] Installing cell-gears without dependencies..."
# Important: do not let cell-gears pull a different Scanpy/NumPy/Zarr stack.
python -m pip install --no-deps cell-gears==0.1.2

echo "[INFO] Verifying environment..."
python - <<'PY'
import sys
import numpy, pandas, scipy, sklearn, scanpy, anndata, zarr, numcodecs
from gears import PertData

print("[OK] Python:", sys.version.replace("\n", " "))
print("[OK] numpy:", numpy.__version__)
print("[OK] pandas:", pandas.__version__)
print("[OK] scipy:", scipy.__version__)
print("[OK] sklearn:", sklearn.__version__)
print("[OK] scanpy:", scanpy.__version__)
print("[OK] anndata:", anndata.__version__)
print("[OK] zarr:", zarr.__version__)
print("[OK] numcodecs:", numcodecs.__version__)
print("[OK] gears import works")
PY

echo
echo "[DONE] Environment is ready."
echo "Activate it with:"
echo "  conda activate ${ENV_NAME}"
