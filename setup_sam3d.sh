#!/bin/bash
# SAM-3D-Objects Installation Script
# Follows the official setup.md instructions using conda/mamba

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAM3D_DIR="${SCRIPT_DIR}/../sam-3d-objects"
CONDA_BIN="/home/disi/miniforge3/bin/conda"
MAMBA_BIN="/home/disi/miniforge3/bin/mamba"

echo "=========================================="
echo "SAM-3D-Objects Installation"
echo "=========================================="
echo ""

# Step 1: Clone SAM-3D-Objects repository
echo "[1/5] Checking SAM-3D-Objects repository..."
if [ ! -d "${SAM3D_DIR}" ]; then
    echo "Cloning SAM-3D-Objects repository..."
    git clone https://github.com/facebookresearch/sam-3d-objects.git "${SAM3D_DIR}"
else
    echo "SAM-3D-Objects repository already exists at ${SAM3D_DIR}"
fi

# Step 2: Install miniforge if needed
echo ""
echo "[2/5] Checking conda/mamba..."
if [ ! -f "${CONDA_BIN}" ]; then
    echo "Miniforge not found. Installing..."
    bash "${SCRIPT_DIR}/miniforge.sh" -b -p /home/disi/miniforge3
fi

# Source conda
eval "$('${CONDA_BIN}' 'shell.bash' 'hook')"

# Use mamba if available, otherwise conda
if [ -f "${MAMBA_BIN}" ]; then
    CONDA_CMD="mamba"
else
    CONDA_CMD="conda"
fi
echo "Using ${CONDA_CMD}"

# Step 3: Create conda environment from official default.yml
echo ""
echo "[3/5] Creating sam3d-objects conda environment..."
cd "${SAM3D_DIR}"

# Remove existing environment if it exists
if ${CONDA_CMD} env list | grep -q "sam3d-objects"; then
    echo "Removing existing sam3d-objects environment..."
    ${CONDA_CMD} env remove -n sam3d-objects -y
fi

# Create environment from official YAML
${CONDA_CMD} env create -f environments/default.yml

# Activate environment using conda run for non-interactive scripts
echo ""
echo "Environment created! Using mamba run for remaining steps..."
CONDA_RUN="${MAMBA_BIN} run -n sam3d-objects"

# Verify environment
${CONDA_RUN} python --version
${CONDA_RUN} nvcc --version | head -4

# Step 4: Install SAM-3D-Objects packages (following official setup.md)
echo ""
echo "[4/5] Installing SAM-3D-Objects packages..."

# Set environment variables as per setup.md
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

# Install PyTorch with CUDA 12.1 (must be installed before sam3d-objects)
${CONDA_RUN} pip install torch==2.5.1+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
${CONDA_RUN} pip install -e '.[dev]'
${CONDA_RUN} pip install -e '.[p3d]'  # pytorch3d dependency fix

# Install inference dependencies
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
${CONDA_RUN} pip install -e '.[inference]'

# Install utils3d (required by notebook/inference.py but missing from requirements)
${CONDA_RUN} pip install utils3d

# Install transformers (required for CLIP-based caption matching)
${CONDA_RUN} pip install transformers

# Install additional dependencies for inference.py
${CONDA_RUN} pip install seaborn gradio

# Apply Hydra patches
if [ -f "patching/hydra" ]; then
    echo "Applying Hydra patches..."
    chmod +x patching/hydra
    ${CONDA_RUN} ./patching/hydra || echo "Hydra patching failed, continuing..."
fi

# Step 5: Download checkpoints
echo ""
echo "[5/5] Downloading model checkpoints..."

${CONDA_RUN} pip install 'huggingface-hub[cli]<1.0'

# Check login status
if ! ${CONDA_RUN} huggingface-cli whoami &> /dev/null 2>&1; then
    echo ""
    echo "=========================================="
    echo "Hugging Face Login Required"
    echo "=========================================="
    echo "Please login to Hugging Face."
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo ""
    ${CONDA_RUN} huggingface-cli login
fi

TAG="hf"
CHECKPOINT_DIR="${SAM3D_DIR}/checkpoints"

echo "Downloading checkpoints to ${CHECKPOINT_DIR}/${TAG}..."
mkdir -p "${CHECKPOINT_DIR}"

${CONDA_RUN} huggingface-cli download \
    --repo-type model \
    --local-dir "${CHECKPOINT_DIR}/${TAG}-download" \
    --max-workers 1 \
    nvidia/SAM-3D-Objects

# Organize checkpoints
if [ -d "${CHECKPOINT_DIR}/${TAG}-download/checkpoints" ]; then
    mv "${CHECKPOINT_DIR}/${TAG}-download/checkpoints" "${CHECKPOINT_DIR}/${TAG}"
    rm -rf "${CHECKPOINT_DIR}/${TAG}-download"
else
    mv "${CHECKPOINT_DIR}/${TAG}-download" "${CHECKPOINT_DIR}/${TAG}"
fi

cd "${SCRIPT_DIR}"

echo ""
echo "=========================================="
echo "Installation Complete! ✓"
echo "=========================================="
echo ""
echo "Python: $(${CONDA_RUN} python --version)"
echo "PyTorch: $(${CONDA_RUN} python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(${CONDA_RUN} python -c 'import torch; print(torch.cuda.is_available())')"
echo "mamba-ssm: $(${CONDA_RUN} python -c 'import mamba_ssm; print(\"installed\")' 2>/dev/null || echo 'not installed')"
echo ""
echo "Checkpoints: ${CHECKPOINT_DIR}/${TAG}/"
echo ""
echo "To activate the environment:"
echo "  ${CONDA_CMD} activate sam3d-objects"
echo ""
echo "To test SAM-3D-Objects:"
echo "  cd ${SAM3D_DIR}"
echo "  ${CONDA_RUN} python demo.py"
echo ""
