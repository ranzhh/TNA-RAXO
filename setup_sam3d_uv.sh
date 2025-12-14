#!/bin/bash
# SAM-3D-Objects Installation Script using uv
# 
# IMPORTANT LIMITATION:
# pytorch3d has NO pre-built Linux wheels and must be compiled from source.
# This requires the CUDA toolkit (nvcc) installed system-wide.
#
# If you don't have nvcc, consider using the conda/mamba script instead
# (setup_sam3d.sh) which bundles CUDA in the environment.
#
# To install CUDA toolkit: sudo apt install nvidia-cuda-toolkit

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAM3D_DIR="${SCRIPT_DIR}/../sam-3d-objects"
VENV_DIR="${SAM3D_DIR}/.venv"

echo "=========================================="
echo "SAM-3D-Objects Installation (uv version)"
echo "=========================================="
echo ""

# Check prerequisites
echo "[0/6] Checking prerequisites..."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check for CUDA compiler (needed for pytorch3d)
HAS_NVCC=false
if command -v nvcc &> /dev/null; then
    HAS_NVCC=true
    echo "Found nvcc: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
else
    echo "WARNING: nvcc not found. Required for compiling pytorch3d."
    echo ""
    read -p "Install CUDA toolkit now? (requires sudo) [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing nvidia-cuda-toolkit..."
        sudo apt update && sudo apt install -y nvidia-cuda-toolkit
        if command -v nvcc &> /dev/null; then
            HAS_NVCC=true
            echo "CUDA toolkit installed successfully!"
        else
            echo "ERROR: CUDA toolkit installation failed."
        fi
    else
        echo "Skipping CUDA toolkit installation."
        echo "pytorch3d may not be available without nvcc."
    fi
    echo ""
fi

# Check for GCC (needed for some Python packages)
if ! command -v gcc &> /dev/null; then
    echo "WARNING: gcc not found. Some packages may fail to install."
    echo "Install with: sudo apt install build-essential"
fi

# Step 1: Clone SAM-3D-Objects repository
echo ""
echo "[1/6] Checking SAM-3D-Objects repository..."
if [ ! -d "${SAM3D_DIR}" ]; then
    echo "Cloning SAM-3D-Objects repository..."
    git clone https://github.com/facebookresearch/sam-3d-objects.git "${SAM3D_DIR}"
else
    echo "SAM-3D-Objects repository already exists at ${SAM3D_DIR}"
fi

# Step 2: Create virtual environment with uv
echo ""
echo "[2/6] Creating virtual environment with uv..."
cd "${SAM3D_DIR}"

if [ -d "${VENV_DIR}" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "${VENV_DIR}"
fi

# Use Python 3.10 - has pre-built pytorch3d wheels (3.11 does not)
uv venv "${VENV_DIR}" --python 3.10

# Activation helper for running commands
UV_RUN="uv run --directory ${SAM3D_DIR}"

# Step 3: Install PyTorch with CUDA 12.1
echo ""
echo "[3/6] Installing PyTorch with CUDA 12.1..."

uv pip install --directory "${SAM3D_DIR}" \
    torch==2.5.1+cu121 \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Step 4: Install SAM-3D-Objects packages
echo ""
echo "[4/6] Installing SAM-3D-Objects packages..."

# Create a filtered requirements file (skip nvidia-pyindex and bpy which have compatibility issues)
grep -v -E 'nvidia-pyindex|bpy' "${SAM3D_DIR}/requirements.txt" > "${SAM3D_DIR}/requirements.uv.txt"

# Install dependencies from filtered requirements
uv pip install --directory "${SAM3D_DIR}" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://pypi.ngc.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r "${SAM3D_DIR}/requirements.uv.txt"

# Install the sam3d_objects package itself (editable, no deps since we installed them)
uv pip install --directory "${SAM3D_DIR}" -e . --no-deps

# Install dev dependencies
uv pip install --directory "${SAM3D_DIR}" pytest findpydeps pipdeptree lovely_tensors

# Install pytorch3d - try pre-built wheel first (available for Python 3.10)
echo "Installing pytorch3d..."
# Try pre-built wheel first (Python 3.10 + CUDA 12.1 + PyTorch 2.5)
if uv pip install --directory "${SAM3D_DIR}" \
    pytorch3d \
    --find-links https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/download.html 2>/dev/null; then
    echo "pytorch3d installed from pre-built wheel!"
elif [ "$HAS_NVCC" = true ]; then
    echo "Pre-built wheel not found. Compiling pytorch3d from source (this may take 10-20 minutes)..."
    # Set CUDA architecture for compilation
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
    uv pip install --directory "${SAM3D_DIR}" \
        "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git" \
        --no-build-isolation 2>&1 || {
        echo ""
        echo "WARNING: pytorch3d compilation failed."
        echo "Some 3D features will be unavailable."
        echo ""
    }
else
    echo "SKIPPING pytorch3d (no pre-built wheel found and no nvcc available)"
    echo "To install later with nvcc: pip install 'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git'"
fi

# Install flash-attn - also benefits from nvcc but has some pre-built wheels
echo "Installing flash-attn..."
if ! uv pip install --directory "${SAM3D_DIR}" flash-attn --no-build-isolation 2>/dev/null; then
    if ! uv pip install --directory "${SAM3D_DIR}" flash-attn 2>/dev/null; then
        echo "WARNING: flash-attn not available - attention will use standard implementation."
    fi
fi

# Install kaolin from NVIDIA's pre-built wheels
echo "Installing kaolin..."
uv pip install --directory "${SAM3D_DIR}" \
    kaolin \
    --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html

# Install gsplat - try pre-built first, skip if fails (optional for some features)
echo "Installing gsplat..."
uv pip install --directory "${SAM3D_DIR}" gsplat \
    || echo "WARNING: gsplat installation failed. Some features may not work."

# Step 5: Install additional dependencies
echo ""
echo "[5/6] Installing additional dependencies..."

uv pip install --directory "${SAM3D_DIR}" \
    utils3d \
    transformers \
    seaborn \
    gradio \
    segment-anything

# Apply Hydra patches
if [ -f "patching/hydra" ]; then
    echo "Applying Hydra patches..."
    chmod +x patching/hydra
    ${UV_RUN} ./patching/hydra || echo "Hydra patching failed, continuing..."
fi

# Step 6: Download checkpoints
echo ""
echo "[6/6] Downloading model checkpoints..."

uv pip install --directory "${SAM3D_DIR}" 'huggingface-hub[cli]<1.0'

# Check login status
if ! ${UV_RUN} huggingface-cli whoami &> /dev/null 2>&1; then
    echo ""
    echo "=========================================="
    echo "Hugging Face Login Required"
    echo "=========================================="
    echo "Please login to Hugging Face."
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo ""
    ${UV_RUN} huggingface-cli login
fi

TAG="hf"
CHECKPOINT_DIR="${SAM3D_DIR}/checkpoints"

echo "Downloading checkpoints to ${CHECKPOINT_DIR}/${TAG}..."
mkdir -p "${CHECKPOINT_DIR}"

${UV_RUN} huggingface-cli download \
    --repo-type model \
    --local-dir "${CHECKPOINT_DIR}/${TAG}-download" \
    --max-workers 1 \
    facebook/sam-3d-objects

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
echo "Python: $(${UV_RUN} python --version)"
echo "PyTorch: $(${UV_RUN} python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(${UV_RUN} python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "Checkpoints: ${CHECKPOINT_DIR}/${TAG}/"
echo ""
echo "To activate the environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Or use uv run:"
echo "  cd ${SAM3D_DIR}"
echo "  uv run python demo.py"
echo ""
