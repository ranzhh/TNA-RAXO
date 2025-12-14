#!/bin/bash
# Download SAM-3D-Objects model checkpoints from Hugging Face

set -e

echo "=========================================="
echo "Downloading SAM-3D-Objects Checkpoints"
echo "=========================================="

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Login to Hugging Face (you'll need a token)
echo ""
echo "Please make sure you're logged in to Hugging Face:"
echo "Run: huggingface-cli login"
echo ""
read -p "Have you logged in? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please run: huggingface-cli login"
    echo "Then run this script again"
    exit 1
fi

# Download from Hugging Face
TAG=hf
echo "Downloading to checkpoints/${TAG}-download..."

huggingface-cli download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects

# Move files to correct location
echo "Moving checkpoints to checkpoints/${TAG}..."
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download

echo ""
echo "=========================================="
echo "Checkpoint download complete!"
echo "=========================================="
echo "Checkpoints are in: checkpoints/${TAG}/"
echo ""

# Also download SAM checkpoint if needed
echo "Downloading SAM checkpoint..."
mkdir -p checkpoints/sam

SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_PATH="checkpoints/sam/sam_vit_h_4b8939.pth"

if [ ! -f "$SAM_PATH" ]; then
    wget -O "$SAM_PATH" "$SAM_URL"
    echo "SAM checkpoint downloaded to $SAM_PATH"
else
    echo "SAM checkpoint already exists at $SAM_PATH"
fi

echo ""
echo "All checkpoints ready!"