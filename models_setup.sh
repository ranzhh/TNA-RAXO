#!/bin/bash

# This file takes care of setting up the required environment for raxo2.
# It installs necessary dependencies and configures the system.

# Ensure curl and git are installed
if ! command -v curl &> /dev/null
then
    echo "curl could not be found, please install curl and rerun the script."
    exit 1
fi

if ! command -v git &> /dev/null
then
    echo "git could not be found, please install git and rerun the script."
    exit 1
fi

# Install uv if not already installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed."
fi

# Reload bashrc to ensure uv is in PATH
source ~/.bashrc

# Install required Python packages
uv sync
touch .env

# Download SAM2 and checkpoints
cd ~
git clone https://github.com/facebookresearch/sam2.git
cd sam2
cd checkpoints
./download_ckpts.sh
cd ..
echo "SAM2_PATH=$(pwd)" >> ~/.env

# LEO YOU CAN ADD YOUR STUFF HERE FOR REPLICABILITY, LOOK ABOVE


# ---------------------------------------------
echo "Environment setup complete."
echo "Don't forget to download the datasets and precomputed results as indicated in the README."