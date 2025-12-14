#!/bin/bash

# This file takes care of setting up the required environment for raxo2.
# It installs necessary dependencies and configures the system.

WORK_DIR=$(pwd)

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
touch $WORK_DIR/.env

# Install Blender from source if not already installed. Using version 4.4.3 as an example, make sure it's aligned with your BPY version!
if ! command -v blender &> /dev/null
then
    echo "Blender could not be found, installing..."
    mkdir -p $HOME/.local/share
    mkdir -p $HOME/.local/bin
    curl -L https://download.blender.org/release/Blender4.4/blender-4.4.3-linux-x64.tar.xz | tar -xJ -C $HOME/.local/share/
    mv $HOME/.local/share/blender-4.4.3-linux-x64/ $HOME/.local/share/blender

    # Create a symlink to make blender accessible from anywhere
    ln -s $HOME/.local/share/blender/blender $HOME/.local/bin/blender
    
    # Ensure ~/.local/bin is in PATH
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        export PATH="$HOME/.local/bin:$PATH"
    fi
else
    echo "Blender is already installed." 

fi

# Download SAM2 and checkpoints
if [ ! -d "$HOME/sam2" ]; then
    echo "SAM2 not found, cloning and downloading checkpoints..."
    cd ~
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2
    cd checkpoints
    ./download_ckpts.sh
    cd ..
    echo "SAM2_PATH=$(pwd)" >> ~$WORK_DIR/.env
else
    echo "SAM2 is already installed."
fi


# LEO YOU CAN ADD YOUR STUFF HERE FOR REPLICABILITY, LOOK ABOVE


# ---------------------------------------------
echo "Environment setup complete."
echo "Don't forget to download the datasets and precomputed results as indicated in the README."
echo "Also, make sure to set the correct paths in the .env file located at $WORK_DIR/.env"
# ---------------------------------------------
