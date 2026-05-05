#!/bin/bash
set -e

DEST_DIR="/mnt/data1/persistent-gpu-server/_deps/mathdx"

if [ -d "$DEST_DIR" ] && [ -f "$DEST_DIR/include/nvcompdx.hpp" ]; then
    echo "nvCOMPDx (MathDx) already exists at $DEST_DIR. Skipping download."
    exit 0
fi

echo "MathDx not found. Downloading to $DEST_DIR..."
mkdir -p $DEST_DIR

# Using the public NVIDIA developer redist repository for MathDx (CUDA 12)
# If this URL 404s, NVIDIA has updated the version. 
URL="https://developer.download.nvidia.com/compute/mathdx/redist/mathdx/linux-x86_64/mathdx-linux-x86_64-24.04.1.tar.gz"
# Another possible path
URL2="https://developer.download.nvidia.com/compute/mathdx/redist/mathdx/linux-x86_64/mathdx-linux-x86_64-23.10.0-cuda12.tar.gz"

echo "Attempting to download MathDx from NVIDIA..."
TMP_TAR="/tmp/mathdx.tar.gz"

if wget -q -O $TMP_TAR $URL; then
    echo "Downloaded successfully."
elif wget -q -O $TMP_TAR $URL2; then
    echo "Downloaded successfully (fallback URL)."
else
    echo "ERROR: Failed to download MathDx from public NVIDIA repositories."
    echo "Please manually download the MathDx package for CUDA 12 from https://developer.nvidia.com/mathdx"
    echo "and extract it to $DEST_DIR"
    exit 1
fi

echo "Extracting to $DEST_DIR..."
tar -xzf $TMP_TAR -C $DEST_DIR --strip-components=1
rm -f $TMP_TAR

echo "nvCOMPDx installed successfully at $DEST_DIR"
