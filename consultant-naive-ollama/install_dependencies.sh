#!/bin/bash
#
# Install all Qdrant and Sentence Transformers dependencies
# This script is used as a workaround when Docker build fails due to network timeout
#
# Usage: docker exec naive-ollma-gpu-consultant bash /app/install_dependencies.sh
#
# Author: AI Assistant
# Date: 2025-12-19
#

set -e  # Exit on error

echo "================================================================================"
echo "INSTALLING QDRANT AND SENTENCE TRANSFORMERS DEPENDENCIES"
echo "================================================================================"

# Step 1: Install Qdrant core dependencies
echo ""
echo "[1/10] Installing Qdrant core dependencies..."
pip install --no-cache-dir \
    protobuf>=5.26.0 \
    grpcio>=1.60.0 \
    grpcio-tools>=1.60.0 \
    httpx>=0.25.0 \
    'portalocker>=2.7.0,<3.0.0'

# Step 2: Install Qdrant client
echo ""
echo "[2/10] Installing Qdrant client..."
pip install --no-cache-dir qdrant-client>=1.7.0

# Step 3: Install Sentence Transformers dependencies
echo ""
echo "[3/10] Installing Sentence Transformers dependencies..."
pip install --no-cache-dir \
    huggingface-hub>=0.20.0 \
    'Pillow>=10.0.0' \
    safetensors>=0.4.0 \
    tokenizers==0.20.3 \
    'transformers>=4.41.0,<5.0.0'

# Step 4: Install Sentence Transformers
echo ""
echo "[4/10] Installing Sentence Transformers..."
pip install --no-cache-dir sentence-transformers>=2.2.0

# Step 5: Install PyTorch CUDA support
echo ""
echo "[5/10] Installing PyTorch CUDA support..."
pip install --no-cache-dir \
    sympy>=1.12 \
    jinja2>=3.1.0 \
    networkx>=3.0

# Step 6-9: Install NVIDIA CUDA libraries (in smaller batches to avoid timeout)
echo ""
echo "[6/10] Installing NVIDIA CUDA libraries (batch 1/4)..."
pip install --no-cache-dir \
    nvidia-cublas-cu12==12.1.3.1 \
    nvidia-cuda-cupti-cu12==12.1.105 \
    nvidia-cuda-nvrtc-cu12==12.1.105

echo ""
echo "[7/10] Installing NVIDIA CUDA libraries (batch 2/4)..."
pip install --no-cache-dir \
    nvidia-cuda-runtime-cu12==12.1.105 \
    nvidia-cudnn-cu12==9.1.0.70 \
    nvidia-cufft-cu12==11.0.2.54

echo ""
echo "[8/10] Installing NVIDIA CUDA libraries (batch 3/4)..."
pip install --no-cache-dir \
    nvidia-curand-cu12==10.3.2.106 \
    nvidia-cusolver-cu12==11.4.5.107 \
    nvidia-cusparse-cu12==12.1.0.106

echo ""
echo "[9/10] Installing NVIDIA CUDA libraries (batch 4/4)..."
pip install --no-cache-dir \
    nvidia-nccl-cu12==2.20.5 \
    nvidia-nvtx-cu12==12.1.105 \
    triton==3.0.0

# Step 10: Install additional dependencies
echo ""
echo "[10/10] Installing additional dependencies..."
pip install --no-cache-dir \
    'tqdm>4' \
    'distro>=1.7.0,<2'

# Verification
echo ""
echo "================================================================================"
echo "VERIFYING INSTALLATION"
echo "================================================================================"

python3 -c "
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
print('✅ Qdrant client imported successfully')
print('✅ Sentence Transformers imported successfully')
print('')
print('All dependencies installed successfully!')
"

echo "================================================================================"
echo "✅ INSTALLATION COMPLETE"
echo "================================================================================"

