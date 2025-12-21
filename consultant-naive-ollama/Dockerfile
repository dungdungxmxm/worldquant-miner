FROM nvidia/cuda:11.6.2-base-ubuntu20.04

# Set environment variables for CUDA
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /app

# Install system dependencies including Python and CUDA tools
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Step 1: Install core dependencies first (to avoid conflicts)
RUN pip install --no-cache-dir \
    requests>=2.31.0 \
    urllib3 \
    idna \
    certifi \
    charset_normalizer

# Step 2: Install OpenAI with its dependencies
RUN pip install --no-cache-dir openai==1.12.0

# Step 3: Install dependencies in groups to avoid timeout
# Group 1: Core Python packages
RUN pip install --no-cache-dir \
    typing-extensions>=4.9.0 \
    pandas \
    scipy>=1.10.0 \
    scikit-learn \
    joblib>=1.1.1 \
    threadpoolctl>=2.0.0 \
    numpy>=1.24.0 \
    schedule>=1.2.0 \
    flask>=2.3.0

# Group 2: Pydantic
RUN pip install --no-cache-dir \
    pydantic==2.10.6 \
    pydantic-core==2.27.2 \
    annotated-types==0.7.0 \
    python-dotenv>=1.0.0

# Group 3: PyTorch (large download, separate step)
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    torchaudio>=2.0.0

# Group 4: Qdrant dependencies
RUN pip install --no-cache-dir \
    protobuf>=5.26.0 \
    grpcio>=1.60.0 \
    grpcio-tools>=1.60.0 \
    httpx>=0.25.0 \
    'portalocker>=2.7.0,<3.0.0'

# Group 5: Qdrant client
RUN pip install --no-cache-dir qdrant-client>=1.7.0

# Group 6: Sentence Transformers dependencies
RUN pip install --no-cache-dir \
    huggingface-hub>=0.20.0 \
    'Pillow>=10.0.0' \
    safetensors>=0.4.0 \
    tokenizers==0.20.3 \
    'transformers>=4.41.0,<5.0.0'

# Group 7: Sentence Transformers
RUN pip install --no-cache-dir sentence-transformers>=2.2.0

# Group 8: PyTorch CUDA support
RUN pip install --no-cache-dir \
    sympy>=1.12 \
    jinja2>=3.1.0 \
    networkx>=3.0

# Group 9: NVIDIA CUDA libraries (specific versions for torch 2.4.1)
RUN pip install --no-cache-dir \
    nvidia-cublas-cu12==12.1.3.1 \
    nvidia-cuda-cupti-cu12==12.1.105 \
    nvidia-cuda-nvrtc-cu12==12.1.105 \
    nvidia-cuda-runtime-cu12==12.1.105 \
    nvidia-cudnn-cu12==9.1.0.70 \
    nvidia-cufft-cu12==11.0.2.54 \
    nvidia-curand-cu12==10.3.2.106 \
    nvidia-cusolver-cu12==11.4.5.107 \
    nvidia-cusparse-cu12==12.1.0.106 \
    nvidia-nccl-cu12==2.20.5 \
    nvidia-nvtx-cu12==12.1.105 \
    triton==3.0.0

# Group 10: Additional dependencies
RUN pip install --no-cache-dir \
    'tqdm>4' \
    'distro>=1.7.0,<2'

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy application code
COPY . .

# Copy the start scripts and make them executable
COPY start.sh /app/start.sh
COPY start_ollama.sh /app/start_ollama.sh
RUN chmod +x /app/start.sh /app/start_ollama.sh

# Expose Ollama port
EXPOSE 11434

# Set the entrypoint (will be overridden by docker-compose)
ENTRYPOINT ["/app/start.sh"]
