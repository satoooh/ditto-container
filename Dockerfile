# Use NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    git-lfs \
    vim \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install cuDNN (CUDA 12.x builds ship cuDNN 9)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Ensure pip can fall back to PyPI when NVIDIA's index lacks a dependency
# (e.g., tensorrt_libs -> nvidia-cublas-cu12 is not on https://pypi.nvidia.com)
ENV PIP_EXTRA_INDEX_URL=https://pypi.org/simple

# Install PyTorch with CUDA support first
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install TensorRT-RTX and runtime dependencies
RUN pip install --extra-index-url https://pypi.org/simple \
    "tensorrt-cu12[rtx]>=10.8.0,<11.0.0" \
    librosa \
    tqdm \
    filetype \
    imageio \
    opencv-python-headless \
    scikit-image \
    cython \
    'cuda-python==12.6.*' \
    imageio-ffmpeg \
    colored \
    polygraphy \
    numpy==2.1.1 \
    fastapi \
    uvicorn[standard] \
    websockets \
    python-multipart \
    pyaudio

# Create working directory
WORKDIR /app

# Install git-lfs globally
RUN git lfs install

# Copy the application source into the container
COPY src/ /app/src/

# Create a non-root user
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# Set the default command
CMD ["/bin/bash"] 
