# Use NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

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
    gnupg \
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
    libvpx-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install cuDNN
RUN apt-get update && apt-get install -y \
    libcudnn8=8.9.7.29-1+cuda11.8 \
    libcudnn8-dev=8.9.7.29-1+cuda11.8 \
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
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Add NVIDIA ML repo and install TensorRT 8.6.1 runtime + Python bindings (CUDA 11.8)
RUN wget -qO /usr/share/keyrings/nvidia-ml-keyring.gpg https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2204/x86_64/7fa2af80.pub \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-ml-keyring.gpg] https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libnvinfer8=8.6.1-1+cuda11.8 \
        libnvinfer-plugin8=8.6.1-1+cuda11.8 \
        libnvonnxparsers8=8.6.1-1+cuda11.8 \
        libnvparsers8=8.6.1-1+cuda11.8 \
        libnvinfer-dev=8.6.1-1+cuda11.8 \
        libnvinfer-plugin-dev=8.6.1-1+cuda11.8 \
        python3-libnvinfer=8.6.1-1+cuda11.8 \
    && rm -rf /var/lib/apt/lists/*

# Install remaining Python dependencies (TensorRT is provided by apt above)
RUN pip install --extra-index-url https://pypi.org/simple \
    librosa \
    tqdm \
    filetype \
    imageio \
    opencv-python-headless \
    scikit-image \
    cython \
    cuda-python==12.6.* \
    imageio-ffmpeg \
    colored \
    polygraphy \
    numpy==2.0.1 \
    fastapi \
    uvicorn[standard] \
    websockets \
    python-multipart \
    pyaudio \
    aiortc \
    av \
    aiohttp

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
