# Use NVIDIA TensorRT base image (CUDA 11.8, TensorRT 8.6.1, Ubuntu 22.04)
# Requires: docker login nvcr.io (NGC API key)
FROM nvcr.io/nvidia/tensorrt:23.06-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PIP_NO_CACHE_DIR=1 \
    PYTHONNOUSERSITE=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple

# Install system dependencies (TensorRT libs already present)
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

# Remove any distro-provided numpy to avoid ABI/version conflicts
RUN apt-get purge -y python3-numpy || true \
    && rm -rf /usr/lib/python3/dist-packages/numpy* \
    && rm -rf /usr/local/lib/python3.10/dist-packages/numpy* \
    && rm -rf /usr/local/lib/python3.10/dist-packages/numpy-*.dist-info

# cuDNN / TensorRT are already included in the base image

# Upgrade pip
RUN python -m pip install --upgrade pip

# Constraints to keep numpy/opencv at ABI-safe versions
RUN printf "numpy==1.26.4\nopencv-python-headless==4.8.1.78\n" > /tmp/pip-constraints.txt
ENV PIP_CONSTRAINT=/tmp/pip-constraints.txt

# Pin numpy/opencv first with ignore-installed to override any leftovers
RUN pip install --no-cache-dir --upgrade --ignore-installed \
    numpy==1.26.4 \
    opencv-python-headless==4.8.1.78

# Install PyTorch with CUDA 11.8 wheels (pinned to avoid cu12 split packages)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# Install remaining Python dependencies (TensorRT is provided by base image)
RUN pip install --extra-index-url https://pypi.org/simple \
    librosa \
    tqdm \
    filetype \
    imageio \
    scikit-image \
    cython \
    cuda-python==11.8.* \
    imageio-ffmpeg \
    colored \
    polygraphy \
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
