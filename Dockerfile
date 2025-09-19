# File: Dockerfile
# Purpose: Provide a TensorRT 10.9 Blackwell-ready runtime for Ditto TalkingHead.
# Why: Aligns the container with NVIDIA NGC releases that officially support SM 12.x GPUs.

FROM nvcr.io/nvidia/tensorrt:25.08-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
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

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir \
    --extra-index-url https://pypi.nvidia.com \
    "tensorrt==10.13.3.9" \
    "tensorrt-cu13==10.13.3.9" \
    "tensorrt-cu13-bindings==10.13.3.9" \
    "tensorrt-cu13-libs==10.13.3.9"

ENV PIP_EXTRA_INDEX_URL=https://pypi.org/simple

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

RUN pip install --extra-index-url https://pypi.org/simple \
    librosa \
    tqdm \
    filetype \
    imageio \
    opencv-python-headless \
    scikit-image \
    cython \
    "cuda-python>=12.6,<13" \
    imageio-ffmpeg \
    colored \
    polygraphy \
    numpy==2.1.1 \
    fastapi \
    uvicorn[standard] \
    websockets \
    python-multipart \
    pyaudio

RUN python - <<'PY'
import sysconfig, pathlib, textwrap
purelib = pathlib.Path(sysconfig.get_path('purelib'))
sitecustomize = purelib / 'sitecustomize.py'
shim = textwrap.dedent('''
import sys, importlib

def _ensure_trt_bindings():
    module = sys.modules.get('tensorrt')
    if module is not None and hasattr(module, 'Logger') and hasattr(module, 'OnnxParserFlag'):
        return
    for candidate in (
        'tensorrt_cu13_bindings',
        'tensorrt.tensorrt',
    ):
        try:
            impl = importlib.import_module(candidate)
        except Exception:
            continue
        else:
            sys.modules['tensorrt'] = impl
            return

_ensure_trt_bindings()
''')
if sitecustomize.exists():
    existing = sitecustomize.read_text()
    if existing and not existing.endswith('\n'):
        existing += '\n'
else:
    existing = ''
sitecustomize.write_text(existing + shim)
PY

RUN python - <<'PY'
from importlib.metadata import version
import tensorrt
ver = version('tensorrt')
print('Verified TensorRT version:', ver)
assert ver.startswith('10.13'), ver
import tensorrt as trt
print('Has Logger:', hasattr(trt, 'Logger'))
print('Has OnnxParserFlag:', hasattr(trt, 'OnnxParserFlag'))
PY

WORKDIR /app

RUN git lfs install

COPY src/ /app/src/

RUN if ! getent passwd 1000 >/dev/null; then useradd -m -u 1000 user; fi \
    && chown -R 1000:1000 /app
USER 1000

CMD ["/bin/bash"]
