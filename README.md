# Ditto TalkingHead Docker Container

Dockerized environment for the Ditto Talking Head project (Motion‑Space Diffusion for controllable, realtime talking head synthesis).

This repo does not use git submodules. The `src/` directory lives in this repository and is copied into the image (or mounted via Compose) as-is.

## Prerequisites
- Docker with GPU support and NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support (Ampere or newer recommended)
- Docker Compose v2 (optional but recommended)

## Quick Start

### 1) Clone
```bash
git clone https://github.com/your-username/ditto-container.git
cd ditto-container
```

### 2) Build (choose one)
- Compose (recommended for development):
  ```bash
  docker compose up -d --build
  # or: docker-compose up -d --build
  ```
- Plain Docker:
  ```bash
  docker build -t ditto-talkinghead .
  ```

### 3) Run (if you built with plain Docker)
```bash
docker run -d -it --gpus all \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  -p 8000:8000 \
  --restart unless-stopped \
  --name ditto-container \
  ditto-talkinghead bash -lc 'sleep infinity'
```

### Common Ops
- Enter container (Compose v2): `docker compose exec ditto-talkinghead bash`
- Enter container (Compose v1): `docker-compose exec ditto-talkinghead bash`
- Stop (Compose v2): `docker compose down`

## What’s in the Container
- Base: NVIDIA CUDA 11.8 (Ubuntu 22.04) + cuDNN 8
- Python 3.10, PyTorch (cu118), TensorRT 8.6.1
- Tools: git, git-lfs, ffmpeg, vim
- App source: `src/` → `/app/src` (copied for Docker build, mounted in Compose)

## Directory Layout
```
./
├── src/          # App source; copied to /app/src (docker) or mounted (compose)
├── checkpoints/  # Model checkpoints -> /app/checkpoints
├── data/         # Input data        -> /app/data
├── output/       # Generated output  -> /app/output
└── docker files
```

### Build/Run Diagram
```mermaid
flowchart LR
  subgraph Host
    A[src/]:::code
    B[checkpoints/]
    C[data/]
    D[output/]
  end
  subgraph Image
    I[/app/src]:::code
  end
  subgraph Container
    X[/app/src]:::code
    Y[/app/checkpoints]
    Z[/app/data]
    O[/app/output]
  end

  A -- docker build --> I
  A -- compose mount --- X
  B --- Y
  C --- Z
  D --- O

  classDef code fill:#eef,stroke:#88f
```

## Set Up Models and Try Inference
Inside the container:
```bash
cd /app
git lfs install
git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints

cd /app/src
python inference.py \
  --data_root "/app/checkpoints/ditto_trt_Ampere_Plus" \
  --cfg_pkl "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
  --audio_path "/app/data/audio.wav" \
  --source_path "/app/data/source_image.png" \
  --output_path "/app/output/result.mp4"
```

## Realtime Streaming (WebSocket + FastAPI)
Inside the container:
```bash
cd /app/src

# Start the streaming server
python streaming_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --cfg_pkl "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl" \
  --data_root "/app/checkpoints/ditto_trt_Ampere_Plus"

# Simple test client
python streaming_client.py \
  --server ws://localhost:8000 \
  --client_id test_client \
  --audio_path /app/src/example/audio.wav \
  --source_path /app/src/example/image.png \
  --timeout 60
```
From your browser: `http://localhost:8000/demo` (or `http://YOUR_SERVER_IP:8000/demo`).

Notes:
- Put a source image at `/app/data/source_image.png`.
- The demo page infers WebSocket URL from the browser location.
- See `src/STREAMING_SETUP.md` and `src/STREAMING_OPTIMIZATIONS.md` for more.

## Working with Source Code (No Submodules)
`src/` lives in this repo. Edit and commit directly:
```bash
git add src
git commit -m "Update app code"
git push
```
Tracking upstream (optional):
```bash
cd src
git remote add upstream https://github.com/antgroup/ditto-talkinghead.git
git fetch upstream
# merge or cherry-pick as needed
```

## GPU Compatibility
Prebuilt TensorRT models target Ampere+ GPUs. If your GPU differs, convert ONNX → TensorRT in the container:
```bash
cd /app/src
python scripts/cvt_onnx_to_trt.py \
  --onnx_dir "/app/checkpoints/ditto_onnx" \
  --trt_dir "/app/checkpoints/ditto_trt_custom"
```
Then pass `--data_root=/app/checkpoints/ditto_trt_custom` to inference.

## Troubleshooting
- GPU not detected: check host NVIDIA drivers, NVIDIA Container Toolkit, and `--gpus all`/Compose GPU config.
- Permission issues with volumes: `sudo chown -R 1000:1000 ./checkpoints ./data ./output ./src`
- Out of memory: model is heavy; 8GB+ VRAM recommended.
- cuDNN errors: the image installs `libcudnn8`/`libcudnn8-dev` via apt; rebuild if you modified the base.

## License
Apache-2.0 (aligned with the upstream Ditto Talking Head project).

