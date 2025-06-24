# Ditto TalkingHead Docker Container

This repository contains a Docker setup for the [ditto-talkinghead](https://github.com/antgroup/ditto-talkinghead) project, which provides Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis.

## Prerequisites

- Docker with GPU support
- NVIDIA Docker runtime
- NVIDIA GPU with CUDA support
- Docker Compose (optional but recommended)

## Quick Start

### Using Docker Compose (Recommended)

1. **Build and run the container:**
   ```bash
   docker-compose up -d --build
   ```

2. **Access the container:**
   ```bash
   docker-compose exec ditto-talkinghead bash
   ```

3. **Stop the container:**
   ```bash
   docker-compose down
   ```

### Using Docker directly

1. **Build the image:**
   ```bash
   docker build -t ditto-talkinghead .
   ```

2. **Run the container:**
   ```bash
   docker run -it --gpus all \
     -v $(pwd)/checkpoints:/app/checkpoints \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/output:/app/output \
     -v $(pwd)/src:/app/src \
     --name ditto-container \
     ditto-talkinghead
   ```

## Container Features

- **Base Image:** NVIDIA CUDA 11.8 with Ubuntu 22.04
- **Python:** 3.10
- **GPU Support:** Full CUDA and TensorRT support
- **Pre-installed Dependencies:**
  - PyTorch with CUDA support
  - TensorRT 8.6.1
  - OpenCV
  - librosa
  - All other required packages from the original repository

## Directory Structure

The container expects the following directory structure for volume mounts:

```
./
├── checkpoints/          # Model checkpoints (to be downloaded)
├── data/                # Input data (images, audio files)
├── output/              # Generated outputs
├── src/                 # Source code from ditto-talkinghead
└── docker files...
```

## Setting Up the Project

After running the container, you'll need to:

1. **Clone the source code inside the container:**
   ```bash
   cd /app/src
   git clone https://github.com/antgroup/ditto-talkinghead .
   ```

2. **Download the model checkpoints:**
   ```bash
   cd /app
   git lfs install
   git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
   ```

3. **Run inference:**
   ```bash
   cd /app/src
   python inference.py \
     --data_root "/app/checkpoints/ditto_trt_Ampere_Plus" \
     --cfg_pkl "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
     --audio_path "/app/data/audio.wav" \
     --source_path "/app/data/image.png" \
     --output_path "/app/output/result.mp4"
   ```

## GPU Compatibility

The pre-built TensorRT models are compatible with `Ampere_Plus` GPUs. If your GPU doesn't support this, you'll need to convert the ONNX models to TensorRT inside the container:

```bash
cd /app/src
python script/cvt_onnx_to_trt.py \
  --onnx_dir "/app/checkpoints/ditto_onnx" \
  --trt_dir "/app/checkpoints/ditto_trt_custom"
```

Then use `--data_root=/app/checkpoints/ditto_trt_custom` in your inference command.

## Troubleshooting

### GPU Not Detected
Ensure you have:
- NVIDIA drivers installed on the host
- NVIDIA Docker runtime installed
- Used `--gpus all` flag or proper docker-compose GPU configuration

### Permission Issues
The container runs as a non-root user. If you encounter permission issues with mounted volumes, adjust the ownership:
```bash
sudo chown -R 1000:1000 ./checkpoints ./data ./output ./src
```

### Memory Issues
This model requires significant GPU memory. Ensure your GPU has enough VRAM (recommended: 8GB+).

## License

This Docker setup is provided under the same Apache-2.0 license as the original ditto-talkinghead project. 