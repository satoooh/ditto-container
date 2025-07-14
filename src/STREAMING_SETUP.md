# Ditto Streaming Architecture Setup

This document describes how to set up and test the real-time streaming architecture for Ditto.

## Overview

The streaming architecture consists of:
1. **Server** (`streaming_server.py`): FastAPI server with WebSocket that runs inference and streams frames
2. **Client** (`streaming_client.py`): Python CLI client that receives frames and computes performance stats

## Prerequisites

### Install Dependencies

```bash
# Install streaming dependencies
pip install -r streaming_requirements.txt

# OR if using uv:
uv pip install -r streaming_requirements.txt
```

### Required Files

Make sure you have:
- `../checkpoints/ditto_trt_Ampere_Plus/` - TensorRT models for Ampere+ GPUs
- `../checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl` - TensorRT configuration
- `./example/audio.wav` - Test audio file
- `./example/image.png` - Test source image

## Usage

### Method 1: Run Server and Client Separately

#### Start the Server
```bash
python streaming_server.py \
    --data_root "../checkpoints/ditto_trt_Ampere_Plus/" \
    --cfg_pkl "../checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
    --host 0.0.0.0 \
    --port 8000
```

#### Start the Client (in another terminal)
```bash
python streaming_client.py \
    --server ws://localhost:8000 \
    --client_id test_client \
    --audio_path ./example/audio.wav \
    --source_path ./example/image.png \
    --timeout 60
```

### Method 2: Use Launch Script

```bash
# Run both server and client
python run_streaming_test.py --mode both

# Run only server
python run_streaming_test.py --mode server

# Run only client (server must be running)
python run_streaming_test.py --mode client
```

## Expected Output

### Server Output
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:streaming_server:Client test_client connected
INFO:streaming_server:Starting streaming for client test_client
```

### Client Output
```
INFO:streaming_client:Connected to server as client test_client
INFO:streaming_client:Sent start streaming command
INFO:streaming_client:Streaming started: ./example/audio.wav -> ./example/image.png
INFO:streaming_client:Metadata: 15.75s, 394 frames
INFO:streaming_client:Received frame 0, size: 45.2KB
INFO:streaming_client:Received frame 25, size: 44.8KB
...
INFO:streaming_client:Streaming completed
INFO:streaming_client:Writer closed. Total frames: 394

======================================================================
STREAMING CLIENT STATISTICS
======================================================================
Total Frames Received: 394
Total Duration: 12.45s
Streaming Latency: 1.23s (time to first frame)
Average FPS: 31.6
Total Data: 17.8 MB
Average Frame Size: 46.3 KB
Bandwidth: 11.4 Mbps

FRAME INTERVALS:
  Count: 393
  Mean: 31.7ms
  Median: 29.4ms
  Std Dev: 12.3ms
  Min: 15.2ms
  Max: 85.7ms

REAL-TIME SUITABILITY:
✅ Frame rate suitable for real-time (31.6 >= 22.5 FPS)
✅ Streaming latency acceptable (1.2s <= 3.0s)
✅ Frame intervals consistent (31.7ms <= 60.0ms)
======================================================================
```

## Key Metrics

The client computes several important metrics:

- **Streaming Latency**: Time from start command to first frame (target: <3s)
- **Frame Rate**: Average FPS (target: >22.5 FPS for smooth playback)
- **Frame Intervals**: Consistency of frame delivery (target: <60ms mean)
- **Bandwidth**: Network usage (typical: 10-20 Mbps)

## Troubleshooting

### Common Issues

1. **Connection Refused**: Make sure server is running and ports are open
2. **File Not Found**: Check paths to audio/image files and model checkpoints
3. **Out of Memory**: Reduce batch size or use smaller models
4. **Slow Performance**: Ensure using TensorRT models on compatible GPU

### Debug Mode

Add `--log-level debug` to uvicorn for more detailed server logs:
```bash
python streaming_server.py --log-level debug
```

### Save Frames

To save received frames for inspection:
```bash
python streaming_client.py --save_frames
```

Frames will be saved to `received_frames/` directory.

## Performance Tuning

### Server-Side
- Use TensorRT models (`ditto_trt_Ampere_Plus`)
- Ensure GPU memory is sufficient
- Adjust chunk size in configuration

### Client-Side
- Reduce JPEG quality in `WebSocketFrameWriter` (line with `IMWRITE_JPEG_QUALITY`)
- Increase timeout for slower connections
- Use local network for testing

### Network
- Use wired connection for best performance
- Consider reducing frame resolution for bandwidth-limited scenarios
- Test with different JPEG quality settings

## Integration Notes

This streaming architecture demonstrates:
- Real-time frame delivery with ~1-2s latency
- Consistent frame rates suitable for live streaming
- Bandwidth usage compatible with typical network conditions
- WebSocket-based communication for low-latency bidirectional messaging

The client statistics provide comprehensive metrics to evaluate real-time suitability. 