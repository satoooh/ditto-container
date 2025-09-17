# Streaming Performance Optimizations

> **Update (2025-09-17)**: These optimizations are now the default behavior of `streaming_server.py` and `streaming_client.py`. The previous `_optimized` variants have been removed.

## 🚨 **Critical Issues Fixed**

### **Original Problems (Causing 3.6 FPS vs 21 FPS network capacity)**

1. **Threading.Queue in Async Context** - Major bottleneck
2. **Excessive Logging** - Blocking critical path  
3. **Base64 + JSON Overhead** - 33% size increase
4. **Polling Frame Queue** - CPU waste
5. **Large Queue Size** - Memory bloat

## ✅ **Optimizations Implemented**

### **1. Queue Management Overhaul**
```python
# ❌ BEFORE: threading.Queue with polling
self.frame_queues[client_id] = queue.Queue(maxsize=500)  # 23MB memory
message = frame_queue.get_nowait()  # Non-blocking polling
await asyncio.sleep(0.01)  # Constant polling waste

# ✅ AFTER: asyncio.Queue with blocking get
self.frame_queues[client_id] = asyncio.Queue(maxsize=50)  # 2.3MB memory
message = await asyncio.wait_for(frame_queue.get(), timeout=1.0)  # Efficient blocking
```
**Impact**: 90% less memory usage, no CPU polling waste

### **2. Logging Reduction**
```python
# ❌ BEFORE: Logging every frame (huge overhead)
logger.info(f"WebSocketFrameWriter called for frame {self.frame_count}")
logger.info(f"Frame {self.frame_count} queued for client {self.client_id}")
logger.info(f"📤 Sending message type '{message.get('type')}' to {client_id}")

# ✅ AFTER: Minimal logging
if self.frame_count % 100 == 0:  # Log every 100th frame
    logger.info(f"Frame {self.frame_count} queued (quality: {quality}%)")
```
**Impact**: 99% less logging I/O, faster frame processing

### **3. Binary WebSocket Messages (WebP)**
```python
# ❌ BEFORE: Base64 + JSON overhead
frame_base64 = base64.b64encode(buffer).decode('utf-8')  # +33% size
message = {"type": "frame", "frame_data": frame_base64, ...}
await websocket.send_text(json.dumps(message))

# ✅ AFTER: Binary protocol
from streaming_protocol import build_binary_frame_payload

binary_message = build_binary_frame_payload(frame_id, timestamp, jpeg_bytes)
await websocket.send_bytes(binary_message)
```
**Impact**: 25% smaller messages, faster transmission

### **4. Adaptive Quality Control**
```python
# ✅ NEW: Dynamic JPEG quality based on queue depth
queue_size = self.server.frame_queues[self.client_id].qsize()
if queue_size > 30:
    quality = 60    # Lower quality when backed up
elif queue_size > 15:
    quality = 75    # Medium quality  
else:
    quality = 85    # Normal quality
```
**Impact**: Maintains framerate under load

### **5. Frame Dropping vs Blocking**
```python
# ❌ BEFORE: Block when queue full (causes cascade delays)
self.frame_queues[client_id].put_nowait(message)  # Exception if full

# ✅ AFTER: Drop frames gracefully
try:
    self.frame_queues[client_id].put_nowait(message)
except asyncio.QueueFull:
    logger.warning(f"Dropping frame {frame_count} - queue full")
```
**Impact**: No cascade blocking, smooth operation

## 📊 **Expected Performance Improvements**

### **Frame Processing Pipeline**
| Component | Original | Optimized | Improvement |
|-----------|----------|-----------|-------------|
| Queue Operations | Threading.Queue polling | asyncio.Queue blocking | **5-10x faster** |
| Message Encoding | Base64 + JSON | Binary protocol | **25% less data** |
| Logging Overhead | Every frame | Every 100th frame | **99% less I/O** |
| Memory Usage | 23MB queue | 2.3MB queue | **90% less memory** |
| Frame Drops | Cascade blocking | Graceful dropping | **Smooth operation** |

### **Network Efficiency**
| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Frame Size | 61KB (46KB + base64) | 32KB (WebP, binary) | **≈50% smaller** |
| Protocol Overhead | JSON headers | 16-byte binary header | **Minimal overhead** |
| Queue Memory | 500 × 61KB = 30.5MB | 50 × 46KB = 2.3MB | **93% less memory** |

## 🎯 **Performance Targets**

**Network Capacity**: 21 FPS (from bandwidth test)
**Optimized Target**: **18-20 FPS** (vs original 3.6 FPS)

### **Bottleneck Analysis**
1. ✅ **Queue Management**: `asyncio.Queue` に統一しポーリングを排除
2. ✅ **Logging Overhead**: 100 フレームごとの統計ログのみ
3. ✅ **Message Efficiency**: `build_binary_frame_payload` によるバイナリ転送
4. ✅ **Memory Usage**: キュー長 50 で約 90% のメモリ削減
5. ⚠️ **GPU Processing**: 依然としてモデル推論速度が全体の上限

## 🧪 **Testing the Optimizations**

### **Current Workflow**
```bash
# Server (optimized pipeline now default)
python streaming_server.py \
  --host 0.0.0.0 --port 8000 \
  --cfg_pkl "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl" \
  --data_root "/app/checkpoints/ditto_trt_Ampere_Plus"

# Client (binary transport enabled by default)
python streaming_client.py \
  --server ws://localhost:8000 \
  --client_id bench \
  --audio_path /app/src/example/audio.wav \
  --source_path /app/src/example/image.png

# Protocol unit tests
pip install -r requirements-dev.txt
pytest -k binary_frame
```

目標値は GPU 性能に依存しますが、ネットワーク帯域の観点では 18–20 FPS を維持しつつ、初回フレーム遅延を 3 秒以内に収めることを想定しています。

## 🔍 **Monitoring Performance**

### **Key Metrics to Watch**
1. **Frame Rate**: Should approach bandwidth test results (21 FPS)
2. **Queue Depth**: Should stay below 20 frames
3. **Memory Usage**: 90% reduction expected
4. **Latency**: First frame time should improve
5. **Bandwidth**: 25% reduction from binary protocol

### **Debugging Commands**
```bash
# Monitor queue sizes (if logging enabled)
grep "Queue size" server.log

# Monitor frame drops
grep "Dropping frame" server.log

# Check memory usage
ps aux | grep streaming_server

# Network monitoring
nethogs  # Real-time bandwidth per process
```

## 🚀 **Next Optimizations (Future)**

1. **Frame Batching**: Send multiple frames per WebSocket message
2. **Compression**: Use WebSocket compression extensions  
3. **Parallel Encoding**: マルチスレッド WebP エンコード
4. **Dynamic Resolution**: Reduce resolution under network stress
5. **Predictive Dropping**: Drop frames before queue fills

## 💡 **Architecture Insights**

### **Why Original Was Slow**
1. **Thread-Async Mismatch**: `threading.Queue` in `asyncio` context created sync barriers
2. **Logging Storm**: 400+ log messages per second overwhelmed I/O
3. **Protocol Bloat**: Base64 encoding added 33% overhead unnecessarily  
4. **Memory Pressure**: 30MB queues caused garbage collection pauses
5. **Cascade Blocking**: Full queues blocked entire pipeline

### **Why Optimized Is Fast**
1. **Native Async**: `asyncio.Queue` with natural `await` patterns
2. **Minimal Logging**: Only essential information logged
3. **Efficient Protocol**: Binary messages with minimal headers
4. **Right-Sized Queues**: 50 frames ≈ 2 seconds of buffering
5. **Graceful Degradation**: Frame dropping prevents cascade failures

The optimizations target the **application-level bottlenecks** that were preventing the streaming from achieving the network's 21 FPS capacity. GPU inference speed remains the ultimate limit, but these changes should bring performance much closer to that theoretical maximum. 
