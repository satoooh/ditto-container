# Bandwidth Test Guide

These scripts measure WebSocket throughput between your server and client to determine optimal streaming parameters.

## 🚀 Quick Start

### **Server Side (GPU server)**
```bash
# Install dependencies
pip install -r bandwidth_test_requirements.txt

# Run bandwidth test server
python bandwidth_test_server.py --host 0.0.0.0 --port 8001
```

### **Client Side (remote machine)**
```bash
# Install dependencies  
pip install -r bandwidth_test_requirements.txt

# Run bandwidth test (replace with your server IP)
python bandwidth_test_client.py --server ws://10.49.167.242:8001 --client_id my_client
```

## 📊 What the Test Measures

### **Progressive Chunk Sizes**: 1KB → 150KB
- **Small chunks (1-10KB)**: Latency + overhead effects
- **Medium chunks (25-50KB)**: Typical frame sizes (our frames: ~46KB)
- **Large chunks (75-150KB)**: Above-normal capacity testing

### **Sustained Throughput**: 10-second continuous test
- **Most important metric** for streaming
- Real-world bandwidth under continuous load
- What your streaming will actually achieve

### **Bidirectional Testing**:
- **Server→Client**: Streaming video frames  
- **Client→Server**: Control messages, acknowledgments

## 🎯 Interpreting Results

### **Example Output:**
```
📡 SERVER → CLIENT (Download):
       1KB:     2.15 Mbps (  3.8ms)
      50KB:    12.45 Mbps ( 32.1ms)  
     100KB:    15.23 Mbps ( 52.4ms)
     500KB:     8.91 Mbps (448.2ms)
    2000KB:     6.12 Mbps (2608.7ms)
 sustained_10s:  5.80 Mbps         ← **KEY METRIC**

🎯 STREAMING ANALYSIS:
  Sustained bandwidth: 5.80 Mbps
  Current frame size: 46 KB  
  Maximum FPS: 15.7 fps          ← **BOTTLENECK IDENTIFIED**
  Current target: 40 fps
  ⚠️  Network limited to 15.7 FPS
  💡 Reduce frame size to 18 KB for 40 FPS
```

### **Key Insights:**

1. **Sustained bandwidth** = Real streaming capacity
2. **Large chunks perform worse** = Network buffering limits
3. **Bidirectional differences** = Upload vs download asymmetry

## 🔧 Streaming Optimization

### **If bandwidth is limited:**

| **Current** | **Bandwidth Limited Solution** |
|-------------|--------------------------------|
| 46KB @ 40fps = 15 Mbps | **Reduce frame size** to match bandwidth |
| High quality JPEG | **Lower JPEG quality** (95% → 75%) |
| Full resolution | **Scale resolution** (1080p → 720p) |
| Continuous streaming | **Frame skipping** during congestion |

### **Calculation Examples:**
```python
# Available bandwidth: 6 Mbps
bandwidth_mbps = 6.0

# Current requirements: 46KB @ 40fps  
current_mbps = (46 * 1024 * 8 * 40) / 1_000_000  # = 15.1 Mbps

# Solutions:
# Option 1: Reduce frame size
new_frame_size = int(46 * bandwidth_mbps / current_mbps)  # = 18 KB

# Option 2: Reduce frame rate  
new_fps = int(40 * bandwidth_mbps / current_mbps)  # = 3.7 fps

# Option 3: Hybrid approach
moderate_frame_size = 50  # KB (close to our current 46KB)
moderate_fps = (bandwidth_mbps * 1_000_000) / (moderate_frame_size * 1024 * 8)  # = 14.6 fps
```

## 🐛 Troubleshooting

### **Connection Issues:**
```bash
# Test basic connectivity first
ping 10.49.167.242

# Check if port is open
telnet 10.49.167.242 8001

# Test with curl
curl http://10.49.167.242:8001/
```

### **Permission Issues:**
```bash
# Server might need firewall rules
sudo ufw allow 8001/tcp

# Or run on different port
python bandwidth_test_server.py --port 8080
```

### **Results Interpretation:**
- **Very low bandwidth (<1 Mbps)**: Network connectivity issues
- **Inconsistent results**: Network congestion or interference  
- **Upload ≠ Download**: Asymmetric connection (common with home internet)
- **Large chunks slower**: Network buffering/latency effects

## 📈 Next Steps

1. **Run the test** to get baseline measurements
2. **Compare** to streaming requirements (46KB @ 40fps = 15.1 Mbps)
3. **Adjust streaming parameters** based on results:
   - If bandwidth sufficient: Keep current settings
   - If bandwidth limited: Implement adaptive quality
4. **Re-test streaming** with optimized parameters 