# Remote Streaming Client Setup

This guide helps you run the streaming client on a remote machine, separate from the server.

## Quick Setup

1. **Copy client files to remote machine:**
   ```bash
   scp streaming_client.py client_requirements.txt user@remote-host:~/
   ```

2. **Install dependencies on remote machine:**
   ```bash
   pip install -r client_requirements.txt
   ```

3. **Run the client:**
   ```bash
   python streaming_client.py \
     --server ws://YOUR_SERVER_IP:8000 \
     --client_id remote_client \
     --audio_path ./example/audio.wav \
     --source_path ./example/image.png \
     --timeout 30
   ```

## Alternative: Using Virtual Environment

```bash
# On remote machine
python -m venv streaming_client_env
source streaming_client_env/bin/activate  # Linux/Mac
# or: streaming_client_env\Scripts\activate  # Windows

pip install -r client_requirements.txt

python streaming_client.py --server ws://YOUR_SERVER_IP:8000 --client_id remote_client ...
```

## Example Usage

**Server running on**: `192.168.1.100:8000`  
**Client command**:
```bash
python streaming_client.py \
  --server ws://192.168.1.100:8000 \
  --client_id laptop_client \
  --audio_path /path/to/audio.wav \
  --source_path /path/to/image.png \
  --timeout 60
```

## Notes

- Only **3 lightweight dependencies** vs ~50+ for the full server
- **~50MB download** vs ~5GB+ for server dependencies
- Works on **any Python 3.8+** system
- No GPU/CUDA requirements on client side
- Client receives and processes ~40MB of video data in ~6 seconds 