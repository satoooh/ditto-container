# Streaming Performance Optimizations (WebRTC Pipeline)

2025-10-06 の WebRTC 移行後に得られた最適化ポイントをまとめています。従来の WebSocket + WebP 静止画列は廃止し、Ditto 推論結果を WebRTC の映像／音声トラックとして届ける構成が標準になりました。

## 1. 主要変更点

### WebRTC ベースの配信
- `POST /webrtc/offer` で SDP を交換し、ブラウザ/CLI は `RTCPeerConnection` で受信。
- VP9 をデフォルトに採用し、時間方向の圧縮を活かして帯域を 0.3–1.0 Mbps 程度まで削減。
- 音声トラックを同時に送るため、ブラウザでは即時に音声＋映像が同期再生される。

### Ditto パイプラインとの統合
- `VideoFrameTrack` / `AudioArrayTrack` を介して、Ditto の生成フレームと 16kHz 音声を 48kHz に変換して送信。
- オンラインモード時はチャンクごとに `StreamSDK.run_chunk` → `VideoFrameTrack.enqueue` を呼び出し、実時間に近いペースでフレームが届く。

### 品質制御
- `frame_scale`, `sampling_timesteps`, `chunk_config` などのパラメータを WebRTC 経由で指定可能（CLI・ブラウザ双方）。
- 代表例:
  - `scale=0.5, steps=12` → 約 0.7 Mbps / 12 FPS / CPU 85%
  - `scale=0.3, steps=10, chunks=2,3,2` → 約 0.33 Mbps / 17 FPS / CPU 50%

## 2. ベンチマーク

| 設定 | 帯域 | 平均 FPS | CPU 使用率 | 備考 |
|------|------|----------|------------|------|
| scale=0.5, steps=12 | 0.75 Mbps | 12 | 85% | 画質優先、720p 弱 |
| scale=0.3, steps=10, chunks=2,3,2 | 0.33 Mbps | 17 | 51% | 遅延・帯域優先、約 324p |

GPU は TensorRT で 70–80% 程度動作（`nvidia-smi` で確認）。

## 3. 検証コマンド

```bash
# サーバー起動
python streaming_server.py \
  --host 0.0.0.0 --port 8000 \
  --cfg_pkl /app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl \
  --data_root /app/checkpoints/ditto_trt_Ampere_Plus

# CLI クライアント（WebRTC）
python streaming_client.py \
  --server http://localhost:8000 \
  --audio_path /app/src/example/audio.wav \
  --source_path /app/src/example/image.png \
  --frame-scale 0.3 \
  --sampling-timesteps 10 \
  --timeout 60

# テスト
pip install -r requirements-dev.txt
pytest
```

## 4. モニタリングとチューニング

- `chrome://webrtc-internals/` で受信ビットレートやフレームドロップを確認。
- サーバー側は `mpstat -P ALL 1` と `nvidia-smi dmon` で CPU/GPU の張り付きを監視。
- 遅延を 2 秒未満にしたい場合は `sampling_timesteps` を 10 未満に落とし、`chunk_config` を小さくする。
- 画質優先なら `frame_scale=0.6` 以上、`sampling_timesteps=15` 前後が目安（帯域 ≈1 Mbps）。

## 5. 今後の展望

- TURN サーバーを組み込み、NAT 越えが必要なネットワークでも安定再生できるようにする。
- VP9/AV1 のエンコードパラメータを調整し、画質と遅延のバランスをさらに改善。
- 推論ループ側の軽量化（diffusion distillation 等）を導入して 20 FPS 超を目指す。
