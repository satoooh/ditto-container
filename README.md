# Ditto TalkingHead Docker Container（日本語ガイド）

TensorRT 版 Ditto Talking Head を GPU 対応 Docker コンテナで再現し、リアルタイム配信まで実行するための環境です。`src/` には上流 (`antgroup/ditto-talkinghead`) のコードを同梱しつつ、FastAPI ストリーミングサーバーやブラウザデモ、テストユーティリティなど追加機能を含んでいます。

---
## 1. 前提条件
- NVIDIA GPU（Ampere 世代以上推奨）と対応ドライバ
- Docker / Docker Compose v2（v1 でも可）
- NVIDIA Container Toolkit（`nvidia-container-toolkit`）
- モデルチェックポイント（Hugging Face: `digital-avatar/ditto-talkinghead`）

---
## 2. セットアップ手順
### 2-1. リポジトリ取得
```bash
git clone https://github.com/your-username/ditto-container.git
cd ditto-container
```

### 2-2. チェックポイント配置
```bash
mkdir -p checkpoints
git lfs install
git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
```
- ストリーミング用設定: `checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl`
- Ampere+ 向け TensorRT エンジン: `checkpoints/ditto_trt_Ampere_Plus/`
- 異なる GPU を利用する場合は `python src/scripts/cvt_onnx_to_trt.py` で再生成

### 2-3. コンテナのビルドと起動
```bash
./setup.sh all     # build + run
```
`./setup.sh` は以下を実行します。
- `checkpoints/`,`data/`,`output/` の作成
- Docker Compose v2 → v1 → plain docker の順に起動を試行
- fallback 時は `bash -lc 'sleep infinity'` でコンテナ終了を防止

手動操作の例:
```bash
./setup.sh build   # docker build / docker compose build
./setup.sh run     # docker compose up -d （fallback: docker run）
```

---
## 3. コンテナ内での推論
1. コンテナへ入る
   ```bash
   docker compose exec ditto-talkinghead bash
   # もしくは docker exec -it ditto-container bash
   ```
2. 推論を実行
   ```bash
   cd /app/src
   python inference.py \
     --data_root "/app/checkpoints/ditto_trt_Ampere_Plus" \
     --cfg_pkl "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
     --audio_path "/app/data/audio.wav" \
     --source_path "/app/data/image.png" \
     --output_path "/app/output/result.mp4"
   ```
   PyTorch 版を利用する場合は `ditto_pytorch/` と `v0.4_hubert_cfg_pytorch.pkl` を指定します。

---
## 4. リアルタイムストリーミング
### 4-1. アーキテクチャ概要（WebRTC）
- **FastAPI + aiortc** でシグナリングと WebRTC メディア送信を行います。
- `StreamSDK` の推論結果（映像）と 16 kHz 音声をそれぞれ `MediaStreamTrack` に流し込み、ブラウザや CLI が `RTCPeerConnection` で受信します。
- VP9（デフォルト）による時間方向圧縮のおかげで、同品質で従来より大幅に帯域を削減できます。

### 4-2. 依存インストール
Docker を使わずにローカルで試す場合は WebRTC 依存 (`aiortc`, `av`, `aiohttp`, `ruff` など) を含めた開発環境をインストールしてください。
```bash
pip install -r requirements-dev.txt
```

Docker コンテナを使う場合は `./setup.sh all` を再実行してイメージを再ビルドしてください（Dockerfile に WebRTC 依存を追加済み）。

### 4-3. サーバー起動
```bash
cd /app/src
python streaming_server.py \
  --host 0.0.0.0 --port 8000 \
  --cfg_pkl "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl" \
  --data_root "/app/checkpoints/ditto_trt_Ampere_Plus" \
  --frame-scale 0.5 \
  --online-sampling-steps 12
```
- FastAPI + WebRTC によるリアルタイム配信（シグナリング API: `POST /webrtc/offer`）。
- 起動時に TensorRT / StreamSDK をプリウォームし、推論と同時に WebRTC トラックへ映像・音声を送信。
- `--frame-scale`, `--online-sampling-steps` などの既定値を CLI から調整可能。
- `/upload` エンドポイントは従来通り利用可能（ブラウザからも利用できます）。

### 4-4. クライアント（CLI）
```bash
python streaming_client.py \
  --server http://localhost:8000 \
  --audio_path /app/src/example/audio.wav \
  --source_path /app/src/example/image.png \
  --frame-scale 0.5 \
  --sampling-timesteps 12 \
  --timeout 30
```
- WebRTC ベースでサーバーとシグナリングを行い、映像・音声を受信します（`--record-file` を指定すると WebM へ保存）。
- 実行後に受信フレーム数と推定 FPS を標準出力に表示します。

### 4-5. ブラウザデモ
- アクセス先: `http://<ホスト>:8000/demo`
- 「Start」を押すとブラウザが WebRTC で接続し、低レイテンシで映像・音声を再生します。
- `Frame Scale` や `Sampling Steps` をページ上で変更して、帯域と画質のトレードオフを即座に確認できます。
- 必要に応じて `Upload Audio` / `Upload Image` でファイルをアップロードすると、保存先パスが自動で入力欄に反映されます。

### 4-6. 品質プリセットの目安
- **画質優先**: `frame_scale=0.6`, `sampling_steps=15`（帯域 ≈1 Mbps / ≈12 FPS）。
- **遅延優先**: `frame_scale=0.3`, `sampling_steps=10`, `chunk_config=2,3,2`（帯域 ≈0.3 Mbps / ≈17 FPS）。
  - CPU/GPU の余裕を見ながら数値を調整してください。`--record-file` で受信映像を保存し、画質チェックが可能です。

### 4-7. リモートクライアント
- サーバーと異なるマシンで CLI を動かす手順は [`src/REMOTE_CLIENT_SETUP.md`](src/REMOTE_CLIENT_SETUP.md) にまとめています。
- 基本的には `streaming_client.py` と `requirements-dev.txt` を転送し、`pip install -r requirements-dev.txt` 後に CLI を実行すれば動作します。

---
## 5. 運用ノウハウ
### 5-1. コンテナ起動トラブルシュート
- `docker compose ps` で `Up` を確認。すぐ停止する場合は `docker compose logs -f` を確認
- 既存コンテナが残ると名前衝突するため `docker compose down --remove-orphans` を実行
- ボリューム権限エラーは `sudo chown -R 1000:1000 checkpoints data output` で修正

### 5-2. バイナリ配信とプリウォーム
- `streaming_protocol.py` に定義したヘッダ `!IdI`（frame_id, timestamp, payload_len）で圧縮オーバーヘッドを約 25% 削減
- `streaming_server.py` 起動時に `StreamSDK` を初期化し、初回フレーム遅延を短縮
- フレームキューは `asyncio.Queue(maxsize=50)` に統一し、混雑時は自動的に品質を落としてドロップを最小化

### 5-3. ブラウザからのアップロード
- `/demo` 画面で音声・画像ファイルを選択し「Upload」→「Connect」→「Start」で即時配信
- アップロードファイルはサーバー側 `/app/data/uploads/` に保存。Compose 利用時はホストの `./data/uploads/`

### 5-4. 参照資料（リポジトリ内）
| ファイル | 内容 |
|----------|------|
| `src/STREAMING_SETUP.md` | ストリーミング構成の概要と構築手順 |
| `src/STREAMING_OPTIMIZATIONS.md` | 最適化ポイントと検証方法 |
| `src/README.md` | コンテナ内 `/app/src` 利用ガイド |

---
## 6. テスト
```bash
pip install -r requirements-dev.txt
pytest
# プロトコルのみ実行: pytest -k binary_frame
```
`streaming_protocol.py` に対するバイナリヘッダの pack/unpack テストを収録しています。

---
## 7. トラブルシューティングまとめ
- **GPU が認識されない**: ホストのドライバ、`nvidia-smi`、`nvidia-container-toolkit` の導入状況を確認
- **モデル展開で容量不足**: 大容量ディスクをマウントする、または必要ファイルのみ LFS 取得
- **ストリーミングが遅い**: サーバーログで `Dropping frame` や `queue≈` を確認し、帯域や WebP 品質閾値を調整
- **ブラウザで接続できない**: ポート 8000 の開放状況、HTTPS 環境での `wss://` 切り替えを確認

---
## 8. ライセンス
- 本リポジトリおよび同梱コードは Apache-2.0 ライセンスです。
