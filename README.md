# Ditto TalkingHead Docker Container（日本語ガイド）

TensorRT 版 Ditto Talking Head を GPU 対応 Docker コンテナで再現し、リアルタイム配信まで実行するための環境です。`src/` には上流 (`antgroup/ditto-talkinghead`) のコードを同梱しつつ、FastAPI ストリーミングサーバーやブラウザデモ、テストユーティリティなど追加機能を含んでいます。

---
## 1. 前提条件
- NVIDIA GPU（Ampere 〜 Blackwell 世代）と R575.51 以降のホストドライバ（CUDA 12.9 対応）
- Docker / Docker Compose v2（v1 でも可）
- NVIDIA Container Toolkit（`nvidia-container-toolkit`）
- NVIDIA NGC アカウントと `docker login nvcr.io` が可能な資格情報
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
- Ampere/Ada 向け TensorRT エンジン: `checkpoints/ditto_trt_Ampere_Plus/`
- RTX 5090 など Blackwell 向けエンジン: `checkpoints/ditto_trt_blackwell/`
- 新しい GPU で利用する際は `python src/scripts/cvt_onnx_to_trt.py --onnx_dir /app/checkpoints/ditto_onnx --trt_dir /app/checkpoints/ditto_trt_blackwell` で再生成

### 2-3. コンテナのビルドと起動
初回は NGC へログインして TensorRT コンテナを取得します。
```bash
sudo docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC API Key>
```
その後、通常どおりビルド & 起動します。
```bash
./setup.sh all     # build + run
```
`./setup.sh` は以下を実行します。
- `checkpoints/`,`data/`,`output/` の作成
- NGC ベースの `nvcr.io/nvidia/tensorrt:25.08-py3` を元に CUDA 12.9 + TensorRT Python 10.x ホイールをセットアップ
- Docker Compose v2 → v1 → plain docker の順に起動を試行
- fallback 時は `bash -lc 'sleep infinity'` でコンテナ終了を防止

手動操作の例:
```bash
./setup.sh build   # docker build / docker compose build
./setup.sh run     # docker compose up -d （fallback: docker run）
```

### 2-4. TensorRT エンジンを再生成する
TensorRT 10.x (Python バインディング 10 系) は Ampere〜Blackwell まで 1 つのエンジンで共有できます。新しい GPU を追加したら、以下の手順で universal ディレクトリを更新してください。
```bash
# コンテナ内 (/app) で実行
python - <<'PY'
import tensorrt as trt
print('TensorRT Python version:', trt.__version__)
PY

python src/scripts/cvt_onnx_to_trt.py \
  --onnx_dir /app/checkpoints/ditto_onnx \
  --trt_dir /app/checkpoints/ditto_trt_blackwell
```
- `--onnx_dir` には Hugging Face 配布の `checkpoints/ditto_onnx/` を指定（パス名が異なる場合はシンボリックリンクを作るか引数を調整）
- GPU の Compute Capability から Ampere/Ada/Blackwell 向けハードウェア互換レベルを自動選択します（RTX 5090 では `Blackwell_Plus`）
- `--trt_dir` は任意ですが、`ditto_trt_blackwell/` を推奨（既存 Ampere エンジンと共存させる）
- 生成された `.engine` は `streaming_server.py` が自動選択します

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
     --data_root "/app/checkpoints/ditto_trt_blackwell" \
     --cfg_pkl "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
     --audio_path "/app/data/audio.wav" \
     --source_path "/app/data/source_image.png" \
     --output_path "/app/output/result.mp4"
   ```
   PyTorch 版を利用する場合は `ditto_pytorch/` と `v0.4_hubert_cfg_pytorch.pkl` を指定します。

---
## 4. リアルタイムストリーミング
### 4-1. サーバー起動
```bash
cd /app/src
python streaming_server.py \
  --host 0.0.0.0 --port 8000 \
  --cfg_pkl "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
```
- FastAPI + WebSocket によるリアルタイム配信
- `--data_root` を省略すると GPU の世代に応じて `checkpoints/ditto_trt_blackwell/`（Blackwell）→ `checkpoints/ditto_trt_Ampere_Plus/`（Ampere/Ada）を自動で選択
- 起動時に TensorRT / StreamSDK をプリウォーム
- フレームはヘッダ `!IdI` + WebP のバイナリ WebSocket で送信
- キュー深度に応じて WebP 品質を 85/75/60 に自動調整
- `/upload` で音声・画像をアップロード可能

### 4-2. クライアント
```bash
python streaming_client.py \
  --server ws://localhost:8000 \
  --client_id bench \
  --audio_path /app/src/example/audio.wav \
  --source_path /app/src/example/image.png
```
- デフォルトでバイナリ受信。旧 JSON 経路を使う場合は `--transport json` を指定
- 終了時に FPS / 初回フレーム時間 / 帯域などの統計を出力

### 4-3. ブラウザデモ
- アクセス先: `http://<ホスト>:8000/demo`
- WebSocket URL はページ内で自動検出
- アップロード → 推論 → プレビューをブラウザのみで実行可能

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
