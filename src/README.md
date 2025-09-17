# Ditto Container Runtime Guide

このディレクトリには Ditto Talking Head（Ant Group 公開）のコードを同梱しつつ、TensorRT ベースの推論とリアルタイム配信を Docker コンテナ上で再現するための追加スクリプト群が含まれています。上流 (`antgroup/ditto-talkinghead`) との差分として、TensorRT コンテナ環境、FastAPI ストリーミング、ブラウザデモ、テストユーティリティを同梱しています。

## 前提条件
- NVIDIA GPU（Ampere 以降推奨）、対応するホストドライバ
- Docker + NVIDIA Container Toolkit
- Docker Compose v2（推奨）
- モデルチェックポイント（Hugging Face 上の `digital-avatar/ditto-talkinghead`）

## クイックスタート
```bash
git clone https://github.com/your-username/ditto-container.git
cd ditto-container
./setup.sh all        # build + run
```
`./setup.sh` は `checkpoints/`, `data/`, `output/` を自動作成し、Docker Compose (v2 → v1 → plain docker) の順で起動を試みます。手動で操作したい場合は以下を参照してください。

### 手動でのビルド/起動
```bash
# ビルド
./setup.sh build
# or: docker compose up -d --build

# 起動のみ
./setup.sh run
# or: docker compose up -d
```

## チェックポイントの取得
```bash
cd ditto-container
mkdir -p checkpoints
cd checkpoints
git lfs install
git clone https://huggingface.co/digital-avatar/ditto-talkinghead .
```
主要ファイル
- `ditto_cfg/v0.4_hubert_cfg_trt_online.pkl` : ストリーミング用コンフィグ
- `ditto_trt_Ampere_Plus/` : Ampere+ 向け TensorRT エンジン群（別 GPU の場合は `scripts/cvt_onnx_to_trt.py` で再生成）

## コンテナ内での推論ワークフロー
```bash
# ホストから
docker compose exec ditto-talkinghead bash

# コンテナ内
cd /app/src
python inference.py \
  --data_root "/app/checkpoints/ditto_trt_Ampere_Plus" \
  --cfg_pkl "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl" \
  --audio_path "/app/data/audio.wav" \
  --source_path "/app/data/source_image.png" \
  --output_path "/app/output/result.mp4"
```
※ PyTorch 版を利用する場合は `ditto_pytorch/` と `v0.4_hubert_cfg_pytorch.pkl` を指定します。

## ストリーミングワークフロー
### サーバー起動
```bash
cd /app/src
python streaming_server.py \
  --host 0.0.0.0 --port 8000 \
  --cfg_pkl "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl" \
  --data_root "/app/checkpoints/ditto_trt_Ampere_Plus"
```
主な特徴
- FastAPI + WebSocket によりリアルタイム配信
- 起動時に TensorRT/StreamSDK をプリウォーム（初回フレーム短縮）
- フレームはバイナリ WebSocket（ヘッダ `!IdI` + JPEG）で送信。
- キュー長と JPEG 品質を監視し、混雑時に自動で品質を調整
- `/upload` エンドポイントでブラウザから音声・画像をアップロード可能

### クライアント/ブラウザ
- ブラウザ: `http://<HOST>:8000/demo` （バイナリストリームを自動再生）
- CLI: `python streaming_client.py --server ws://<HOST>:8000 --client_id test \
    --audio_path /app/src/example/audio.wav --source_path /app/src/example/image.png`
  - デフォルトでバイナリを受信。旧 JSON 経路を使う場合は `--transport json` を指定。

## テスト
```bash
pip install -r requirements-dev.txt
pytest
# プロトコルのみ: pytest -k binary_frame
```

## 参考ドキュメント
- `_docs/2025-09-04_container_run_keepalive.md` : コンテナ起動時のトラブルシュート
- `_docs/2025-09-04_binary_ws_prewarm.md` : バイナリ化/プリウォーム実装メモ
- `_docs/2025-09-04_browser_upload_streaming.md` : ブラウザアップロード導線
- `_docs/2025-09-17_setup_streaming_optim.md` : 最新実装ログ
- `src/STREAMING_SETUP.md` : ストリーミング構成概略
- `src/STREAMING_OPTIMIZATIONS.md` : 最適化の詳細と検証手順

## トラブルシューティング
- コンテナが即停止する → `setup.sh run` は `sleep infinity` にフォールバックしますが、ログ (`docker logs ditto-container`) を確認してください。
- チェックポイントが読み込めない → `checkpoints/` のパーミッションを `1000:1000` に合わせる。
- ストリーミングが遅い → `streaming_server.py` のログでキューサイズ・ドロップ数を確認。必要に応じて `--host`/`--port` や JPEG 品質閾値を調整。

## ライセンス
- 同梱コードは Apache-2.0（上流 Ditto Talking Head と同一）。
- コンテナ周辺スクリプトも同ライセンスで提供しています。
