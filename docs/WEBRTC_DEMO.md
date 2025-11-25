# WebRTC デモ実行手順（GPU 環境向け）

## これは何か（where / what / why）
- where: 本リポジトリ `docs/WEBRTC_DEMO.md`
- what: WebRTC ストリーミングデモを GPU + Docker 環境で動かすための最小手順とコマンド例
- why: 実機での接続確認を迅速に行うための手順書

## 前提
- NVIDIA GPU 搭載ホスト（Docker と nvidia-container-runtime が有効）
- `data/` に TRT 推論に必要なリソース、`checkpoints/cfg.pkl` にモデル設定が配置済み（非同梱）
- ホストの 8000 番ポートが空いていること

## サーバーの立ち上げ（Docker）
```bash
# 1) 依存ビルド（初回のみ）
docker compose build

# 2) コンテナ起動（バックグラウンド）
docker compose up -d ditto-talkinghead

# 3) コンテナに入る
docker compose exec ditto-talkinghead bash

# 4) サーバー起動（コンテナ内）
python3 src/streaming_server.py \
  --data_root /app/data \
  --cfg_pkl /app/checkpoints/cfg.pkl \
  --host 0.0.0.0 --port 8000 \
  --online-sampling-steps 10 \
  --online-chunk-config 3,5,2 \
  --frame-scale 0.3

# ログは標準出力に出ます。停止は Ctrl+C。
```

## クライアント（同コンテナ内で最小確認）
```bash
# 例: 30 秒受信して統計を表示
python3 src/streaming_client.py \
  --server http://127.0.0.1:8000 \
  --audio_path /app/data/sample.wav \
  --source_path /app/data/sample.png \
  --frame-scale 0.3 \
  --sampling-timesteps 10 \
  --timeout 30
```

## ブラウザデモ（オプション）
- 任意の WebRTC ビューアから `/webrtc/offer` に SDP offer を POST し、answer をセット。
- サーバーは recvonly トランシーバーを広告するので、クライアント側は sendonly/recvonly いずれも可。

## Mac (CPU) での事前ヘルスチェック
- 実パイプラインなしで API/シグナリングの健全性だけ確認する簡易スモーク:
```bash
PYTHONPATH=src:. ./.venv/bin/pytest -q src/tests/test_e2e_smoke.py::test_offer_smoke_frame_scale
```
- これはスタブを使うため GPU / CUDA は不要。実ストリームは行われません。

## 既知の注意点
- 本番ストリーミングには実際のモデルと GPU が必須です。スタブ実行ではメディア生成はスキップされます。
- `UPLOAD_DIR` をカスタムする場合は環境変数で指定可能（未設定なら `/tmp/uploads`）。
- サーバー停止後は `docker compose down` で GPU 資源を解放してください。
