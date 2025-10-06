# Remote WebRTC Streaming Client Setup

このドキュメントでは、WebRTC クライアント (`streaming_client.py`) をサーバーとは別マシンに配置して動かすための最低限の手順をまとめます。詳細はリポジトリ直下の `README.md` を参照してください。

## 1. ファイルのコピー

```bash
mkdir -p ~/ditto-client
scp streaming_client.py requirements-dev.txt -r example user@remote-host:~/ditto-client/
```

## 2. 依存ライブラリのインストール

リモート側で Python 3.10 以上を用意し、仮想環境の利用を推奨します。

```bash
cd ~/ditto-client
python -m venv venv
source venv/bin/activate  # Windows は venv\Scripts\activate
pip install -r requirements-dev.txt
```

`requirements-dev.txt` には WebRTC 実行に必要な `aiortc`, `av`, `aiohttp` などが含まれています。Linux 環境では `ffmpeg`, `libav*` 系パッケージがインストール済みであることを確認してください。

## 3. クライアントの実行例

```bash
python streaming_client.py \
  --server http://YOUR_SERVER_IP:8000 \
  --audio_path ./example/audio.wav \
  --source_path ./example/image.png \
  --frame-scale 0.5 \
  --sampling-timesteps 12 \
  --timeout 60 \
  --record-file remote.webm
```

- `--server` はサーバーの FastAPI エンドポイント。HTTPS 環境では `https://` を指定してください。
- `--record-file` を指定すると受信した映像を WebM で保存できます。
- 画質や帯域のチューニングは `--frame-scale`, `--sampling-timesteps`, `--chunk-config` で行います。

## 4. 補足

- サーバー側で `--frame-scale` などの既定値を指定していない場合、クライアントから送った値がそのまま反映されます。
- NAT 越えが必要な場合は STUN/TURN サーバーの導入が今後の課題です。現状は同一ネットワーク内での利用を想定しています。
- そのほかの注意事項・パラメータ一覧は `README.md` の「リアルタイムストリーミング（WebRTC）」セクションに記載しています。
