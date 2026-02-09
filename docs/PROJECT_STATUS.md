# プロジェクトステータス（ditto-container）

## 1. 目的

Ditto TalkingHead を GPU コンテナ上で実行し、`FastAPI + aiortc` を用いてブラウザ/CLI に低遅延の映像・音声ストリーミングを提供する。

主要要件:
- シグナリング API: `POST /webrtc/offer`
- 入力経路: `/upload`（音声/画像）および CLI 指定
- セッション単位チューニング: `frame_scale`, `sampling_timesteps`, `chunk_config`, `chunk_sleep`

## 2. 進捗一覧（Kiro履歴 + Git履歴ベース）

### 2.1 仕様化・設計整理
- Kiro の仕様策定フローに沿って、要件整理→設計→タスク化までを実施。
- 低遅延 WebRTC 配信に必要な責務を、入力検証・接続監視・メディア同期・運用検証の観点で明文化。
- 仕様メモと実装メモを `_docs/` に継続記録し、設計判断の根拠を蓄積。

### 2.2 配信アーキテクチャの実装
- WebSocket 中心構成から WebRTC 中心構成へ移行。
- `streaming_server.py` に `/webrtc/offer`、`/upload`、`/demo` を整備。
- recvonly トランシーバーを明示し、ブラウザ/CLI 受信の互換性を改善。
- 音声 16kHz→48kHz リサンプル、映像フレーム送出、A/V 同期をトラック層へ実装。

### 2.3 入力検証・エラーハンドリング
- offer payload の必須項目・拡張子・パス存在チェックを追加。
- `streaming_config.py` を中心に、パラメータ正規化と安全なクランプを導入。
- エラー応答 JSON 形式を統一し、異常時の挙動を明確化。
- アップロード時の失敗ロールバック（保存途中失敗時の掃除）を整備。

### 2.4 接続監視・運用可観測性
- `webrtc/monitor.py` で `connectionState` / `iceConnectionState` 監視を導入。
- `webrtc/metrics.py` と `webrtc/tracks.py` にドリフト計測と PTS 管理を実装。
- クライアント側統計（FPS/遅延/システム情報）を拡充し、検証時の観測情報を増強。

### 2.5 Docker / 依存関係の調整
- `docker-compose.yml` を `network_mode: host` 構成へ変更。
- Dockerfile を TensorRT ベース運用に寄せ、CUDA/PyTorch/TensorRT 周辺の依存を段階的に調整。
- numpy 競合回避のための pin / constraint を導入。
- TensorRT 取得失敗時の原因と対処を `_docs/2025-11-25_tensorrt_index.md` に記録。

### 2.6 テスト整備
- `src/tests/` に offer/upload/接続監視/ドリフト/クランプ/cleanup 系テストを追加。
- GPU 非依存で検証可能なスタブ中心テストを整備。
- 現在のテスト実行結果: `44 passed, 1 warning`。

## 3. 現在の実装状態

主要構成:
- サーバ: `src/streaming_server.py`
- クライアント: `src/streaming_client.py`
- WebRTC ヘルパ: `src/webrtc/validators.py`, `src/webrtc/parameters.py`, `src/webrtc/monitor.py`, `src/webrtc/metrics.py`, `src/webrtc/tracks.py`
- 共通設定: `src/streaming_config.py`
- テスト: `src/tests/`

直近で反映済みの整合修正:
- `handle_offer` で SDP answer を即時返却するよう調整（接続待ちによる応答ブロックを回避）。
- `_run_streaming_pipeline` の引数不整合（`connection_ready`）を解消。
- `payload.dict()` を `payload.model_dump()` に置換（Pydantic v2 非推奨対応）。
- 上記挙動に合わせて関連テスト期待値を更新。

## 4. 既知課題と解決状況

### 4.1 numpy ABI 警告
- 状態: **未解決**
- 課題: `numpy 2.x` と Torch/拡張モジュールの ABI 不整合警告が残る。
- 現在の扱い: 暫定的に警告許容で運用。根治にはベースイメージ/依存固定戦略の再設計が必要。
- 検討中の対応案:
  - TensorRT ベースイメージから、CUDA 11.8 ベース + `tensorrt==8.6.1` 導入構成へ切替して依存を明示固定する。
  - もしくは `numpy 1.26.x` を前提に整合が取れる TensorRT イメージ/タグへ寄せる。

### 4.2 WebRTC 接続不安定（ICE / 応答受信）
- 状態: **部分対応（再検証待ち）**
- 課題:
  - ICE が `connecting` から進まないケースがある。
  - 実機報告として「リクエスト送信は可能だが、クライアント側でレスポンス受信できない」事象がある。
- 反映済み対応:
  - サーバ側で offer 応答のブロック条件を削除し、SDP answer を先に返す構成へ変更。
- 残課題:
  - STUN/TURN、NAT、実行環境依存要因を含む実機再検証が必要。
- 検証方針:
  - まず STUN なし（`RTCConfiguration` を空）でローカル CLI→サーバ接続の成立可否を確認する。
  - 外部接続が必要な経路では TURN（coturn）を導入し、`iceServers` に `turn:` を追加して再検証する。

## 5. 次アクション

1. GPU実機で `offer -> answer受信 -> ICE遷移 -> メディア受信` の再試験を実施し、再現条件を確定する。
2. STUN なしローカル優先構成と STUN/TURN 利用構成を切替可能にし、README と実装を同期する。
3. numpy/TensorRT の依存固定方針を確定し、ABI 警告を解消する。
4. 実機寄りの統合テスト（モック依存を減らした経路）を追加する。
5. 実装完了条件を満たした段階で、Kiro 側のステータス情報を更新する。
