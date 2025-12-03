# TODO / Known Issues (2025-12-03)

- numpy 2.x がベースイメージにプリインストールされており、Torch/pybind 拡張と ABI が食い違う警告が出る。現状はサーバー起動はするが警告が残存。  
  - 対応案: TensorRT ベースイメージをやめて CUDA11.8 ベース + pip tensorrt 8.6.1 (NGC) に切り替える、もしくは numpy 1.26 系のタグを持つ別 TensorRT イメージを探す。  
  - 当面は警告を許容し、ストリーミング疎通検証を優先。
- WebRTC ICE が 15 秒で timeout/502 になる。STUN ありでも接続が "connecting" から進まない。  
  - 試験案: STUN 無し (RTCConfiguration 空) でローカル CLI → サーバー接続が通るか確認。  
  - 外部接続が必要な場合は TURN (coturn) を用意して `iceServers` に追加。

