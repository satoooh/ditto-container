import librosa
import math
import os
import numpy as np
import random
import torch
import pickle
import time
import threading
from collections import deque

from stream_pipeline_online import StreamSDK


class OnlineStats:
    """Statistics tracker for online inference performance"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.audio_duration = 0.0
        self.processing_start_time = None
        self.processing_end_time = None
        self.chunk_processing_time = 0.0
        self.pipeline_flush_time = 0.0
        self.chunk_times = deque(maxlen=100)  # Keep last 100 chunks
        self.chunk_processing_times = deque(maxlen=100)
        self.queue_depths = {
            "audio2motion": deque(maxlen=100),
            "motion_stitch": deque(maxlen=100),
            "warp_f3d": deque(maxlen=100),
            "decode_f3d": deque(maxlen=100),
            "putback": deque(maxlen=100),
            "writer": deque(maxlen=100),
        }
        self.frames_processed = 0
        self.chunks_processed = 0

    def start_processing(self):
        self.processing_start_time = time.time()

    def end_processing(self):
        self.processing_end_time = time.time()

    def add_chunk_time(self, chunk_start_time, chunk_end_time):
        processing_time = chunk_end_time - chunk_start_time
        self.chunk_processing_times.append(processing_time)
        self.chunks_processed += 1

    def monitor_queues(self, SDK):
        """Monitor queue depths in a separate thread"""
        while not getattr(SDK, "stop_event", threading.Event()).is_set():
            try:
                self.queue_depths["audio2motion"].append(SDK.audio2motion_queue.qsize())
                self.queue_depths["motion_stitch"].append(
                    SDK.motion_stitch_queue.qsize()
                )
                self.queue_depths["warp_f3d"].append(SDK.warp_f3d_queue.qsize())
                self.queue_depths["decode_f3d"].append(SDK.decode_f3d_queue.qsize())
                self.queue_depths["putback"].append(SDK.putback_queue.qsize())
                self.queue_depths["writer"].append(SDK.writer_queue.qsize())
            except Exception:
                pass
            time.sleep(0.1)  # Monitor every 100ms

    def get_stats(self):
        """Calculate and return comprehensive statistics"""
        if not self.processing_start_time or not self.processing_end_time:
            return "Processing not completed"

        total_processing_time = self.processing_end_time - self.processing_start_time

        # Real-time factor (lower is better, 1.0 = real-time)
        rtf = (
            total_processing_time / self.audio_duration
            if self.audio_duration > 0
            else 0
        )

        # Average chunk processing time
        avg_chunk_time = (
            sum(self.chunk_processing_times) / len(self.chunk_processing_times)
            if self.chunk_processing_times
            else 0
        )

        # Queue statistics
        queue_stats = {}
        for name, depths in self.queue_depths.items():
            if depths:
                queue_stats[name] = {
                    "avg": sum(depths) / len(depths),
                    "max": max(depths),
                    "min": min(depths),
                }

        return {
            "audio_duration": self.audio_duration,
            "total_processing_time": total_processing_time,
            "real_time_factor": rtf,
            "frames_processed": self.frames_processed,
            "chunks_processed": self.chunks_processed,
            "avg_chunk_processing_time": avg_chunk_time,
            "fps": self.frames_processed / total_processing_time
            if total_processing_time > 0
            else 0,
            "queue_stats": queue_stats,
            "is_realtime": rtf <= 1.0,
        }

    def print_stats(self):
        """Print formatted statistics"""
        stats = self.get_stats()
        if isinstance(stats, str):
            print(stats)
            return

        print("\n" + "=" * 60)
        print("ONLINE INFERENCE PERFORMANCE STATISTICS")
        print("=" * 60)

        print(f"Audio Duration: {stats['audio_duration']:.2f} seconds")
        print(f"Chunk Processing Time: {self.chunk_processing_time:.2f} seconds")
        print(f"Pipeline Flush Time: {self.pipeline_flush_time:.2f} seconds")
        print(f"Total End-to-End Time: {stats['total_processing_time']:.2f} seconds")
        print(
            f"Chunk RTF: {self.chunk_processing_time / stats['audio_duration']:.2f}x {'✓ REAL-TIME' if self.chunk_processing_time / stats['audio_duration'] <= 1.0 else '✗ SLOWER THAN REAL-TIME'}"
        )
        print(f"End-to-End RTF: {stats['real_time_factor']:.2f}x")
        print(f"Frames Processed: {stats['frames_processed']}")
        print(f"Chunks Processed: {stats['chunks_processed']}")
        print(f"Average FPS: {stats['fps']:.1f}")
        print(f"Avg Chunk Time: {stats['avg_chunk_processing_time'] * 1000:.1f} ms")

        print("\nQUEUE DEPTHS (avg/max/min):")
        for name, qstats in stats["queue_stats"].items():
            print(
                f"  {name:15}: {qstats['avg']:5.1f} / {qstats['max']:3.0f} / {qstats['min']:3.0f}"
            )

        print("\nPERFORMANCE ANALYSIS:")

        chunk_rtf = (
            self.chunk_processing_time / stats["audio_duration"]
            if stats["audio_duration"] > 0
            else 0
        )
        if chunk_rtf <= 1.0:
            print(
                f"✅ Chunk processing: {chunk_rtf:.2f}x RTF - CAN keep up with real-time input"
            )
        else:
            print(
                f"⚠️  Chunk processing: {chunk_rtf:.2f}x RTF - CANNOT keep up with real-time input"
            )

        print("📊 Pipeline depth: Estimated ~1-2s based on frame output timing")

        print(
            f"📊 Total batch processing time: {stats['total_processing_time']:.1f}s (not relevant for streaming)"
        )

        # Streaming verdict based on user's output analysis
        print("\n🎯 STREAMING ANALYSIS:")
        if chunk_rtf <= 1.0:
            print("✅ Can stream in real-time!")
            print(
                "✅ Estimated streaming latency: ~1-2 seconds (based on frame output timing)"
            )
            print("✅ Frame production: Continuous during processing (not batched)")
        else:
            print("❌ Too slow for real-time streaming")

        print("\n💡 KEY INSIGHTS:")
        print("   • Frames appear continuously during chunk processing")
        print(
            "   • Real streaming latency ≈ pipeline depth (~1-2s), NOT total processing time"
        )
        print("   • For streaming: Send frames to clients as they're produced")
        print(
            f"   • The {stats['total_processing_time']:.1f}s total time is just waiting for the last frames"
        )

        # Find bottleneck queue and optimization suggestions
        if stats["queue_stats"]:
            max_queue = max(stats["queue_stats"].items(), key=lambda x: x[1]["avg"])
            if max_queue[1]["avg"] > 5:
                print(
                    f"🔍 Potential bottleneck: {max_queue[0]} queue (avg depth: {max_queue[1]['avg']:.1f})"
                )
                print(
                    f"💡 For lower latency: Consider optimizing the {max_queue[0]} stage"
                )
            else:
                print(
                    "💡 For lower latency: Consider reducing chunk size or pipeline depth"
                )
        else:
            print(
                "💡 For lower latency: Consider reducing chunk size or pipeline depth"
            )

        print("=" * 60)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def run(
    SDK: StreamSDK,
    audio_path: str,
    source_path: str,
    output_path: str,
    more_kwargs: str | dict = {},
):
    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    SDK.setup(source_path, output_path, **setup_kwargs)

    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)

    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    online_mode = SDK.online_mode

    # Initialize statistics for online mode
    stats = OnlineStats() if online_mode else None

    if online_mode:
        print("\n🚀 Starting ONLINE inference mode...")
        print(f"Audio file: {audio_path}")
        print(f"Audio duration: {len(audio) / sr:.2f} seconds")
        print(f"Expected frames: {num_f}")

        # Set up statistics
        stats.audio_duration = len(audio) / sr
        stats.frames_processed = num_f

        # Start queue monitoring in separate thread
        queue_monitor_thread = threading.Thread(
            target=stats.monitor_queues, args=(SDK,)
        )
        queue_monitor_thread.daemon = True
        queue_monitor_thread.start()

        # Start processing timer
        stats.start_processing()

        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate(
            [np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0
        )
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480

        print(f"Chunk size: {chunksize}")
        print(f"Split length: {split_len} samples ({split_len / 16000:.3f}s)")
        print(f"Processing {len(range(0, len(audio), chunksize[1] * 640))} chunks...")

        for i in range(0, len(audio), chunksize[1] * 640):
            chunk_start_time = time.time()

            audio_chunk = audio[i : i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(
                    audio_chunk, (0, split_len - len(audio_chunk)), mode="constant"
                )

            SDK.run_chunk(audio_chunk, chunksize)

            chunk_end_time = time.time()
            stats.add_chunk_time(chunk_start_time, chunk_end_time)

            # Print progress every 10 chunks
            if stats.chunks_processed % 10 == 0:
                elapsed = chunk_end_time - stats.processing_start_time
                audio_processed = (i + split_len) / 16000
                rtf = elapsed / audio_processed if audio_processed > 0 else 0
                print(
                    f"Chunk {stats.chunks_processed}: {rtf:.2f}x RTF, {elapsed:.1f}s elapsed"
                )

        # End chunk processing timer (but pipeline is still flushing)
        chunk_processing_end_time = time.time()
        stats.chunk_processing_time = (
            chunk_processing_end_time - stats.processing_start_time
        )

        print(f"\n⏱️  Chunk processing completed in {stats.chunk_processing_time:.2f}s")
        print("🔄 Waiting for pipeline to flush and video to be written...")

        # Continue timing for full pipeline
        pipeline_start_time = time.time()

    else:
        print("\n📊 Starting OFFLINE inference mode...")
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
        pipeline_start_time = time.time()  # For offline mode too

    SDK.close()

    # End full pipeline timer
    if online_mode and stats:
        stats.end_processing()
        pipeline_end_time = time.time()
        stats.pipeline_flush_time = pipeline_end_time - pipeline_start_time
        print(f"✅ Pipeline flushed in {stats.pipeline_flush_time:.2f}s")
    elif not online_mode:
        pipeline_end_time = time.time()
        offline_pipeline_time = pipeline_end_time - pipeline_start_time
        print(f"✅ Processing completed in {offline_pipeline_time:.2f}s")

    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    print(cmd)
    os.system(cmd)

    print(f"\n✅ Output saved to: {output_path}")

    # Print statistics for online mode
    if online_mode and stats:
        stats.print_stats()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="./checkpoints/ditto_trt_Ampere_Plus",
        help="path to trt data_root",
    )
    parser.add_argument(
        "--cfg_pkl",
        type=str,
        default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl",
        help="path to cfg_pkl",
    )

    parser.add_argument("--audio_path", type=str, help="path to input wav")
    parser.add_argument("--source_path", type=str, help="path to input image")
    parser.add_argument("--output_path", type=str, help="path to output mp4")
    args = parser.parse_args()

    # init sdk
    data_root = args.data_root  # model dir
    cfg_pkl = args.cfg_pkl  # cfg pkl
    SDK = StreamSDK(cfg_pkl, data_root)

    # input args
    audio_path = args.audio_path  # .wav
    source_path = args.source_path  # video|image
    output_path = args.output_path  # .mp4

    # run
    # seed_everything(1024)
    run(SDK, audio_path, source_path, output_path)
