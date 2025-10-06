# Where: CLI-side streaming client invoked for manual diagnostics.
# What: Connects to Ditto streaming server, decodes frames, and aggregates runtime/system telemetry.
# Why: Provide actionable real-time stats (FPS, RTT, CPU/GPU, bandwidth) for troubleshooting.

import asyncio
import json
import time
import base64
import argparse
import logging
import subprocess
import shutil
from typing import List, Dict, Any, Optional, Deque, Tuple
from collections import deque
from dataclasses import dataclass
import statistics
import cv2
import numpy as np
from streaming_protocol import parse_binary_frame

import websockets
from websockets.exceptions import ConnectionClosed

try:
    import psutil
except ImportError:  # pragma: no cover - fallback path when psutil absent
    psutil = None


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@dataclass
class FrameStats:
    frame_id: int
    timestamp: float
    arrival_time: float
    frame_size: int
    decode_time: float


class StreamingStats:
    def __init__(self):
        self.frames: List[FrameStats] = []
        self.start_time: Optional[float] = None
        self.first_frame_time: Optional[float] = None
        self.last_frame_time: Optional[float] = None
        self.streaming_start_time: Optional[float] = None
        self.total_frames = 0
        self.total_bytes = 0
        self.frame_intervals = deque(maxlen=100)  # Keep last 100 intervals
        self.latency_samples: Deque[float] = deque(maxlen=200)
        self._net_io_start = None
        self._psutil_primed = False
        self._proc_cpu_prev: Optional[Dict[str, Any]] = None
        self._proc_net_prev: Optional[Tuple[int, int]] = None

    def mark_streaming_start(self):
        self.streaming_start_time = time.time()
        if psutil:
            try:
                self._net_io_start = psutil.net_io_counters()
                psutil.cpu_percent(interval=None)
                self._psutil_primed = True
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.debug(f"Failed to prime psutil metrics: {exc}")
                self._net_io_start = None
                self._psutil_primed = False
        else:
            self._net_io_start = None
            self._psutil_primed = False
            self._proc_cpu_prev = self._read_proc_cpu_totals()
            self._proc_net_prev = self._read_proc_net_counters()

    def record_latency(self, latency_seconds: Optional[float]):
        if latency_seconds is None:
            return
        try:
            latency = float(latency_seconds)
        except (TypeError, ValueError):
            return
        self.latency_samples.append(latency)
        
    def add_frame(self, frame_stats: FrameStats):
        self.frames.append(frame_stats)
        self.total_frames += 1
        self.total_bytes += frame_stats.frame_size
        
        current_time = frame_stats.arrival_time
        
        if self.first_frame_time is None:
            self.first_frame_time = current_time
        
        if self.last_frame_time is not None:
            interval = current_time - self.last_frame_time
            self.frame_intervals.append(interval)
        
        self.last_frame_time = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.frames:
            return {"error": "No frames received"}
        
        total_duration = self.last_frame_time - self.first_frame_time if self.first_frame_time and self.last_frame_time else 0
        
        # Calculate streaming latency (time from start to first frame)
        streaming_latency = self.first_frame_time - self.streaming_start_time if self.first_frame_time and self.streaming_start_time else 0
        
        # Frame interval statistics
        intervals = list(self.frame_intervals)
        
        # Decode time statistics
        decode_times = [f.decode_time for f in self.frames]
        latency_stats = self._build_latency_stats()
        system_metrics = self._collect_system_metrics()
        
        return {
            "total_frames": self.total_frames,
            "total_duration": total_duration,
            "streaming_latency": streaming_latency,
            "average_fps": self.total_frames / total_duration if total_duration > 0 else 0,
            "total_bytes": self.total_bytes,
            "average_frame_size": self.total_bytes / self.total_frames if self.total_frames > 0 else 0,
            "frame_intervals": {
                "count": len(intervals),
                "mean": statistics.mean(intervals) if intervals else 0,
                "median": statistics.median(intervals) if intervals else 0,
                "std": statistics.stdev(intervals) if len(intervals) > 1 else 0,
                "min": min(intervals) if intervals else 0,
                "max": max(intervals) if intervals else 0,
            },
            "decode_times": {
                "mean": statistics.mean(decode_times) if decode_times else 0,
                "median": statistics.median(decode_times) if decode_times else 0,
                "std": statistics.stdev(decode_times) if len(decode_times) > 1 else 0,
                "min": min(decode_times) if decode_times else 0,
                "max": max(decode_times) if decode_times else 0,
            },
            "bandwidth_mbps": (self.total_bytes * 8) / (total_duration * 1_000_000) if total_duration > 0 else 0,
            "latency": latency_stats,
            "system_metrics": system_metrics,
        }
    
    def print_stats(self):
        stats = self.get_stats()
        
        if "error" in stats:
            print(f"❌ {stats['error']}")
            return
            
        print("\n" + "="*70)
        print("STREAMING CLIENT STATISTICS")
        print("="*70)
        
        print(f"Total Frames Received: {stats['total_frames']}")
        print(f"Total Duration: {stats['total_duration']:.2f}s")
        print(f"Streaming Latency: {stats['streaming_latency']:.2f}s (time to first frame)")
        print(f"Average FPS: {stats['average_fps']:.1f}")
        print(f"Total Data: {stats['total_bytes']/1024/1024:.1f} MB")
        print(f"Average Frame Size: {stats['average_frame_size']/1024:.1f} KB")
        print(f"Bandwidth: {stats['bandwidth_mbps']:.1f} Mbps")
        
        print(f"\nFRAME INTERVALS:")
        intervals = stats['frame_intervals']
        print(f"  Count: {intervals['count']}")
        print(f"  Mean: {intervals['mean']*1000:.1f}ms")
        print(f"  Median: {intervals['median']*1000:.1f}ms") 
        print(f"  Std Dev: {intervals['std']*1000:.1f}ms")
        print(f"  Min: {intervals['min']*1000:.1f}ms")
        print(f"  Max: {intervals['max']*1000:.1f}ms")
        
        print(f"\nDECODE TIMES:")
        decode = stats['decode_times']
        print(f"  Mean: {decode['mean']*1000:.1f}ms")
        print(f"  Median: {decode['median']*1000:.1f}ms")
        print(f"  Std Dev: {decode['std']*1000:.1f}ms")
        print(f"  Min: {decode['min']*1000:.1f}ms")
        print(f"  Max: {decode['max']*1000:.1f}ms")

        latency = stats.get('latency', {})
        if latency.get('count', 0) > 0:
            print(f"\nRTT (Ping/Pong):")
            print(f"  Samples: {latency['count']}")
            print(f"  Latest: {latency['latest']*1000:.1f}ms")
            print(f"  Mean: {latency['mean']*1000:.1f}ms")
            print(f"  Median: {latency['median']*1000:.1f}ms")
            print(f"  Std Dev: {latency['std']*1000:.1f}ms")
            print(f"  Min: {latency['min']*1000:.1f}ms")
            print(f"  Max: {latency['max']*1000:.1f}ms")
        else:
            print("\nRTT (Ping/Pong): no samples collected")

        sys_metrics = stats.get('system_metrics', {})
        cpu_metrics = sys_metrics.get('cpu')
        if cpu_metrics:
            available = cpu_metrics.get('available', True)
            if available:
                print("\nCPU:")
                print(f"  Usage: {cpu_metrics['percent']:.1f}%")
                per_cpu = cpu_metrics.get('per_cpu')
                if per_cpu:
                    print(f"  Per CPU: {[round(v, 1) for v in per_cpu]}")
                else:
                    print("  Per CPU: n/a")
                if cpu_metrics.get('frequency_mhz') is not None:
                    print(f"  Frequency: {cpu_metrics['frequency_mhz']:.0f} MHz")
                if cpu_metrics.get('source'):
                    print(f"  Source: {cpu_metrics['source']}")
            else:
                print(f"\nCPU metrics unavailable ({cpu_metrics.get('reason')})")

        memory_metrics = sys_metrics.get('memory')
        if memory_metrics:
            if memory_metrics.get('available', True):
                print("\nMemory:")
                print(
                    f"  Used: {memory_metrics['used_gb']:.2f} GB / "
                    f"{memory_metrics['total_gb']:.2f} GB ({memory_metrics['percent']:.1f}% used)"
                )
                if memory_metrics.get('source'):
                    print(f"  Source: {memory_metrics['source']}")
            else:
                print(f"\nMemory metrics unavailable ({memory_metrics.get('reason')})")

        gpu_metrics = sys_metrics.get('gpu') or []
        if gpu_metrics:
            print("\nGPU:")
            for idx, gpu in enumerate(gpu_metrics):
                prefix = f"  GPU {idx}:" if len(gpu_metrics) > 1 else "  "
                print(
                    f"{prefix} {gpu['name']} | Util: {gpu['util']}% | "
                    f"Mem: {gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB | "
                    f"Temp: {gpu['temperature_c']}°C"
                )
        else:
            print("\nGPU: no data (nvidia-smi not available)")

        network_metrics = sys_metrics.get('network')
        if network_metrics:
            if network_metrics.get('available', True):
                print("\nNetwork IO:")
                print(f"  Bytes Sent: {network_metrics['bytes_sent']/1024/1024:.2f} MB")
                print(f"  Bytes Received: {network_metrics['bytes_recv']/1024/1024:.2f} MB")
                print(f"  Avg Throughput: {network_metrics['throughput_mbps']:.2f} Mbps")
                if network_metrics.get('duration'):
                    print(f"  Measurement Window: {network_metrics['duration']:.2f}s")
                if network_metrics.get('source'):
                    print(f"  Source: {network_metrics['source']}")
            else:
                print(f"\nNetwork IO unavailable ({network_metrics.get('reason')})")
        else:
            print("\nNetwork IO: not collected")
        
        print(f"\nREAL-TIME SUITABILITY:")
        target_fps = 25  # Assuming 25 FPS target
        target_interval = 1.0 / target_fps
        
        if stats['average_fps'] >= target_fps * 0.9:
            print(f"✅ Frame rate suitable for real-time ({stats['average_fps']:.1f} >= {target_fps*0.9:.1f} FPS)")
        else:
            print(f"⚠️  Frame rate below target ({stats['average_fps']:.1f} < {target_fps*0.9:.1f} FPS)")
        
        if stats['streaming_latency'] <= 3.0:
            print(f"✅ Streaming latency acceptable ({stats['streaming_latency']:.1f}s <= 3.0s)")
        else:
            print(f"⚠️  Streaming latency high ({stats['streaming_latency']:.1f}s > 3.0s)")
        
        if intervals['mean'] <= target_interval * 1.5:
            print(f"✅ Frame intervals consistent ({intervals['mean']*1000:.1f}ms <= {target_interval*1.5*1000:.1f}ms)")
        else:
            print(f"⚠️  Frame intervals inconsistent ({intervals['mean']*1000:.1f}ms > {target_interval*1.5*1000:.1f}ms)")
        
        print("="*70)

    def _build_latency_stats(self) -> Dict[str, Any]:
        samples = list(self.latency_samples)
        if not samples:
            return {
                "count": 0,
                "latest": None,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        return {
            "count": len(samples),
            "latest": samples[-1],
            "mean": statistics.mean(samples),
            "median": statistics.median(samples),
            "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "min": min(samples),
            "max": max(samples),
        }

    def _collect_system_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        metrics["cpu"] = self._cpu_metrics()
        metrics["memory"] = self._memory_metrics()
        metrics["gpu"] = self._gpu_metrics()
        network_metrics = self._network_metrics()
        if network_metrics:
            metrics["network"] = network_metrics
        return metrics

    def _cpu_metrics(self) -> Dict[str, Any]:
        if not psutil:
            return self._cpu_metrics_procfs()
        try:
            if not self._psutil_primed:
                psutil.cpu_percent(interval=None)
                self._psutil_primed = True
            per_cpu = psutil.cpu_percent(interval=None, percpu=True)
            overall = float(sum(per_cpu) / len(per_cpu)) if per_cpu else float(psutil.cpu_percent(interval=None))
            freq = psutil.cpu_freq()
            return {
                "available": True,
                "percent": overall,
                "per_cpu": per_cpu,
                "frequency_mhz": freq.current if freq else None,
                "logical_cores": psutil.cpu_count(logical=True),
                "source": "psutil",
            }
        except Exception as exc:  # pragma: no cover - diagnostics should not crash stats
            logger.debug(f"CPU metrics collection failed: {exc}")
            return self._cpu_metrics_procfs()

    def _memory_metrics(self) -> Optional[Dict[str, Any]]:
        if not psutil:
            return self._memory_metrics_procfs()
        try:
            mem = psutil.virtual_memory()
            return {
                "available": True,
                "total_gb": mem.total / (1024 ** 3),
                "used_gb": (mem.total - mem.available) / (1024 ** 3),
                "percent": mem.percent,
                "source": "psutil",
            }
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Memory metrics collection failed: {exc}")
            return self._memory_metrics_procfs()

    def _gpu_metrics(self) -> List[Dict[str, Any]]:
        if not shutil.which("nvidia-smi"):
            return []
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (subprocess.SubprocessError, FileNotFoundError) as exc:  # pragma: no cover
            logger.debug(f"GPU metrics collection failed: {exc}")
            return []

        gpus: List[Dict[str, Any]] = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 6:
                continue
            name, mem_total, mem_used, util_gpu, util_mem, temperature = parts[:6]
            gpus.append(
                {
                    "name": name,
                    "memory_total_mib": int(float(mem_total)),
                    "memory_used_mib": int(float(mem_used)),
                    "util": float(util_gpu),
                    "memory_util": float(util_mem),
                    "temperature_c": float(temperature),
                }
            )
        return gpus

    def _network_metrics(self) -> Optional[Dict[str, Any]]:
        if not self.streaming_start_time:
            return None
        if psutil and self._net_io_start:
            try:
                current = psutil.net_io_counters()
            except Exception as exc:  # pragma: no cover
                logger.debug(f"Network metrics collection failed: {exc}")
            else:
                duration = max(time.time() - self.streaming_start_time, 0.0)
                sent_delta = max(current.bytes_sent - self._net_io_start.bytes_sent, 0)
                recv_delta = max(current.bytes_recv - self._net_io_start.bytes_recv, 0)
                total_bytes = sent_delta + recv_delta
                throughput_mbps = (total_bytes * 8) / (duration * 1_000_000) if duration > 0 else 0.0
                return {
                    "available": True,
                    "bytes_sent": sent_delta,
                    "bytes_recv": recv_delta,
                    "throughput_mbps": throughput_mbps,
                    "duration": duration,
                    "source": "psutil",
                }
        return self._network_metrics_procfs()

    def _cpu_metrics_procfs(self) -> Dict[str, Any]:
        snapshot = self._read_proc_cpu_totals()
        if snapshot is None:
            return {"available": False, "reason": "procfs not accessible"}

        if self._proc_cpu_prev is None:
            self._proc_cpu_prev = snapshot
            return {"available": False, "reason": "collecting baseline"}

        prev = self._proc_cpu_prev
        self._proc_cpu_prev = snapshot

        usage = self._compute_cpu_usage(prev, snapshot)
        return usage

    def _compute_cpu_usage(self, prev: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        overall_prev = prev.get("overall")
        overall_curr = current.get("overall")
        if not overall_prev or not overall_curr:
            return {"available": False, "reason": "invalid procfs data"}

        total_diff = overall_curr[0] - overall_prev[0]
        idle_diff = overall_curr[1] - overall_prev[1]
        if total_diff <= 0:
            percent = 0.0
        else:
            percent = max(0.0, min(100.0, (1 - idle_diff / total_diff) * 100.0))

        per_cpu_usages: List[float] = []
        prev_list = prev.get("per_cpu", [])
        curr_list = current.get("per_cpu", [])
        for idx, curr_entry in enumerate(curr_list):
            try:
                prev_entry = prev_list[idx]
            except IndexError:
                continue
            total = curr_entry[0] - prev_entry[0]
            idle = curr_entry[1] - prev_entry[1]
            if total <= 0:
                per_cpu_usages.append(0.0)
            else:
                per_cpu_usages.append(max(0.0, min(100.0, (1 - idle / total) * 100.0)))

        return {
            "available": True,
            "percent": percent,
            "per_cpu": per_cpu_usages,
            "frequency_mhz": None,
            "logical_cores": len(per_cpu_usages) if per_cpu_usages else None,
            "source": "procfs",
        }

    def _memory_metrics_procfs(self) -> Dict[str, Any]:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as meminfo:
                data = meminfo.read().splitlines()
        except OSError as exc:
            logger.debug(f"Memory metrics (procfs) failed: {exc}")
            return {"available": False, "reason": "procfs not accessible"}

        info = {}
        for line in data:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()

        try:
            total_kib = float(info["MemTotal"].split()[0])
            available_kib = float(info.get("MemAvailable", info.get("MemFree")).split()[0])
        except (KeyError, ValueError, AttributeError) as exc:
            logger.debug(f"Memory metrics parse failed: {exc}")
            return {"available": False, "reason": "meminfo parse error"}

        used_kib = total_kib - available_kib
        percent = (used_kib / total_kib) * 100 if total_kib else 0.0

        return {
            "available": True,
            "total_gb": total_kib / (1024 ** 2),
            "used_gb": used_kib / (1024 ** 2),
            "percent": percent,
            "source": "procfs",
        }

    def _network_metrics_procfs(self) -> Optional[Dict[str, Any]]:
        current = self._read_proc_net_counters()
        if current is None:
            return {"available": False, "reason": "procfs not accessible"}
        if self._proc_net_prev is None:
            self._proc_net_prev = current
            return {"available": False, "reason": "collecting baseline"}

        prev_rx, prev_tx = self._proc_net_prev
        self._proc_net_prev = current

        rx_diff = max(current[0] - prev_rx, 0)
        tx_diff = max(current[1] - prev_tx, 0)
        duration = max(time.time() - self.streaming_start_time, 0.0)
        total_bytes = rx_diff + tx_diff
        throughput_mbps = (total_bytes * 8) / (duration * 1_000_000) if duration > 0 else 0.0

        return {
            "available": True,
            "bytes_sent": tx_diff,
            "bytes_recv": rx_diff,
            "throughput_mbps": throughput_mbps,
            "duration": duration,
            "source": "procfs",
        }

    def _read_proc_cpu_totals(self) -> Optional[Dict[str, Any]]:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as stat_file:
                lines = stat_file.readlines()
        except OSError as exc:
            logger.debug(f"CPU metrics (procfs) failed: {exc}")
            return None

        if not lines:
            return None

        def parse_cpu_line(line: str) -> Optional[Tuple[float, float]]:
            parts = line.strip().split()
            if not parts or not parts[0].startswith("cpu"):
                return None
            try:
                values = [float(v) for v in parts[1:]]
            except ValueError:
                return None
            if len(values) < 4:
                return None
            total = sum(values)
            idle = values[3] + values[4] if len(values) > 4 else values[3]
            return total, idle

        overall_line = parse_cpu_line(lines[0])
        if overall_line is None:
            return None

        per_cpu_totals: List[Tuple[float, float]] = []
        for line in lines[1:]:
            parsed = parse_cpu_line(line)
            if parsed is None:
                continue
            per_cpu_totals.append(parsed)

        return {
            "overall": (overall_line[0], overall_line[1]),
            "per_cpu": per_cpu_totals,
        }

    def _read_proc_net_counters(self) -> Optional[Tuple[int, int]]:
        try:
            with open("/proc/net/dev", "r", encoding="utf-8") as net_file:
                lines = net_file.readlines()
        except OSError as exc:
            logger.debug(f"Network metrics (procfs) failed: {exc}")
            return None

        rx_total = 0
        tx_total = 0

        for line in lines[2:]:  # skip headers
            if ":" not in line:
                continue
            interface, data = line.split(":", 1)
            fields = data.split()
            if len(fields) < 10:
                continue
            try:
                rx = int(fields[0])
                tx = int(fields[8])
            except ValueError:
                continue
            rx_total += rx
            tx_total += tx

        return (rx_total, tx_total)


class StreamingClient:
    def __init__(self, server_url: str, client_id: str, prefer_binary: bool = True):
        self.server_url = server_url
        self.client_id = client_id
        self.stats = StreamingStats()
        self.websocket = None
        self.save_frames = False
        self.frame_save_dir = "received_frames"
        self.prefer_binary = prefer_binary
        self.active_binary = prefer_binary
        
    async def connect(self):
        """Connect to the streaming server"""
        try:
            self.websocket = await websockets.connect(f"{self.server_url}/ws/{self.client_id}")
            logger.info(f"Connected to server as client {self.client_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
    
    async def start_streaming(self, audio_path: str, source_path: str, config: Dict[str, Any] = None):
        """Send start streaming command to server"""
        if not self.websocket:
            logger.error("Not connected to server")
            return False
        
        if config is None:
            config = {}
        
        message = {
            "type": "start_streaming",
            "audio_path": audio_path,
            "source_path": source_path,
            "setup_kwargs": config.get("setup_kwargs", {}),
            "run_kwargs": config.get("run_kwargs", {}),
            "binary": self.prefer_binary
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            logger.info("Sent start streaming command")
            self.stats.mark_streaming_start()
            return True
        except Exception as e:
            logger.error(f"Failed to send start streaming command: {e}")
            return False
    
    async def stop_streaming(self):
        """Send stop streaming command to server"""
        if not self.websocket:
            return False
        
        try:
            await self.websocket.send(json.dumps({"type": "stop_streaming"}))
            logger.info("Sent stop streaming command")
            return True
        except Exception as e:
            logger.error(f"Failed to send stop streaming command: {e}")
            return False
    
    async def handle_message(self, message: Dict[str, Any]):
        """Handle incoming messages from server"""
        message_type = message.get("type")
        
        if message_type == "frame":
            await self.handle_json_frame(message)
        elif message_type == "streaming_started":
            self.active_binary = bool(message.get('binary', self.prefer_binary))
            logger.info(
                f"Streaming started: {message.get('audio_path')} -> {message.get('source_path')} "
                f"(binary={self.active_binary})"
            )
        elif message_type == "metadata":
            logger.info(f"Metadata: {message.get('audio_duration'):.2f}s, {message.get('expected_frames')} frames")
        elif message_type == "streaming_completed":
            logger.info("Streaming completed")
        elif message_type == "writer_closed":
            logger.info(f"Writer closed. Total frames: {message.get('total_frames')}")
        elif message_type == "error":
            logger.error(f"Server error: {message.get('message')}")
        elif message_type == "pong":
            logger.debug("Received pong")
        else:
            logger.debug(f"Unknown message type: {message_type}")
    
    async def handle_json_frame(self, message: Dict[str, Any]):
        """Handle incoming frame"""
        frame_id = message.get("frame_id")
        frame_data = message.get("frame_data")
        timestamp = message.get("timestamp")
        try:
            frame_bytes = base64.b64decode(frame_data)
        except Exception as e:
            logger.error(f"Error decoding frame {frame_id}: {e}")
            return

        self._record_frame(frame_id, timestamp, frame_bytes)

    async def handle_binary_frame(self, payload: bytes):
        try:
            frame_id, timestamp, frame_bytes = parse_binary_frame(payload)
        except ValueError as exc:
            logger.error(f"Invalid binary frame received: {exc}")
            return

        self._record_frame(frame_id, timestamp, frame_bytes)

    def _record_frame(self, frame_id: int, timestamp: float, frame_bytes: bytes) -> None:
        arrival_time = time.time()

        decode_start = time.time()
        frame_size = len(frame_bytes)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame_img = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        decode_time = time.time() - decode_start

        if frame_img is None:
            logger.warning(f"Failed to decode frame {frame_id}")
            return

        if self.save_frames:
            import os

            os.makedirs(self.frame_save_dir, exist_ok=True)
            cv2.imwrite(f"{self.frame_save_dir}/frame_{frame_id:06d}.webp", frame_img)

        frame_stats = FrameStats(
            frame_id=frame_id,
            timestamp=timestamp,
            arrival_time=arrival_time,
            frame_size=frame_size,
            decode_time=decode_time
        )

        self.stats.add_frame(frame_stats)

        if frame_id % 25 == 0:
            logger.info(f"Received frame {frame_id}, size: {frame_size/1024:.1f}KB")
    
    async def listen(self):
        """Listen for messages from server"""
        if not self.websocket:
            logger.error("Not connected to server")
            return
        
        try:
            while True:
                incoming = await self.websocket.recv()
                if isinstance(incoming, bytes):
                    await self.handle_binary_frame(incoming)
                else:
                    try:
                        message = json.loads(incoming)
                    except json.JSONDecodeError as exc:
                        logger.error(f"Failed to decode server message: {exc}")
                        continue
                    await self.handle_message(message)
        except ConnectionClosed:
            logger.info("Connection closed by server")
        except Exception as e:
            logger.error(f"Error listening to server: {e}")
    
    async def ping(self):
        """Send periodic ping to server"""
        if not self.websocket:
            return

        try:
            await self.websocket.send(json.dumps({"type": "ping"}))
        except Exception as e:
            logger.error(f"Error sending ping: {e}")

    async def close(self):
        """Close connection to server"""
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from server")

    async def monitor_latency(self, interval: float = 5.0):
        """Periodically sample WebSocket RTT using control ping/pong."""
        if not self.websocket:
            return

        try:
            while True:
                if not self.websocket or self.websocket.closed:
                    break

                try:
                    send_time = time.perf_counter()
                    pong_waiter = await self.websocket.ping()
                    latency = await pong_waiter
                    if not latency and hasattr(self.websocket, "latency"):
                        latency = getattr(self.websocket, "latency")
                    if not latency:
                        latency = time.perf_counter() - send_time
                    self.stats.record_latency(latency)
                except ConnectionClosed:
                    break
                except Exception as exc:
                    logger.debug(f"Latency probe failed: {exc}")
                    break

                await asyncio.sleep(interval)
        finally:
            # ensure we capture final latency snapshot if websocket exposes it
            if self.websocket and hasattr(self.websocket, "latency"):
                self.stats.record_latency(getattr(self.websocket, "latency"))


async def main():
    parser = argparse.ArgumentParser(description="Ditto Streaming Client")
    parser.add_argument("--server", type=str, default="ws://localhost:8000", 
                      help="Server WebSocket URL")
    parser.add_argument("--client_id", type=str, default="test_client",
                      help="Client ID")
    parser.add_argument("--audio_path", type=str, default="./example/audio.wav",
                      help="Audio file path on server")
    parser.add_argument("--source_path", type=str, default="./example/image.png",
                      help="Source image path on server")
    parser.add_argument("--save_frames", action="store_true",
                      help="Save received frames to disk")
    parser.add_argument("--timeout", type=int, default=60,
                      help="Connection timeout in seconds")
    parser.add_argument("--transport", choices=["binary", "json"], default="binary",
                      help="Preferred frame transport (default: binary)")
    
    args = parser.parse_args()
    
    # Create client
    client = StreamingClient(args.server, args.client_id, prefer_binary=(args.transport == "binary"))
    client.save_frames = args.save_frames
    
    # Connect to server
    if not await client.connect():
        return
    
    try:
        # Start streaming
        streaming_config = {
            "setup_kwargs": {},
            "run_kwargs": {
                "chunksize": (3, 5, 2),
                "fade_in": -1,
                "fade_out": -1
            }
        }
        
        if not await client.start_streaming(args.audio_path, args.source_path, streaming_config):
            return
        
        listen_task = asyncio.create_task(client.listen())
        latency_task = asyncio.create_task(client.monitor_latency())

        # Listen for messages with timeout
        try:
            await asyncio.wait_for(listen_task, timeout=args.timeout)
        except asyncio.TimeoutError:
            logger.info("Timeout reached, stopping...")
        finally:
            latency_task.cancel()
            listen_task.cancel()
            await asyncio.gather(listen_task, latency_task, return_exceptions=True)

        # Print final statistics
        client.stats.print_stats()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main()) 
