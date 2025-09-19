#!/usr/bin/env python3
"""
Script to run streaming tests with the Ditto server and client.
"""

import subprocess
import time
import sys
from pathlib import Path


def run_server():
    """Run the streaming server"""
    cmd = [
        sys.executable,
        "streaming_server.py",
        "--data_root",
        "../checkpoints/ditto_trt_Ampere_Plus/",
        "--cfg_pkl",
        "../checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]

    print(f"Starting server with command: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=Path(__file__).parent)


def run_client():
    """Run the streaming client"""
    cmd = [
        sys.executable,
        "streaming_client.py",
        "--server",
        "ws://localhost:8000",
        "--client_id",
        "test_client",
        "--audio_path",
        "./example/audio.wav",
        "--source_path",
        "./example/image.png",
        "--timeout",
        "30",
    ]

    print(f"Starting client with command: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=Path(__file__).parent)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run streaming test")
    parser.add_argument(
        "--mode",
        choices=["server", "client", "both"],
        default="both",
        help="Which component to run",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=3,
        help="Seconds to wait between starting server and client",
    )

    args = parser.parse_args()

    server_proc = None
    client_proc = None

    try:
        if args.mode in ["server", "both"]:
            server_proc = run_server()
            if args.mode == "both":
                print(f"Waiting {args.wait} seconds for server to start...")
                time.sleep(args.wait)

        if args.mode in ["client", "both"]:
            client_proc = run_client()

        # Wait for processes to complete
        if server_proc and client_proc:
            # Wait for client to complete
            client_proc.wait()
            print("Client completed")

            # Terminate server
            server_proc.terminate()
            server_proc.wait()
            print("Server terminated")

        elif server_proc:
            print("Server running... Press Ctrl+C to stop")
            server_proc.wait()

        elif client_proc:
            client_proc.wait()
            print("Client completed")

    except KeyboardInterrupt:
        print("Interrupted by user")
        if server_proc:
            server_proc.terminate()
        if client_proc:
            client_proc.terminate()

    except Exception as e:
        print(f"Error: {e}")
        if server_proc:
            server_proc.terminate()
        if client_proc:
            client_proc.terminate()


if __name__ == "__main__":
    main()
