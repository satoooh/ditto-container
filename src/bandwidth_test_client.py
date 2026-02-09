#!/usr/bin/env python3
"""
Bandwidth Test Client
Connects to bandwidth test server to measure WebSocket throughput.
Usage: python src/bandwidth_test_client.py --server ws://<server-ip>:8001 --client_id test_client

Operational guide: docs/BANDWIDTH_TEST_GUIDE.md
"""

import asyncio
import json
import time
import logging
import argparse
import websockets
from websockets.exceptions import ConnectionClosed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BandwidthTestClient:
    def __init__(self, server_url: str, client_id: str):
        self.server_url = server_url
        self.client_id = client_id
        self.websocket = None
        self.test_results = {}

    async def connect(self):
        """Connect to the bandwidth test server"""
        try:
            url = f"{self.server_url}/bandwidth_test/{self.client_id}"
            logger.info(f"🔌 Connecting to {url}...")

            self.websocket = await websockets.connect(url)
            logger.info(f"✅ Connected to bandwidth test server as {self.client_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            return False

    async def handle_server_to_client_test(self):
        """Handle server→client bandwidth test"""
        logger.info("📡 Starting server→client test...")

        test_sizes = [1, 10, 25, 50, 75, 100, 150]  # KB

        # Progressive chunk size test
        for size_kb in test_sizes:
            logger.info(f"📊 Expecting {size_kb}KB chunk...")

            try:
                # Receive test data
                start_time = time.time()
                data = await self.websocket.recv()
                end_time = time.time()

                # Send acknowledgment
                await self.websocket.send("ack")

                duration = end_time - start_time
                mbps = (len(data) * 8) / (duration * 1_000_000)

                logger.info(
                    f"✅ Received {len(data)} bytes in {duration * 1000:.1f}ms = {mbps:.2f} Mbps"
                )

            except Exception as e:
                logger.error(f"❌ Error receiving {size_kb}KB: {e}")
                await self.websocket.send("error")

        # Sustained throughput test
        logger.info("📊 Starting sustained test (10 seconds)...")
        total_bytes = 0
        chunk_count = 0
        start_time = time.time()

        try:
            while True:
                try:
                    data = await asyncio.wait_for(self.websocket.recv(), timeout=0.5)

                    if isinstance(data, str) and data == "sustained_test_complete":
                        # Send final ack
                        await self.websocket.send("ack")
                        break

                    total_bytes += len(data)
                    chunk_count += 1

                    # Send ack every 10 chunks
                    if chunk_count % 10 == 0:
                        await self.websocket.send("ack")

                except asyncio.TimeoutError:
                    # Check if test is still running
                    if time.time() - start_time > 12:  # 10s test + 2s buffer
                        break
                    continue

            end_time = time.time()
            duration = end_time - start_time
            sustained_mbps = (total_bytes * 8) / (duration * 1_000_000)

            logger.info("✅ Sustained test complete:")
            logger.info(
                f"   📊 Received: {total_bytes / (1024 * 1024):.1f} MB in {duration:.1f}s"
            )
            logger.info(f"   📊 Bandwidth: {sustained_mbps:.2f} Mbps")
            logger.info(f"   📊 Chunks: {chunk_count}")

        except Exception as e:
            logger.error(f"❌ Sustained test error: {e}")

    async def handle_client_to_server_test(self):
        """Handle client→server bandwidth test"""
        logger.info("📡 Starting client→server test...")

        # Wait for server signal
        signal = await self.websocket.recv()
        if signal != "start_client_to_server_test":
            logger.error(f"❌ Unexpected signal: {signal}")
            return

        test_sizes = [1, 10, 25, 50, 75, 100, 150]  # KB

        # Progressive chunk size test
        for size_kb in test_sizes:
            logger.info(f"📊 Sending {size_kb}KB chunk...")

            try:
                test_data = b"x" * (size_kb * 1024)

                start_time = time.time()
                await self.websocket.send(test_data)

                # Wait for acknowledgment
                ack = await self.websocket.recv()
                end_time = time.time()

                if ack == "ack":
                    duration = end_time - start_time
                    mbps = (len(test_data) * 8) / (duration * 1_000_000)
                    logger.info(
                        f"✅ Sent {len(test_data)} bytes in {duration * 1000:.1f}ms = {mbps:.2f} Mbps"
                    )
                else:
                    logger.error(f"❌ No ack received for {size_kb}KB")

            except Exception as e:
                logger.error(f"❌ Error sending {size_kb}KB: {e}")

        # Sustained throughput test
        logger.info("📊 Starting sustained upload test (10 seconds)...")
        chunk_size = 100 * 1024  # 100KB chunks
        total_bytes = 0
        start_time = time.time()
        chunk_count = 0

        try:
            while time.time() - start_time < 10:
                test_data = b"x" * chunk_size
                await self.websocket.send(test_data)
                total_bytes += len(test_data)
                chunk_count += 1

                # Wait for ack every 10 chunks
                if chunk_count % 10 == 0:
                    ack = await self.websocket.recv()
                    if ack != "ack":
                        logger.warning("⚠️ Missing ack during sustained upload")

            # Send completion signal
            await self.websocket.send("sustained_test_complete")
            ack = await self.websocket.recv()

            end_time = time.time()
            duration = end_time - start_time
            sustained_mbps = (total_bytes * 8) / (duration * 1_000_000)

            logger.info("✅ Sustained upload complete:")
            logger.info(
                f"   📊 Sent: {total_bytes / (1024 * 1024):.1f} MB in {duration:.1f}s"
            )
            logger.info(f"   📊 Bandwidth: {sustained_mbps:.2f} Mbps")
            logger.info(f"   📊 Chunks: {chunk_count}")

        except Exception as e:
            logger.error(f"❌ Sustained upload error: {e}")

    async def run_test(self):
        """Run the complete bandwidth test"""
        if not await self.connect():
            return False

        try:
            logger.info("🧪 Starting bandwidth test...")

            # Handle server→client test
            await self.handle_server_to_client_test()

            # Small delay between tests
            await asyncio.sleep(1)

            # Handle client→server test
            await self.handle_client_to_server_test()

            # Wait for final results
            logger.info("⏳ Waiting for test results...")
            while True:
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=10)
                    data = json.loads(message)

                    if data.get("type") == "test_complete":
                        self.test_results = data.get("results", {})
                        logger.info("✅ Bandwidth test completed!")
                        self.print_client_summary()
                        break
                    elif data.get("type") == "test_error":
                        logger.error(f"❌ Test error: {data.get('error')}")
                        break

                except asyncio.TimeoutError:
                    logger.warning("⏰ Timeout waiting for results")
                    break
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Non-JSON message received: {message}")
                    continue

            return True

        except ConnectionClosed:
            logger.info("🔌 Connection closed by server")
            return False
        except Exception as e:
            logger.error(f"❌ Test error: {e}")
            return False
        finally:
            if self.websocket:
                await self.websocket.close()

    def print_client_summary(self):
        """Print client-side test summary"""
        if not self.test_results:
            logger.warning("⚠️ No test results to display")
            return

        print("\n" + "=" * 60)
        print("BANDWIDTH TEST RESULTS (CLIENT VIEW)")
        print("=" * 60)

        # Server to Client results
        s2c = self.test_results.get("server_to_client", {})
        print("\n📡 SERVER → CLIENT (Download):")
        if s2c:
            for size, data in s2c.items():
                if isinstance(data, dict) and "mbps" in data:
                    print(
                        f"  {size:>10}: {data['mbps']:>8.2f} Mbps ({data['duration_ms']:>6.1f}ms)"
                    )
        else:
            print("  No results available")

        # Client to Server results
        c2s = self.test_results.get("client_to_server", {})
        print("\n📡 CLIENT → SERVER (Upload):")
        if c2s:
            for size, data in c2s.items():
                if isinstance(data, dict) and "mbps" in data:
                    print(
                        f"  {size:>10}: {data['mbps']:>8.2f} Mbps ({data['duration_ms']:>6.1f}ms)"
                    )
        else:
            print("  No results available")

        # Network analysis for streaming
        print("\n🎯 STREAMING FEASIBILITY:")
        if "sustained_10s" in s2c and "mbps" in s2c["sustained_10s"]:
            sustained_mbps = s2c["sustained_10s"]["mbps"]

            # Calculate streaming capacity for different frame sizes
            frame_sizes = [50, 100, 150, 200, 250]  # KB

            print(f"  Sustained Download: {sustained_mbps:.2f} Mbps")
            print("  Maximum FPS by frame size:")

            for frame_kb in frame_sizes:
                max_fps = (sustained_mbps * 1_000_000) / (frame_kb * 1024 * 8)
                status = "✅" if max_fps >= 30 else "⚠️" if max_fps >= 15 else "❌"
                print(f"    {frame_kb:3d}KB frames: {max_fps:5.1f} fps {status}")

            print("\n  Current streaming (46KB @ 40fps):")
            required_mbps = (46 * 1024 * 8 * 40) / 1_000_000
            print(f"    Required: {required_mbps:.1f} Mbps")
            print(f"    Available: {sustained_mbps:.1f} Mbps")

            if sustained_mbps >= required_mbps:
                print("    Status: ✅ Network supports current streaming")
            else:
                ratio = sustained_mbps / required_mbps
                print(
                    f"    Status: ❌ Network only supports {ratio * 100:.0f}% of required bandwidth"
                )
                print(
                    f"    Suggested: Reduce to {int(46 * ratio)}KB frames or {int(40 * ratio)}fps"
                )

        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Bandwidth Test Client")
    parser.add_argument(
        "--server",
        required=True,
        help="WebSocket server URL (e.g., ws://10.49.167.242:8001)",
    )
    parser.add_argument("--client_id", default="test_client", help="Client identifier")
    args = parser.parse_args()

    print("🚀 Starting Bandwidth Test Client")
    print(f"📡 Server: {args.server}")
    print(f"🆔 Client ID: {args.client_id}")

    client = BandwidthTestClient(args.server, args.client_id)
    success = await client.run_test()

    if success:
        print("\n✅ Bandwidth test completed successfully!")
    else:
        print("\n❌ Bandwidth test failed!")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
