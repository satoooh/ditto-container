#!/usr/bin/env python3
"""
Bandwidth Test Server
Tests WebSocket throughput between server and client to determine optimal streaming parameters.
Usage: python bandwidth_test_server.py --host 0.0.0.0 --port 8001
"""

import asyncio
import json
import time
import logging
import argparse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class BandwidthTester:
    def __init__(self):
        self.active_connections = {}

    async def test_server_to_client(self, websocket: WebSocket, client_id: str):
        """Test bandwidth from server to client"""
        logger.info(f"🚀 Starting server→client bandwidth test for {client_id}")
        results = {}

        # Test progressive chunk sizes - realistic for our 46KB frames
        test_sizes = [1, 10, 25, 50, 75, 100, 150]  # KB

        for size_kb in test_sizes:
            logger.info(f"📊 Testing {size_kb}KB chunks...")
            test_data = b"x" * (size_kb * 1024)

            # Send test chunk and measure time
            start_time = time.time()
            await websocket.send_bytes(test_data)

            # Wait for acknowledgment
            ack = await websocket.receive_text()
            end_time = time.time()

            if ack == "ack":
                duration = end_time - start_time
                mbps = (len(test_data) * 8) / (duration * 1_000_000)
                results[f"{size_kb}KB"] = {
                    "mbps": round(mbps, 2),
                    "duration_ms": round(duration * 1000, 1),
                }
                logger.info(
                    f"✅ {size_kb}KB: {mbps:.2f} Mbps ({duration * 1000:.1f}ms)"
                )
            else:
                logger.error(f"❌ No ack received for {size_kb}KB test")

        # Test sustained throughput (10 seconds)
        logger.info("📊 Testing sustained throughput (10 seconds)...")
        chunk_size = 100 * 1024  # 100KB chunks
        total_bytes = 0
        start_time = time.time()
        chunk_count = 0

        try:
            while time.time() - start_time < 10:
                test_data = b"x" * chunk_size
                await websocket.send_bytes(test_data)

                # Wait for ack every 10 chunks to avoid overwhelming
                chunk_count += 1
                if chunk_count % 10 == 0:
                    ack = await websocket.receive_text()
                    if ack != "ack":
                        logger.warning("⚠️ Missing ack during sustained test")

                total_bytes += len(test_data)

            # Final ack
            await websocket.send_text("sustained_test_complete")
            ack = await websocket.receive_text()

            end_time = time.time()
            duration = end_time - start_time
            sustained_mbps = (total_bytes * 8) / (duration * 1_000_000)

            results["sustained_10s"] = {
                "mbps": round(sustained_mbps, 2),
                "total_mb": round(total_bytes / (1024 * 1024), 1),
                "chunks_sent": chunk_count,
            }
            logger.info(
                f"✅ Sustained: {sustained_mbps:.2f} Mbps ({total_bytes / (1024 * 1024):.1f}MB total)"
            )

        except Exception as e:
            logger.error(f"❌ Sustained test error: {e}")
            results["sustained_10s"] = {"error": str(e)}

        return results

    async def test_client_to_server(self, websocket: WebSocket, client_id: str):
        """Test bandwidth from client to server"""
        logger.info(f"🚀 Starting client→server bandwidth test for {client_id}")
        results = {}

        # Signal client to start sending
        await websocket.send_text("start_client_to_server_test")

        test_sizes = [1, 10, 25, 50, 75, 100, 150]  # KB

        for size_kb in test_sizes:
            logger.info(f"📊 Expecting {size_kb}KB from client...")

            try:
                start_time = time.time()
                data = await websocket.receive_bytes()
                end_time = time.time()

                # Send acknowledgment
                await websocket.send_text("ack")

                duration = end_time - start_time
                mbps = (len(data) * 8) / (duration * 1_000_000)
                results[f"{size_kb}KB"] = {
                    "mbps": round(mbps, 2),
                    "duration_ms": round(duration * 1000, 1),
                    "received_bytes": len(data),
                }
                logger.info(
                    f"✅ {size_kb}KB: {mbps:.2f} Mbps ({duration * 1000:.1f}ms)"
                )

            except Exception as e:
                logger.error(f"❌ Error receiving {size_kb}KB: {e}")
                results[f"{size_kb}KB"] = {"error": str(e)}

        # Sustained test
        logger.info("📊 Testing sustained client→server (10 seconds)...")
        total_bytes = 0
        start_time = time.time()
        chunk_count = 0

        try:
            while time.time() - start_time < 10:
                data = await websocket.receive_bytes()
                total_bytes += len(data)
                chunk_count += 1

                # Send ack every 10 chunks
                if chunk_count % 10 == 0:
                    await websocket.send_text("ack")

            # Wait for completion signal
            await websocket.receive_text()
            await websocket.send_text("ack")

            end_time = time.time()
            duration = end_time - start_time
            sustained_mbps = (total_bytes * 8) / (duration * 1_000_000)

            results["sustained_10s"] = {
                "mbps": round(sustained_mbps, 2),
                "total_mb": round(total_bytes / (1024 * 1024), 1),
                "chunks_received": chunk_count,
            }
            logger.info(
                f"✅ Sustained: {sustained_mbps:.2f} Mbps ({total_bytes / (1024 * 1024):.1f}MB total)"
            )

        except Exception as e:
            logger.error(f"❌ Sustained test error: {e}")
            results["sustained_10s"] = {"error": str(e)}

        return results

    async def run_full_test(self, websocket: WebSocket, client_id: str):
        """Run complete bidirectional bandwidth test"""
        logger.info(f"🧪 Starting full bandwidth test for {client_id}")

        results = {
            "client_id": client_id,
            "timestamp": time.time(),
            "server_to_client": {},
            "client_to_server": {},
        }

        try:
            # Test 1: Server to Client
            results["server_to_client"] = await self.test_server_to_client(
                websocket, client_id
            )

            # Small delay between tests
            await asyncio.sleep(1)

            # Test 2: Client to Server
            results["client_to_server"] = await self.test_client_to_server(
                websocket, client_id
            )

            # Send final results
            await websocket.send_text(
                json.dumps({"type": "test_complete", "results": results})
            )

            logger.info(f"✅ Full bandwidth test completed for {client_id}")
            self.print_summary(results)

        except Exception as e:
            logger.error(f"❌ Bandwidth test failed for {client_id}: {e}")
            await websocket.send_text(
                json.dumps({"type": "test_error", "error": str(e)})
            )

        return results

    def print_summary(self, results):
        """Print human-readable test summary"""
        print("\n" + "=" * 60)
        print("BANDWIDTH TEST RESULTS SUMMARY")
        print("=" * 60)

        # Server to Client results
        s2c = results["server_to_client"]
        print("\n📡 SERVER → CLIENT:")
        for size, data in s2c.items():
            if isinstance(data, dict) and "mbps" in data:
                print(
                    f"  {size:>10}: {data['mbps']:>8.2f} Mbps ({data['duration_ms']:>6.1f}ms)"
                )

        # Client to Server results
        c2s = results["client_to_server"]
        print("\n📡 CLIENT → SERVER:")
        for size, data in c2s.items():
            if isinstance(data, dict) and "mbps" in data:
                print(
                    f"  {size:>10}: {data['mbps']:>8.2f} Mbps ({data['duration_ms']:>6.1f}ms)"
                )

        # Streaming capacity analysis
        print("\n🎯 STREAMING ANALYSIS:")
        if "sustained_10s" in s2c and "mbps" in s2c["sustained_10s"]:
            sustained_mbps = s2c["sustained_10s"]["mbps"]
            frame_size_kb = 200  # Current frame size
            max_fps = (sustained_mbps * 1_000_000) / (frame_size_kb * 1024 * 8)

            print(f"  Sustained bandwidth: {sustained_mbps:.2f} Mbps")
            print(f"  Current frame size: {frame_size_kb} KB")
            print(f"  Maximum FPS: {max_fps:.1f} fps")
            print("  Current target: 40 fps")

            if max_fps >= 40:
                print("  ✅ Network can handle 40 FPS streaming")
            else:
                print(f"  ⚠️  Network limited to {max_fps:.1f} FPS")
                print(
                    f"  💡 Reduce frame size to {int(frame_size_kb * max_fps / 40)} KB for 40 FPS"
                )

        print("=" * 60)


# FastAPI WebSocket endpoint
tester = BandwidthTester()


@app.websocket("/bandwidth_test/{client_id}")
async def bandwidth_test_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    logger.info(f"🔌 Client {client_id} connected for bandwidth test")

    try:
        tester.active_connections[client_id] = websocket
        await tester.run_full_test(websocket, client_id)

    except WebSocketDisconnect:
        logger.info(f"🔌 Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"❌ Error with client {client_id}: {e}")
    finally:
        if client_id in tester.active_connections:
            del tester.active_connections[client_id]


@app.get("/")
async def get_homepage():
    return {
        "message": "Bandwidth Test Server",
        "endpoint": "/bandwidth_test/{client_id}",
    }


def main():
    parser = argparse.ArgumentParser(description="Bandwidth Test Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    args = parser.parse_args()

    print(f"🚀 Starting Bandwidth Test Server on {args.host}:{args.port}")
    print(
        f"📡 Test endpoint: ws://{args.host}:{args.port}/bandwidth_test/{{client_id}}"
    )
    print("🔧 WebSocket max frame size: 5MB")

    # Configure WebSocket to allow up to 5MB frames for bandwidth testing
    uvicorn.run(app, host=args.host, port=args.port, ws_max_size=5 * 1024 * 1024)


if __name__ == "__main__":
    main()
