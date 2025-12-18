#!/usr/bin/env python3
"""
A10 Performance Testing Script for LightTTS
Tests TTFA (Time To First Audio) and throughput
"""

import asyncio
import aiohttp
import time
import wave
import numpy as np
from pathlib import Path

async def test_ttfa(session, url, prompt_wav_path, text):
    """Test Time To First Audio"""
    start_time = time.time()

    # Read prompt audio
    with open(prompt_wav_path, 'rb') as f:
        prompt_audio = f.read()

    # Prepare request
    files = {
        'prompt_audio': ('prompt.wav', prompt_audio, 'audio/wav'),
    }
    data = {
        'text': text,
        'streaming': 'true',
        'language': 'zh'
    }

    first_chunk_time = None
    total_chunks = 0
    total_size = 0

    async with session.post(url, data=data, files=files) as response:
        if response.status == 200:
            async for chunk in response.content.iter_chunked(4096):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    ttfa = first_chunk_time - start_time
                    print(f"TTFA: {ttfa*1000:.2f}ms")
                total_chunks += 1
                total_size += len(chunk)

            total_time = time.time() - start_time
            throughput = total_size / total_time if total_time > 0 else 0
            print(f"Total time: {total_time*1000:.2f}ms")
            print(f"Total audio size: {total_size/1024:.2f}KB")
            print(f"Throughput: {throughput/1024:.2f}KB/s")

            return {
                'ttfa': ttfa,
                'total_time': total_time,
                'throughput': throughput,
                'chunks': total_chunks
            }
        else:
            print(f"Error: {response.status}")
            print(await response.text())
            return None

async def test_concurrent_requests(session, url, prompt_wav_path, text, num_requests=5):
    """Test concurrent request handling"""
    print(f"\nTesting {num_requests} concurrent requests...")

    tasks = []
    for i in range(num_requests):
        task = test_ttfa(session, url, prompt_wav_path, f"{text} (request {i+1})")
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Calculate statistics
    valid_results = [r for r in results if r is not None and isinstance(r, dict)]
    if valid_results:
        avg_ttfa = np.mean([r['ttfa'] for r in valid_results])
        avg_throughput = np.mean([r['throughput'] for r in valid_results])
        print(f"\nConcurrent Request Statistics:")
        print(f"  Average TTFA: {avg_ttfa*1000:.2f}ms")
        print(f"  Average Throughput: {avg_throughput/1024:.2f}KB/s")
        print(f"  Successful requests: {len(valid_results)}/{num_requests}")

async def main():
    # Test configuration
    url = "http://localhost:8080/inference_zero_shot"
    prompt_wav = "test/prompt.wav"  # You need to provide this
    test_text = "你好，这是A10性能测试。"

    # Check if prompt audio exists
    if not Path(prompt_wav).exists():
        print(f"Error: Prompt audio file {prompt_wav} not found!")
        print("Please provide a prompt WAV file for testing.")
        return

    async with aiohttp.ClientSession() as session:
        # Test server health
        try:
            async with session.get("http://localhost:8080/healthz") as resp:
                if resp.status != 200:
                    print("Server health check failed!")
                    return
        except Exception as e:
            print(f"Cannot connect to server: {e}")
            print("Please ensure the server is running on localhost:8080")
            return

        print("=== LightTTS A10 Performance Test ===\n")

        # Single request test
        print("1. Single Request Test:")
        result = await test_ttfa(session, url, prompt_wav, test_text)

        # Concurrent requests test
        print("\n2. Concurrent Requests Test:")
        await test_concurrent_requests(session, url, prompt_wav, test_text, num_requests=3)

        # Load test
        print("\n3. Load Test (10 requests in parallel):")
        await test_concurrent_requests(session, url, prompt_wav, test_text, num_requests=10)

        print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(main())