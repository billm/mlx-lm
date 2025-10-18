#!/usr/bin/env python3
"""
Example script demonstrating concurrent requests to the batched MLX-LM server.

This script shows how multiple clients can send requests concurrently to the
server running in batched mode, achieving better throughput than processing
requests sequentially.

Usage:
    1. Start the server in batched mode:
       mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --batched

    2. Run this script:
       python examples/batched_client.py
"""

import concurrent.futures
import json
import time

import requests


def make_completion_request(prompt, server_url="http://localhost:8080", max_tokens=50):
    """
    Send a completion request to the server.

    Args:
        prompt: Text prompt
        server_url: Server URL
        max_tokens: Maximum tokens to generate

    Returns:
        Response dictionary
    """
    start_time = time.time()

    response = requests.post(
        f"{server_url}/v1/completions",
        json={
            "model": "default_model",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
    )

    elapsed = time.time() - start_time
    result = response.json()

    return {
        "prompt": prompt,
        "text": result["choices"][0]["text"],
        "elapsed": elapsed,
        "usage": result.get("usage", {}),
    }


def make_streaming_request(
    prompt, server_url="http://localhost:8080", max_tokens=50
):
    """
    Send a streaming completion request to the server.

    Args:
        prompt: Text prompt
        server_url: Server URL
        max_tokens: Maximum tokens to generate

    Returns:
        Response dictionary with collected text
    """
    start_time = time.time()
    collected_text = ""
    first_token_time = None

    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": "default_model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True,
        },
        stream=True,
    )

    for chunk in response.iter_lines():
        if chunk:
            data = chunk.decode("utf-8")
            if data.startswith("data: ") and data != "data: [DONE]":
                chunk_data = json.loads(data[6:])
                delta = chunk_data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    collected_text += content

    elapsed = time.time() - start_time

    return {
        "prompt": prompt,
        "text": collected_text,
        "elapsed": elapsed,
        "time_to_first_token": first_token_time,
    }


def sequential_requests():
    """Run requests sequentially (baseline)."""
    print("\n=== Sequential Requests ===")

    prompts = [
        "Write a haiku about computers.",
        "Explain quantum computing in simple terms.",
        "List three benefits of exercise.",
        "Describe the water cycle.",
        "What is photosynthesis?",
    ]

    start_time = time.time()
    results = []

    for prompt in prompts:
        result = make_completion_request(prompt)
        results.append(result)
        print(f"✓ Completed: {prompt[:40]}... ({result['elapsed']:.2f}s)")

    total_time = time.time() - start_time

    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average time per request: {total_time / len(prompts):.2f}s")

    return results, total_time


def concurrent_requests():
    """Run requests concurrently (batched mode)."""
    print("\n=== Concurrent Requests (Batched Mode) ===")

    prompts = [
        "Write a haiku about computers.",
        "Explain quantum computing in simple terms.",
        "List three benefits of exercise.",
        "Describe the water cycle.",
        "What is photosynthesis?",
    ]

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [executor.submit(make_completion_request, p) for p in prompts]
        results = [f.result() for f in futures]

    total_time = time.time() - start_time

    for result in results:
        print(f"✓ Completed: {result['prompt'][:40]}... ({result['elapsed']:.2f}s)")

    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average time per request: {total_time / len(prompts):.2f}s")
    print(f"Speedup: {results[0]['elapsed'] * len(prompts) / total_time:.2f}x")

    return results, total_time


def streaming_example():
    """Demonstrate streaming with batched mode."""
    print("\n=== Streaming Example ===")

    prompts = [
        "Tell me a short story about a robot.",
        "Explain machine learning briefly.",
        "What are the planets in our solar system?",
    ]

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [executor.submit(make_streaming_request, p) for p in prompts]
        results = [f.result() for f in futures]

    total_time = time.time() - start_time

    for result in results:
        ttft = result.get("time_to_first_token", 0)
        print(
            f"✓ {result['prompt'][:40]}... "
            f"(TTFT: {ttft:.2f}s, Total: {result['elapsed']:.2f}s)"
        )

    print(f"\nTotal time: {total_time:.2f}s")

    return results


def main():
    """Main function to run examples."""
    print("MLX-LM Batched Server Client Examples")
    print("=" * 50)
    print("\nMake sure the server is running with --batched flag:")
    print("  mlx_lm.server --model <model-path> --batched\n")

    # Check if server is available
    try:
        response = requests.get("http://localhost:8080/health")
        if response.status_code == 200:
            print("✓ Server is running\n")
        else:
            print("✗ Server health check failed")
            return
    except requests.ConnectionError:
        print("✗ Could not connect to server at http://localhost:8080")
        print("  Please start the server first.")
        return

    # Run examples
    try:
        # Example 1: Concurrent requests
        concurrent_results, concurrent_time = concurrent_requests()

        # Example 2: Streaming
        streaming_results = streaming_example()

        print("\n" + "=" * 50)
        print("Examples completed successfully!")
        print(
            "\nNote: Sequential execution is omitted in batched mode demo, "
            "but typically takes ~5x longer"
        )

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
