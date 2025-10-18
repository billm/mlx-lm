# Batched Serving Mode

This document describes the pipelined batching feature for the MLX-LM HTTP server, which enables concurrent processing of multiple client requests for improved throughput and latency.

## Overview

The batched serving mode uses a `BatchWorker` that owns a single `BatchGenerator` to process multiple requests concurrently. Unlike traditional batching that requires all requests to start at the same time, pipelined batching allows:

- **Dynamic admission**: New requests can be admitted at any time
- **Pipelined execution**: Prompt prefill and token generation are interleaved across requests
- **Streaming support**: Each request streams tokens as they are generated
- **Early stopping**: Requests can finish independently when they hit stop conditions

## How It Works

### Components

1. **BatchGenerator**: Core batching logic that manages multiple concurrent generations
   - Maintains an active batch of requests being processed
   - Queues unprocessed prompts waiting for capacity
   - Supports cancellation of specific requests

2. **BatchWorker**: Server-side thread that drives the BatchGenerator
   - Accepts request submissions via `submit()`
   - Continuously calls `BatchGenerator.next()` to process batches
   - Dispatches generated tokens back to individual request handlers
   - Supports cancellation via `cancel()`

3. **ClientContext**: Per-request state for batched mode
   - Token queue for streaming tokens to the handler
   - Finish event to signal completion
   - Finish reason (stop, length)

### Request Flow

1. Client sends HTTP request to `/v1/completions` or `/v1/chat/completions`
2. APIHandler tokenizes the prompt and creates a ClientContext
3. Request is submitted to BatchWorker via `submit()`
4. BatchWorker queues the request in BatchGenerator
5. BatchGenerator processes batches, interleaving prefill and generation
6. Tokens are dispatched to ClientContext.token_queue
7. APIHandler reads from queue and streams to client
8. When stop sequence is matched or max tokens reached, request finishes

### Cancellation

Requests can be canceled in two ways:
- **Client disconnect**: Handler detects broken pipe and calls `worker.cancel(uid)`
- **Stop sequence matched**: Handler matches stop sequence in tokens and calls `worker.cancel(uid)`

Canceled requests are removed from:
- Unprocessed prompt queue
- Active batch (if currently being processed)
- Worker's uid-to-context mapping

## Usage

### Starting the Server

To enable batched mode, use the `--batched` flag:

```bash
mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --batched
```

### Configuration Options

- `--batched`: Enable batched serving mode (default: off)
- `--completion-batch-size <int>`: Maximum concurrent decoding batch size (default: 32)
- `--prefill-batch-size <int>`: Maximum prompts to prefill when capacity is available (default: 8)
- `--prefill-step-size <int>`: Chunk size for long prompt prefill (default: 2048)

Example with custom settings:

```bash
mlx_lm.server \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --batched \
  --completion-batch-size 16 \
  --prefill-batch-size 4 \
  --prefill-step-size 1024
```

### Client Requests

Clients interact with the server exactly as before. The batched mode is transparent:

```python
import requests

# Streaming request
response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "model": "default_model",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": True,
        "max_tokens": 100,
    },
    stream=True,
)

for chunk in response.iter_lines():
    if chunk:
        data = chunk.decode("utf-8")
        if data.startswith("data: "):
            print(data)
```

### Concurrent Requests

The main benefit of batched mode is handling concurrent requests efficiently:

```python
import concurrent.futures
import requests

def make_request(prompt):
    response = requests.post(
        "http://localhost:8080/v1/completions",
        json={"model": "default_model", "prompt": prompt, "max_tokens": 50},
    )
    return response.json()

# Send 5 concurrent requests
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    prompts = [f"Tell me about topic {i}" for i in range(5)]
    futures = [executor.submit(make_request, p) for p in prompts]
    results = [f.result() for f in futures]
```

## Performance Considerations

### When to Use Batched Mode

Batched mode is most beneficial when:
- Multiple clients are making concurrent requests
- Request arrival rate is high
- Hardware can efficiently process multiple sequences (e.g., GPUs with high memory bandwidth)

### When to Use Non-Batched Mode

Non-batched mode may be better when:
- Only a single client at a time (no concurrency benefit)
- Very long prompts that fill device memory
- Models that don't benefit from batching (very large models on memory-constrained devices)

### Tuning Parameters

- **completion_batch_size**: Controls how many sequences decode together. Higher values increase throughput but require more memory.
- **prefill_batch_size**: Controls how many new prompts are processed at once. Higher values improve prefill efficiency but may delay completion tokens.
- **prefill_step_size**: Controls chunking of long prompts. Smaller values reduce memory peaks but increase overhead.

## Limitations

- **Single sampler config**: All requests use the same sampler parameters (temperature, top_p, etc.) configured at server startup or from CLI defaults. Per-request sampling parameters from the API are not yet supported in batched mode.
- **No prompt caching**: The batched mode does not currently integrate with the server's prompt cache feature.
- **No draft model**: Speculative decoding is not yet supported in batched mode.

## Implementation Details

### Thread Safety

The BatchWorker uses a lock to protect:
- BatchGenerator operations (insert, next, cancel)
- uid_to_ctx mapping

The lock is held only during critical operations to minimize contention.

### Memory Management

Memory is managed through MLX's `wired_limit` context manager, which adjusts the memory limit based on model size. The BatchWorker runs generation within this context.

### Error Handling

- Errors in the BatchWorker loop are logged but don't crash the worker
- Client disconnects are caught and requests are canceled gracefully
- Invalid requests fail early before submission to the worker

## Future Enhancements

Potential improvements for future versions:

1. **Per-request sampling**: Support different sampling parameters for each request
2. **Per-request logits processors**: Allow custom logits processors per request
3. **Prompt caching**: Integrate with the server's existing prompt cache
4. **Draft model support**: Enable speculative decoding in batched mode
5. **Dynamic batch sizing**: Automatically adjust batch size based on load
6. **Priority scheduling**: Allow high-priority requests to jump the queue
7. **Request timeouts**: Automatically cancel requests that take too long
