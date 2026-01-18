# C++ Bench Client
This is an optional C++ client to benchmark vLLM OpenAI server without Python client overhead.

## Build (Linux/WSL)

Prereqs:
- g++ (C++17)
- curl CLI

Example:

```bash
cd cpp/bench_client
./build.sh
```

## Run

```bash
./sedac_bench_client \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 \
  --max-tokens 128 \
  --warmup 1 \
  --repeat 5 \
  --json-out /tmp/cpp_bench.json
```
