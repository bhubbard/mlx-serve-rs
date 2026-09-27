# mlx-serve-rs 🚀⚡

[![Crates.io](https://img.shields.io/badge/crates.io-v0.0.1-orange.svg)](https://crates.io/crates/mlx-serve-rs)
[![Documentation](https://img.shields.io/badge/docs-GitHub_Pages-blue.svg)](http://code.brandonhubbard.com/mlx-serve-rs/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MLX_Accelerated-black.svg?logo=apple)](https://github.com/ml-explore/mlx)

High-performance OpenAI-compatible and Ollama-compatible LLM inference server for Apple Silicon in pure Rust using MLX.

A complete Rust port of [ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve), providing zero-overhead unified memory tensor execution, continuous Server-Sent Events (SSE) token streaming, and OpenAI API wire compatibility.

---

## 🚀 Features

- **OpenAI API Wire-Compatible**:
  - `POST /v1/chat/completions` (JSON & SSE streaming chunks)
  - `POST /v1/completions` (raw text prompt completion)
  - `POST /v1/embeddings` (L2-normalized dense vector embeddings)
  - `GET /v1/models` (loaded model registry)
- **Ollama API Compatibility**:
  - `GET /api/tags`
  - `POST /api/chat`
  - `POST /api/generate`
- **Native Apple Silicon MLX Execution**:
  - Zero Python runtime dependencies
  - Multi-head and rotary attention with dynamic `KVCache`
  - Zero-copy tensor transfers across CPU and Apple GPU cores
- **Telemetry & Health**:
  - `GET /health` & `GET /metrics` reporting active requests, prompt tokens, and throughput
- **Interactive Documentation**:
  - Test streaming SSE and compare endpoints live at [code.brandonhubbard.com/mlx-serve-rs](http://code.brandonhubbard.com/mlx-serve-rs/).

---

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mlx-serve-rs = "0.0.1"
```

Or install the standalone binary:

```bash
cargo install mlx-serve-rs
```

---

## 🛠️ CLI Usage

```bash
# Launch server on default port 8080
mlx-serve --port 8080 --model mlx-community/Llama-3.2-3B-Instruct-4bit

# Custom host and port
mlx-serve --host 127.0.0.1 --port 9000
```

---

## 💻 Python Client Integration

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "What makes Apple Silicon fast?"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 🧪 Testing

Run all unit and integration tests:

```bash
cargo test
```

---

## 📄 License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))
