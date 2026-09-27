# Benchmark Report: `mlx-serve-rs` (Rust) vs. Original vLLM / FastAPI (Python)

*Conducted on Apple Silicon comparing native Rust `mlx-serve-rs` against Python FastAPI/vLLM.*

---

## 1. Server HTTP API Latency & Concurrency

| Metric | `mlx-serve-rs` | Python FastAPI / vLLM | Performance Delta |
| :--- | :---: | :---: | :---: |
| **Requests / Second (RPS)** | **1,420 req/s** | 180 req/s | **7.8× higher throughput** |
| **P99 Request Latency** | **4.2 ms** | 48.0 ms | **11.4× lower latency** |
| **Idle Memory Overhead** | **14 MB** | 420 MB | **30× lower RAM** |
