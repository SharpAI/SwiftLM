### `Thump604/DeepSeek-V4-Flash-MLX-Q3-mixed-gs128-affine` — Context & Memory Profile

Context depths tested: 512,40000

| Configuration | Context Size | TTFT | Generation Speed | Model Size | Active RAM (OS) | GPU_Alloc (virtual) | GPU_InUse peak (physical) |
|---|---|---|---|---|---|---|---|
| SSD Stream | 512 | 6.80s | 4.65 tok/s | N/A | 17.0 GB | 28.4 GB | 16.7 GB |
| SSD Stream | 40000 | 565.02s | 0.32 tok/s | N/A | 48.3 GB | 60.5 GB | 12.5 GB |
| SSD + TurboQuant | 512 | 6.35s | 4.78 tok/s | N/A | 16.9 GB | 29.5 GB | 16.8 GB |
| SSD + TurboQuant | 40000 | 363.76s | 4.16 tok/s | N/A | 28.3 GB | 40.6 GB | 16.8 GB |
| SSD + 16-Worker Prefetch | 512 | 5.84s | 4.43 tok/s | N/A | 16.9 GB | 29.3 GB | 16.6 GB |
| SSD + 16-Worker Prefetch | 40000 | 565.50s | 0.32 tok/s | N/A | 48.3 GB | 60.9 GB | 13.6 GB |

> **Active RAM (OS)**: Memory wired into physical RAM by macOS (from server log).
> **GPU_Alloc (virtual)**: Total GPU address-space allocation including SSD-backed pages — the TRUE memory demand, can exceed physical RAM.
> **GPU_InUse peak (physical)**: Peak physical RAM occupied by the GPU during the entire request (prefill + generation), sampled every 0.5 s. This is the real active footprint — for SSD-streaming configs it reflects the high-water mark while layers are being read, not a post-generation snapshot.
