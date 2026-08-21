---
title: "0MGE: Neural Granular Engine"
library_name: 0mge
license: mit
tags:
  - audio
  - granular-synthesis
  - neural-network
  - sound-design
  - quantization
  - pytorch
---

# 0MGE: Neural Granular Engine

Pre-trained granular navigator and grain pool trained on 2389 tracks (~48 hours of music). Generates new drone landscapes, textures, and atmospheres by walking a learned grain field.

**For source code, training pipeline, and the desktop app:** [0MGE on GitHub](https://github.com/0penAGI/0MGE)

---

## Generate

```bash
git clone https://github.com/0penAGI/0MGE.git && cd 0MGE
pip install numpy torch librosa scikit-learn soundfile

python3 granular_field.py --pool granular_pool_v2_int16.npz --model granular_multi_v1.pt --bars 60 --multi-stream --seed 42
```

Output: `granular_output/granular_60bars_*.wav` (stereo, 22050 Hz).

---

## Files

| File | Size | Description |
|------|------|-------------|
| `granular_multi_v1.pt` | 5.3 MB | FP32 6-stream navigator |
| `granular_multi_v1_int8.npz` | 1.4 MB | INT8 quantized (weight-only, per-tensor) |
| `granular_multi_v1_int8_meta.json` | 0.4 KB | INT8 scale metadata |
| `granular_multi_v1_fp16.pt` | 2.7 MB | FP16 half-precision |
| `granular_pool_v2_int16.npz` | 4.9 GB | Full grain pool, 566K grains, INT16 with per-row peak normalization |
| `granular_pool_lite.npz` | 64 MB | Features only (22-dim), no raw audio |

---

## Architecture

**MultiNavigator** — Transformer (4 heads, 3 layers, 192 hidden). 48-dim state, 12-step context. 6 independent stream heads select grains via softmax over pool similarities.

| Stream | Band | Role |
|--------|------|------|
| sub | 20–120 Hz | Low-frequency energy |
| drums | 120–500 Hz | Transient body |
| harmonic | 500–2000 Hz | Tonal content |
| texture | 2–4 kHz | Mid-high presence |
| noise | 4–8 kHz | High-frequency detail |
| air | 8–11 kHz | Upper spectrum |

---

## Grain Pool

Three-tier hierarchy extracted via STFT (n_fft=1024, hop=256):

| Level | Duration | Count |
|-------|----------|-------|
| Micro (μ) | ~55ms | 425K |
| Meso (σ) | ~300ms | 118K |
| Macro (Ω) | ~3s | 23K |
| **Total** | — | **566K** |

22-dimensional spectral features per grain. MiniBatchKMeans clustering (1024 clusters).

---

## Quantization

| Metric | FP32 | INT8 | Delta |
|--------|------|------|-------|
| Critic score | 0.292 | 0.285 | 0.008 |
| File size | 5.3 MB | 1.4 MB | 3.9× |

---

## Citation

```bibtex
@software{0mge2026,
  title={0MGE: Neural Granular Engine},
  author={0penAGI},
  year={2026},
  url={https://github.com/0penAGI/0MGE}
}
```

## License

MIT

Trained on music by **Slut Online** with permission.
