---
title: "0MGE: Neural Granular Engine — Quantized Multi-Stream Navigator"
library_name: 0mge
license: mit
tags:
  - audio
  - granular-synthesis
  - neural-network
  - sound-design
  - quantization
  - pytorch
  - music-generation
---

# 0MGE: Quantized Multi-Stream Neural Navigator

Pre-trained granular navigator models trained on a corpus of 2389 audio tracks (~48 hours of music). Models generate new audio by walking a learned grain field extracted from the source corpus.

**For full architecture, training pipeline, and usage instructions, see [0MGE on GitHub](https://github.com/0penAGI/0MGE).**

---

## Models

| File | Format | Size | Params | Description |
|------|--------|------|--------|-------------|
| `granular_multi_v1.pt` | PyTorch | 5.3 MB | ~180K | FP32 6-stream navigator |
| `granular_multi_v1_int8.npz` | NumPy | 1.4 MB | ~180K | INT8 weight-only quantized |
| `granular_multi_v1_int8_meta.json` | JSON | 0.4 KB | — | Per-tensor scale metadata |
| `granular_multi_v1_fp16.pt` | PyTorch | 2.7 MB | ~180K | FP16 half-precision |

## Grain Pool

| File | Format | Size | Grains | Description |
|------|--------|------|--------|-------------|
| `granular_pool_v2_int16.npz` | NumPy | 4.9 GB | 566K | Full pool, INT16 quantized with per-row peak normalization |
| `granular_pool_lite.npz` | NumPy | 64 MB | 566K | Features only (22-dim), no raw audio |

---

## Architecture

**MultiNavigator** — Transformer-based sequential model:

- Embedding: 48-dim input state → 192 hidden
- Context: 12-step history window
- Backbone: 4 attention heads, 3 transformer layers
- Output: 6 independent stream heads (sub, drums, harmonic, texture, noise, air)

Each stream selects a grain from the pool given the current spectral state and context history. Selection is softmax over pool similarities.

**6-Stream Frequency Decomposition:**

| Stream | Band | Role |
|--------|------|------|
| sub | 20–120 Hz | Low-frequency energy |
| drums | 120–500 Hz | Transient body |
| harmonic | 500–2000 Hz | Tonal content |
| texture | 2–4 kHz | Mid-high presence |
| noise | 4–8 kHz | High-frequency detail |
| air | 8–11 kHz | Upper spectral content |

---

## Quantization

### INT8 (Weight-Only, Per-Tensor)

```
scale = max(|W|) / 127
W_int8 = round(W / scale)
```

| Metric | FP32 | INT8 | Delta |
|--------|------|------|-------|
| Critic score (spectral) | 0.292 | 0.285 | 0.008 |
| File size | 5.3 MB | 1.4 MB | 3.9× compression |

Quantization is weight-only. Activations remain FP32 during inference. No fine-tuning or calibration data required.

### INT16 Grain Pool (Per-Row Peak Normalization)

Each row (grain) is normalized to its peak amplitude, stored as INT16 with a JSON scale factor:

```json
{
  "version": 2,
  "dtype": "int16",
  "rows": 566044,
  "cols": 2208,
  "normalize": true,
  "description": "per-row peak normalization"
}
```

---

## Data

Source corpus: 2389 tracks, ~48 hours of music.

**Grain extraction pipeline:**
1. STFT (n_fft=2048, hop=512)
2. Three-tier hierarchy: micro (μ, ~55ms), meso (σ, ~300ms), macro (Ω, ~3s)
3. 22-dimensional spectral features per grain
4. MiniBatchKMeans clustering (1024 clusters)

**Pool statistics:**

| Grain Type | Count |
|------------|-------|
| Micro (μ) | 425K |
| Meso (σ) | 118K |
| Macro (Ω) | 23K |
| **Total** | **566K** |

---

## Audio Samples

| Sample | Duration | Model | Description |
|--------|----------|-------|-------------|
| `samples/drone-01.mp3` | 16s | INT8 quantized | 4-bar landscape |
| `samples/drone-02.mp3` | 32s | FP32 multi | 16-bar landscape, 6 streams |
| `samples/drone-03.mp3` | 60s | FP32 multi | Full pool demo, 566K grains |
| `samples/drone-04-int8.mp3` | 16s | INT8 | Quantization comparison |

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

---

## License

MIT

## Acknowledgements

Neural engine trained on music by **Slut Online** with permission.
