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
  - music-generation
  - generative-audio
  - drone-synthesis
  - texture-generation
  - spectral-analysis
  - local-ai
  - open-source
  - transformer
  - signal-processing
pipeline_tag: other
---

# 0MGE: Neural Granular Engine

> **AI music generation from YOUR music.** Scan, train, generate — all locally. No cloud, no API, no subscription.

Pre-trained neural granular engine trained on 2389 tracks (~48 hours of music). Generates new drone landscapes, textures, and atmospheres from a learned grain field.

[![GitHub](https://img.shields.io/badge/GitHub-0penAGI%2F0MGE-black?logo=github)](https://github.com/0penAGI/0MGE)
[![Demo](https://img.shields.io/badge/Listen-Demo-orange)](https://0penagi.github.io/0MGE/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

![Desktop App](app.png)

![VST Plugin](vst.png)

**Full source code, training pipeline, and desktop app:** [0MGE on GitHub](https://github.com/0penAGI/0MGE)

**[Listen to demo](https://0penagi.github.io/0MGE/)** — audio-reactive player with GLSL shader

---

## What This Is

A neural network that learns from music and generates new sound. Not text-to-music. It scans audio files, cuts them into millions of micro-grains, and trains a navigator to assemble those grains into new sonic worlds.

This repo contains a **pre-trained model and grain pool** trained on Slut Online's music (2389 tracks). Download, generate, hear what it sounds like.

To generate from your own music — use the [desktop app](https://github.com/0penAGI/0MGE) (scans your library, trains locally).

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
| `granular_multi_v1_int8.npz` | 1.4 MB | Navigator INT8 quantized (weights only, not the grain pool) |
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

## Audio Samples

| Sample | Duration | Model |
|--------|----------|-------|
| `samples/drone-01.mp3` | 16s | INT8 quantized |
| `samples/drone-02.mp3` | 32s | FP32 6-stream |
| `samples/drone-03.mp3` | 60s | Full pool demo |
| `samples/drone-04-int8.mp3` | 16s | Quantization comparison |

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
