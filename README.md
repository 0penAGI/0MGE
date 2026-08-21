# 0MGE — Neural Granular Engine

0MGE is not text-to-music. **It learns from your music and generates new sound from it.**

It scans your library, cuts every track into micro-grains (55ms–3s fragments), builds a pool of millions of grains, then trains a neural navigator to walk that pool and assemble **new drone landscapes, textures, and atmospheres** from the grains.

**Without your music, there is nothing to generate.** The grain pool IS the instrument. Your tracks ARE the training data.

---

## How It Works (Step by Step)

```
your music → cut into grains → pool → navigator → new sound
```

### Step 1: Scanning
0MGE walks your folders (Music, Ableton, Downloads — wherever you point it) and finds all audio files: MP3, FLAC, WAV, OGG, AAC. Deduplicates by MD5 — the same track is not counted twice.

### Step 2: Grain Extraction
Every track is cut at three levels:
- **Micro (μ)** — 5 STFT frames, ~55ms. Short textural elements.
- **Meso (σ)** — 26 frames, ~300ms. Melodic/rhythmic patterns.
- **Macro (Ω)** — 259 frames, ~3s. Long harmonic structures.

Each grain gets 22 spectral features extracted (centroid, energy, spectral flux, etc.).

### Step 3: Clustering
All grains are clustered via MiniBatchKMeans into 1024 clusters. The navigator picks from clusters, not from millions of raw grains.

### Step 4: Navigator Training
**MultiNavigator** — a Transformer with 4 attention heads, 3 layers, 192 hidden neurons. It learns to predict: "which grain to pick next, given the previous 12 steps?"

6 parallel streams (sub, drums, harmonic, texture, noise, air) each handle their own frequency band and pick grains independently.

### Step 5: Generation
The navigator walks the pool, picks a grain for each of the 6 streams at every step, and assembles them into stereo output. Spectral critics monitor quality — if the generation drifts, the critic switches to better grains.

---

## Quick Start

```bash
# macOS / Linux
./bootstrap.sh

# Windows
setup.bat
```

Creates venv, installs deps, opens the desktop app.

---

## Two Modes

### 1. Python Engine (training + generation)

Full pipeline: scan → pool → train → generate.

```bash
# Scan library and build grain pool
python3 granular_field.py --scan /path/to/music

# Train navigator
python3 granular_field.py --train-multi

# Generate 60 bars
python3 granular_field.py --bars 60 --multi-stream --seed 42

# Or use pre-trained pool + model (no training needed)
python3 granular_field.py --pool granular_pool_v2_int16.npz --model granular_multi_v1.pt --bars 60 --multi-stream
```

### 2. VST3/AU Plugin (real-time)

JUCE plugin for DAWs. **Does not generate** — this is a **granular processor** for real-time use. Takes incoming audio and chops it into grains in real time.

- 32 voices, COLA Hanning envelope
- 9 knobs: Density, Pitch, Stretch, Reverse, Size, Scatter, Mix, Freeze, Focus
- Spectral visualization, freeze reverb, spatial spread

---

## Demo (trained on Slut Online, 2389 tracks)

Your results will sound different — that's the point. Every library gives a unique grain pool.

| Sample | Duration | Description |
|--------|----------|-------------|
| [Landscape #1](samples/drone-01.mp3) | 16s | INT8 quantized model |
| [Landscape #2](samples/drone-02.mp3) | 32s | 6-stream multi-generation |
| [Full Pool Demo](samples/drone-03.mp3) | 60s | 566K grains from Slut Online's library |
| [Quantized vs Original](samples/drone-04-int8.mp3) | 16s | 3.9× compression, same quality |

---

## Downloads

### Models (HuggingFace)

**Pre-trained models are on Slut Online's music.** For your own music — train yourself (see CLI above).

| File | Size | Description |
|------|------|-------------|
| [granular_multi_v1_int8.npz](https://huggingface.co/0penAGI/0MGE) | 1.4 MB | **Recommended** — INT8 quantized 6-stream navigator |
| [granular_multi_v1.pt](https://huggingface.co/0penAGI/0MGE) | 5.3 MB | FP32 6-stream navigator |
| [granular_pool_v2_int16.npz](https://huggingface.co/0penAGI/0MGE) | 4.9 GB | Full grain pool with raw audio (INT16) |
| [granular_pool_lite.npz](https://huggingface.co/0penAGI/0MGE) | 64 MB | Lite pool (features only, no raw audio) |

### VST3 / AU Plugin

| Platform | Format | Install |
|----------|--------|---------|
| macOS | VST3 + AU (.pkg) | Double-click installer |
| Windows | VST3 (.exe) | Run installer |

### Standalone App

| Platform | Format | Install |
|----------|--------|---------|
| macOS | .app (PyInstaller) | Drag to Applications |
| Windows | .exe (PyInstaller) | Run installer |

---

## Architecture

### 6-Stream System

| Stream | Frequency | Role |
|--------|-----------|------|
| sub | 20–120 Hz | Deep bass |
| drums | 120–500 Hz | Percussion, body |
| harmonic | 500–2000 Hz | Tonal harmonics |
| texture | 2–4 kHz | Presence |
| noise | 4–8 kHz | High-frequency detail |
| air | 8–11 kHz | Upper spectrum |

### Grain Hierarchy

| Level | Frame Size | Duration | Max per File |
|-------|-----------|----------|-------------|
| **Micro** (μ) | 5 STFT frames | ~55ms | 300 |
| **Meso** (σ) | 26 frames | ~300ms | 100 |
| **Macro** (Ω) | 259 frames | ~3s | 30 |

### Quantization

| Metric | FP32 | INT8 | Delta |
|--------|------|------|-------|
| Critic score (spectral) | 0.292 | 0.285 | 0.008 |
| File size | 5.3 MB | 1.4 MB | 3.9× compression |

---

## What You Need

| Component | Required? | Notes |
|-----------|-----------|-------|
| Your music library | **Yes** | MP3, FLAC, WAV, OGG, AAC — the more the better |
| Python 3.9+ | Yes | numpy, torch, librosa, scikit-learn |
| Grain pool | Built automatically | From your library during scan |
| Navigator | Trained automatically | After pool is built |
| ~2GB RAM | Yes | For extraction, more for large libraries |
| GPU | Optional | CPU works, GPU is faster |

---

## Build from Source

### Plugin

**macOS:**
```bash
cd vst && mkdir build && cd build
cmake -G Xcode .. && cmake --build . --config Release
```

**Windows:**
```bash
cd vst && mkdir build && cd build
cmake -G "Visual Studio 17 2022" -A x64 .. && cmake --build . --config Release
```

### Standalone App
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name 0MGE \
    --add-data "granular_multi_v1_int8.npz:." \
    --add-data "granular_multi_v1_int8_meta.json:." \
    --add-data "granular_pool_lite.npz:." \
    app.py
```

---

## CLI

```bash
# Scan library
python3 granular_field.py --scan /path/to/music

# Train multi-stream navigator
python3 granular_field.py --train-multi

# Generate 60 bars
python3 granular_field.py --bars 60 --multi-stream --seed 42

# With closed-loop critic
python3 granular_field.py --bars 60 --multi-stream --closed-loop 8

# Use pre-trained pool + model (no training)
python3 granular_field.py --pool granular_pool_v2_int16.npz --model granular_multi_v1.pt --bars 60 --multi-stream
```

---

## Links

- [GitHub](https://github.com/0penAGI/0MGE)
- [HuggingFace](https://huggingface.co/0penAGI/0MGE)

---

## Credits

by **0penAGI**

Neural engine trained on music by **Slut Online** with permission.
