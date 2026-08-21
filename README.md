# 0MGE — Neural Granular Engine

A new kind of audio neural network. **It needs your music to work.**

0MGE doesn't generate sound from nothing. It scans your personal music library, breaks every track into micro-grains (tiny fragments of sound), builds a grain pool, then trains a neural navigator to walk through that pool and generate new sound worlds.

**No library = no generation.** The grain pool IS the instrument. Your music IS the training data.

---

## What Actually Happens

```
your music → grain pool → train navigator → generate
```

1. **Scan** — recursively finds all audio in your library (MP3, FLAC, WAV, OGG, AAC)
2. **Extract** — STFT → 22-dim spectral features, three-tier hierarchy (micro/meso/macro)
3. **Build pool** — millions of grains indexed and clustered (MiniBatchKMeans, 1024 clusters)
4. **Train** — MultiNavigator learns to walk your grain field (6 parallel streams)
5. **Generate** — the navigator picks grains from YOUR pool and assembles new sound

Each grain comes from your tracks. The neural network doesn't invent timbres — it recombines fragments of what you gave it into something new. Better source library = richer output.

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

## What You Need

| Component | Required? | Notes |
|-----------|-----------|-------|
| Your music library | **Yes** | MP3, FLAC, WAV, OGG, AAC — the more the better |
| Python 3.9+ | Yes | With numpy, torch, librosa, scikit-learn |
| Grain pool | Built automatically | From YOUR library during scan |
| Navigator model | Trained automatically | After pool is built |
| ~2GB RAM | Yes | For extraction, more for large libraries |
| GPU optional | No | CPU works, GPU is faster |

---

## Pre-built Demo (What It Sounds Like)

These were generated from Slut Online's music (2389 tracks). Your results will sound different — that's the point.

| Sample | Duration | Description |
|--------|----------|-------------|
| [Landscape #1](samples/drone-01.mp3) | 16s | INT8 quantized model |
| [Landscape #2](samples/drone-02.mp3) | 32s | 6-stream multi-generation |
| [Full Pool Demo](samples/drone-03.mp3) | 60s | 566K grains from Slut Online's library |
| [Quantized vs Original](samples/drone-04-int8.mp3) | 16s | 3.9x compression, same quality |

---

## Downloads

### Models (HuggingFace)

Pre-trained on Slut Online's music. **These will NOT work with your grain pool** — you need to train your own.

| File | Size | Description |
|------|------|-------------|
| [granular_multi_v1_int8.npz](https://huggingface.co/0penAGI/0MGE) | 1.4 MB | INT8 quantized 6-stream navigator (Slut Online's pool) |
| [granular_multi_v1.pt](https://huggingface.co/0penAGI/0MGE) | 5.3 MB | FP32 6-stream navigator (Slut Online's pool) |
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

## Components

### Python Engine (`granular_field.py`)

Core engine (~1430 lines). Three-tier grain hierarchy:

| Level | Frame Size | Duration | Max per File |
|-------|-----------|----------|-------------|
| **Micro** (μ) | 5 STFT frames | ~55ms | 300 |
| **Meso** (σ) | 26 STFT frames | ~300ms | 100 |
| **Macro** (Ω) | 259 STFT frames | ~3s | 30 |

**6-Stream Architecture:**

| Stream | Band | Purpose |
|--------|------|---------|
| sub | 20–120 Hz | Deep bass |
| drums | 120–500 Hz | Punch, body |
| harmonic | 500–2 Hz | Fundamental harmonics |
| texture | 2–4 kHz | Presence |
| noise | 4–8 kHz | Brightness |
| air | 8–11 kHz | Top shimmer |

### Desktop App (`app.py`)

PySide6 (Qt) desktop UI — one button, one result.

### Genome Scanner (`genome_scan.py`)

Beat-aware audio collage generator with tempo normalization, harmonic matching, and click-free transitions.

### VST3/AU Plugin (`vst/`)

Real-time granular processing plugin built with JUCE 8.0.6.

- 32 persistent voices, COLA Hanning envelope
- 9 knobs: Density, Pitch, Stretch, Reverse, Size, Scatter, Mix, Freeze, Focus
- Spectral visualization, glassmorphism particles, audio-reactive UI
- Freeze reverb, spatial spread

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
# Scan library and build grain pool
python3 granular_field.py --scan /path/to/music

# Train navigator on your pool
python3 granular_field.py --train-multi

# Generate 60 bars, multi-stream
python3 granular_field.py --bars 60 --multi-stream --seed 42

# With closed-loop critic
python3 granular_field.py --bars 60 --multi-stream --closed-loop 8
```

---

## Links

- [GitHub](https://github.com/0penAGI/0MGE)
- [HuggingFace](https://huggingface.co/0penAGI/0MGE)

---

## Credits

by **0penAGI**

Neural engine trained on music by **Slut Online** with permission.
