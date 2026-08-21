# 0MGE — Neural Granular Engine

A new kind of audio neural network. **Trains on your music library, on your computer.**

No cloud. No API. No subscription. Just your machine and your music.

0MGE scans your audio library, extracts micro-grains (tiny fragments of sound), and learns to rebuild new sonic worlds from them. The result is not a remix — it's a new species of sound: evolving textures, alien landscapes, impossible instruments that never existed.

---

## How It Works

```
your music → extract grains → train navigator → generate new sound
```

1. **Scan** — recursively finds all audio in your library (MP3, FLAC, WAV, OGG, AAC)
2. **Extract** — STFT → 22-dim spectral features, three-tier hierarchy (micro/meso/macro)
3. **Cluster** — MiniBatchKMeans, 1024 clusters across all grains
4. **Train** — MultiNavigator learns to walk the grain field (6 parallel streams)
5. **Generate** — multi-stream synthesis with spectral critics

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

## Demo

| Sample | Duration | Description |
|--------|----------|-------------|
| [Landscape #1](samples/drone-01.mp3) | 16s | INT8 quantized model, trained on 2389 tracks |
| [Landscape #2](samples/drone-02.mp3) | 32s | 6-stream multi-generation |
| [Full Pool Demo](samples/drone-03.mp3) | 60s | 566K grains, full grain pool |
| [Quantized vs Original](samples/drone-04-int8.mp3) | 16s | 3.9x compression, same quality |

---

## Downloads

### Models (HuggingFace)

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
# Generate 60 bars, multi-stream
python3 granular_field.py --bars 60 --multi-stream --seed 42

# With closed-loop critic
python3 granular_field.py --bars 60 --multi-stream --closed-loop 8

# Train multi-stream navigator
python3 granular_field.py --train-multi

# Rebuild pool with full audio
python3 granular_field.py --full-pool --train-multi
```

---

## Links

- [GitHub](https://github.com/0penAGI/0MGE)
- [HuggingFace](https://huggingface.co/0penAGI/0MGE)

---

## Credits

by **0penAGI**

Neural engine trained on music by **Slut Online** with permission.
