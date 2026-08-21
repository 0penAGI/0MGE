# 0MGE — 0penAGI Music Granular Engine

**Drone sound landscape generator** — not a classic music generator.

0MGE transforms your music into evolving soundscapes, textures, and atmospheres. Feed it any audio and it breaks it into micro-grains, then rebuilds new sonic worlds using a neural navigator. Perfect for **sound design, horror games, film scores, and experimental audio**.

The neural engine was trained on music by **Slut Online** (with permission).

---

## Demo

| Sample | Duration | Description |
|--------|----------|-------------|
| [INT8 Test](samples/granular_int8_test.wav) | 32s | Drone landscape from quantized model |
| [Big Pool Demo](samples/granular_bigpool_demo.wav) | 60s | Multi-stream generation from full grain pool |

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

## Downloads

### VST3 / AU Plugin

| Platform | Format | Install |
|----------|--------|---------|
| macOS | VST3 + AU (.pkg) | Double-click installer |
| Windows | VST3 (.exe) | Run installer |

### Standalone App (Python + Neural Engine)

| Platform | Format | Install |
|----------|--------|---------|
| macOS | .app (PyInstaller) | Drag to Applications |
| Windows | .exe (PyInstaller) | Run installer |

### Models (HuggingFace)

| File | Size | Description |
|------|------|-------------|
| [granular_multi_v1_int8.npz](https://huggingface.co/0penAGI/0MGE) | 1.4 MB | **Recommended** — INT8 quantized 6-stream navigator |
| [granular_multi_v1.pt](https://huggingface.co/0penAGI/0MGE) | 5.3 MB | FP32 6-stream navigator |
| [granular_pool_v2_int16.npz](https://huggingface.co/0penAGI/0MGE) | 4.9 GB | Full grain pool with raw audio (INT16) |
| [granular_pool_lite.npz](https://huggingface.co/0penAGI/0MGE) | 64 MB | Lite pool (features only, no raw audio) |

---

## Components

### 1. Python Engine (`granular_field.py`)

Core engine (~1430 lines). Three-tier grain hierarchy:

| Level | Frame Size | Duration | Max per File |
|-------|-----------|----------|-------------|
| **Micro** (μ) | 5 STFT frames | ~55ms | 300 |
| **Meso** (σ) | 26 STFT frames | ~300ms | 100 |
| **Macro** (Ω) | 259 STFT frames | ~3s | 30 |

**Pipeline:**
1. **Scan** — recursive walk of Music/Ableton/etc folders, dedup by MD5
2. **Extract** — STFT → 22-dim spectral features per grain
3. **Cluster** — MiniBatchKMeans, 1024 clusters across all grains
4. **Train** — MultiNavigator learns to walk the grain field (6 streams)
5. **Generate** — multi-stream synthesis with spectral critics

**6-Stream Architecture:**

| Stream | Band | Purpose |
|--------|------|---------|
| sub | 20–120 Hz | Deep bass |
| drums | 120–500 Hz | Punch, body |
| harmonic | 500–2 Hz | Fundamental harmonics |
| texture | 2–4 kHz | Presence |
| noise | 4–8 kHz | Brightness |
| air | 8–11 kHz | Top shimmer |

### 2. Desktop App (`app.py`)

PySide6 (Qt) desktop UI — one button, one result.

### 3. Genome Scanner (`genome_scan.py`)

Beat-aware audio collage generator with tempo normalization, harmonic matching, and click-free transitions.

### 4. VST3/AU Plugin (`vst/`)

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

## HuggingFace

Models, quantized weights, and audio samples: [0penAGI/0MGE on HuggingFace](https://huggingface.co/0penAGI/0MGE)

---

## Credits

by **0penAGI** x TeMeT x Slut Online

Neural engine trained on music by **Slut Online** with permission.
