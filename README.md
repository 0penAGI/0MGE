# 0MGE — 0penAGI Music Granular Engine

Real-time granular music synthesis system. Extract hierarchical grains from audio, train neural navigators, generate new music, and process live audio through a VST3/AU plugin.

---

## Quick Start

```bash
# macOS / Linux
./bootstrap.sh

# Windows
setup.bat
```

Creates venv, installs deps, opens the desktop app. That's it.

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
2. **Extract** — STFT → 22-dim spectral features per grain (centroid, bandwidth, flatness, rolloff, 6 band energies, temporal stats, flux)
3. **Cluster** — MiniBatchKMeans, 1024 clusters across all grains
4. **Train** — Navigator learns to walk the grain field (predict next grain from context)
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

**Neural Navigator** (`Navigator`, `MultiNavigator`):
- Transformer with 4 heads, 3 layers, 192 hidden dim
- 48-dim state, 12-length context window
- 6 independent stream heads for multi-layer synthesis
- Trained on grain feature sequences (8000 steps, batch 128)

**Spectral Critic:**
- `MultiScaleCritic` — energy/temporal flux/rhythm scoring
- `STFTCritic` — spectral envelope matching, band energy ratios
- `SpectralField` — attraction/repulsion between streams

**Audio Processing Chain:**
- Band-pass filtering (butter, per-stream)
- OTT compression
- Stereo enhancement/spread
- Saturation, reverb, limiter
- Final master (limiter + stereo spread)

### 2. Desktop App (`app.py`)

PySide6 (Qt) desktop UI. Minimal design — one button, one result.

**Features:**
- Audio file scanner (Music, Documents, Downloads, Desktop, Ableton)
- Waveform display with playback
- Stream parameter sliders (per-band weights)
- Settings: bars, BPM, seed, temperature, multi-stream toggle
- In-app audio player (QMediaPlayer)
- Generated output history

### 3. Genome Scanner (`genome_scan.py`)

Advanced track generator (~709 lines). Beat-aware audio collage.

**Features:**
- Beat-aligned extraction (not random cuts)
- Tempo normalization (phase vocoder stretch)
- Harmonic/key matching (Krumhansl-Schmuckler profiles)
- Structure-aware arc (intro → build → drop → break → outro)
- Click-free transitions (zero-crossing + equal-power crossfade)

### 4. VST3/AU Plugin (`vst/`)

Real-time granular processing plugin built with JUCE 8.0.6.

**Engine (32 persistent voices):**
- COLA Hanning envelope (click-free overlap-add)
- normFactor = 0.25 / max(1, density) — prevents constructive interference
- Circular buffer input capture
- Freeze reverb: LP filter + 12% L/R crossfeed in feedback path
- Spatial spread: pan ±0.5 with focus-based narrowing
- 8-band spectral analysis (256-point FFT every 512 samples)

**UI (9 knobs, 3 rows):**

| Row 1 | Row 2 | Row 3 |
|-------|-------|-------|
| Density / Size / Scatter / Pitch | Stretch / Reverse / Focus | Mix / Freeze |

**Visualization:**
- Real input waveform from circular buffer
- 80 glassmorphism particles (freq-band tied, trail enabled)
- 7-band spectral bars with animated centroid
- Audio-reactive background pulse
- Crystalline freeze overlay

---

## Installation

### Desktop App
```bash
# macOS / Linux
./bootstrap.sh

# Windows
setup.bat
```

### VST3/AU Plugin

Pre-built binaries in `release/`:

```
release/
  macos/
    0MGE.component    → AU (Logic, GarageBand)
    0MGE.vst3         → VST3 (Ableton, FL Studio, Reaper, etc.)
  windows/
    README.txt        → Build instructions for Windows
```

**Install (macOS):**
```bash
cp -R release/macos/0MGE.component ~/Library/Audio/Plug-Ins/Components/
cp -R release/macos/0MGE.vst3 ~/Library/Audio/Plug-Ins/VST3/
```

**Install (Windows):**
Copy `0MGE.vst3` to `C:\Program Files\Common Files\VST3\`

### Build Plugin from Source

**macOS:**
```bash
cd vst
mkdir build && cd build
cmake -G Xcode ..
cmake --build . --config Release
```

**Windows:**
```bash
cd vst
mkdir build && cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

Requires: CMake 3.22+, internet (fetches JUCE 8.0.6 automatically).

---

## CLI

```bash
# Generate 60 bars, multi-stream (6 layers)
python3 granular_field.py --bars 60 --multi-stream --seed 42

# Generate with closed-loop critic (8 iterations)
python3 granular_field.py --bars 60 --multi-stream --closed-loop 8

# Single-stream (faster)
python3 granular_field.py --bars 60 --seed 42

# Rebuild pool with full audio
python3 granular_field.py --bars 15 --full-pool

# Train multi-stream navigator
python3 granular_field.py --train-multi

# Genome scanner
python3 genome_scan.py
```

---

## Dependencies

```
numpy
torch
librosa
soundfile
scikit-learn
PySide6
```

Platform-specific:
- macOS: Apple Silicon optimized (MPS acceleration)
- GPU: CUDA-compatible PyTorch (optional)

---

## File Structure

| File | Description |
|------|-------------|
| `granular_field.py` | Core engine — extraction, training, generation (~1430 lines) |
| `app.py` | PySide6 desktop UI (~630 lines) |
| `genome_scan.py` | Beat-aware track generator (~709 lines) |
| `granular_pool_lite.npz` | Lite pool: 425K micro + 118K meso + 23K macro grains (features only) |
| `granular_navigator_v2.pt` | Trained single-stream navigator |
| `granular_output/` | Generated audio tracks |
| `genome_filelist.json` | Cached audio file list |
| `scan_index.json` | Desktop app scan cache |
| `app_settings.json` | Desktop app settings |
| `vst/` | JUCE plugin source |
| `release/` | Pre-built plugin binaries |

---

## Architecture

```
Audio Library (2389+ tracks)
    ↓ scan + dedup (MD5)
STFT + Feature Extraction (22-dim)
    ↓ 3-tier hierarchy
Micro / Meso / Macro Grains
    ↓ MiniBatchKMeans (1024 clusters)
Grain Pool (566K grains)
    ↓ train (Transformer, 8000 steps)
Navigator (6-stream heads)
    ↓ generate
Spectral Field (attraction/repulsion)
    ↓ multi-scale critics
Audio Output (WAV)
```

---

## Credits

by **0penAGI** x TeMeT x Slut Online
