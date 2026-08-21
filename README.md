# 0MGE — Neural Granular Engine

A neural network that learns from YOUR music and generates new sound.

Launch the app. Point it at your music folder. It scans, trains, and generates — all locally on your machine. No cloud, no API, no subscription.

---

## Quick Start

```bash
# macOS / Linux
./bootstrap.sh

# Windows
setup.bat
```

Opens the desktop app. Select your music folder, hit Generate.

That's it. It scans your library, builds a grain pool, trains a navigator, and generates new drone landscapes from YOUR music.

---

## How It Works

1. **Scan** — finds all audio in your folders (MP3, FLAC, WAV, OGG, AAC)
2. **Extract** — cuts every track into micro-grains (55ms–3s fragments), 22 spectral features per grain
3. **Cluster** — groups grains into 1024 clusters
4. **Train** — neural navigator learns to walk the grain field (6 parallel streams)
5. **Generate** — assembles new sound from your grains

Your music IS the instrument. Different library = different output. That's the point.

---

## Try It First (Demo)

Don't have your music ready? We trained a model on 2389 tracks and put it on HuggingFace so you can hear what it sounds like.

```bash
# Clone the repo
git clone https://github.com/0penAGI/0MGE.git
cd 0MGE
pip install numpy torch librosa scikit-learn soundfile

# Download granular_pool_v2_int16.npz (4.9 GB) and granular_multi_v1.pt (5.3 MB)
# from https://huggingface.co/0penAGI/0MGE

# Generate (model already trained, no --train-multi needed)
python3 granular_field.py --pool granular_pool_v2_int16.npz --model granular_multi_v1.pt --bars 60 --multi-stream
```

This generates from Slut Online's music. To generate from your own — just use the app.

---

## Components

### Desktop App (`app.py`)

![Desktop App](app.png)

PySide6 (Qt) desktop interface. The main way to use 0MGE — one window, one button, zero friction.

**What it does:**
- Scans your music folders (Music, Downloads, Desktop, Ableton, external drives)
- Extracts micro-grains from every track (55ms–3s fragments)
- Builds a grain pool locally (~64MB lite cache)
- Trains a MultiNavigator on your grains
- Generates new stereo WAV files from your trained pool
- Plays back results with built-in player

**UI:**
- Folder selector (auto-detects Music/Downloads/Ableton)
- Bars, BPM, Temperature, Seed controls
- Multi-stream toggle (6 frequency bands)
- Train MultiNavigator checkbox
- Real-time progress with spectral output preview
- Generated files open in Finder/Explorer

**Settings persist** in `app_settings.json` — remembers your folder, bars, BPM between sessions.

```bash
pip install PySide6 soundfile numpy
python3 app.py
```

### Neural Engine (`granular_field.py`)

CLI for the full pipeline: scan → extract → cluster → train → generate.

```bash
# Generate from your music (app does this automatically)
python3 granular_field.py --bars 60 --multi-stream

# Or use pre-trained demo pool
python3 granular_field.py --pool granular_pool_v2_int16.npz --model granular_multi_v1.pt --bars 60 --multi-stream
```

### VST3/AU Plugin (`vst/`)

![VST Plugin](vst.png)

**Completely separate from the neural engine.** A real-time granular processor built with JUCE 8.0.6. No AI — pure DSP.

Takes incoming audio from your DAW and chops it into grains in real time. Each grain is a tiny snapshot of the input signal, played back with its own pitch, position, direction, and spatial placement. The result: input audio is shredded and reassembled into evolving textures.

**Grain engine:**
- 32 persistent voices, COLA Hanning envelope (click-free)
- Circular buffer (10s at 48kHz) stores incoming audio
- Voices spawn at intervals determined by Density — each voice reads a random slice from the buffer
- Grain size: 30–330ms (Size knob), with ±15% jitter per voice
- Pitch: ±24 semitones, applied as playback rate
- Stretch: slows down or speeds up grain read position (0.25×–4×)
- Reverse: probability-based per grain (0–100%)
- Scatter: random pitch variation per grain (±8% at max)
- Spatial: pan ±0.5, Focus controls stereo spread (0.3–1.0×)

**Freeze reverb:**
- LP filter in feedback path (cutoff 2–8% of SR)
- 12% L/R crossfeed in feedback loop
- Creates evolving drones from any input signal
- When freeze > 50%, feedback buffer holds the last sound indefinitely

**9 knobs:**

| Knob | Range | What it does |
|------|-------|-------------|
| Density | 0.1–6.0 | How many voices spawn per hop (1–4 voices) |
| Pitch | ±24 st | Playback rate of grains |
| Stretch | 0.25–4× | Time-stretch factor |
| Reverse | 0–100% | Probability of grain playing backwards |
| Size | 0–100% | Grain duration (30–330ms) |
| Scatter | 0–100% | Random pitch variation per grain |
| Mix | 0–100% | Dry/wet crossfade |
| Freeze | 0–100% | Feedback amount (reverb/drone) |
| Focus | 0–100% | Stereo spread (narrow ↔ wide) |

**UI:** spectral visualization (8 bands), centroid display, real-time waveform, glassmorphism particles. Light theme.

```bash
cd vst && mkdir build && cd build
cmake -G Xcode .. && cmake --build . --config Release
```

### Genome Scanner (`genome_scan.py`)

Beat-aware audio collage generator. Cuts tracks into beat-aligned fragments, tempo-normalizes, pitch-matches, arranges into new tracks.

---

## Downloads

### Models (HuggingFace)

Pre-trained on Slut Online's music (2389 tracks). For demo purposes.

| File | Size | Description |
|------|------|-------------|
| [granular_multi_v1.pt](https://huggingface.co/0penAGI/0MGE) | 5.3 MB | 6-stream navigator |
| [granular_multi_v1_int8.npz](https://huggingface.co/0penAGI/0MGE) | 1.4 MB | INT8 quantized (3.9× smaller) |
| [granular_pool_v2_int16.npz](https://huggingface.co/0penAGI/0MGE) | 4.9 GB | Full grain pool (566K grains) |
| [granular_pool_lite.npz](https://huggingface.co/0penAGI/0MGE) | 64 MB | Lite pool (features only) |

Your own trained files are much smaller — the app builds what you need locally.

### VST3 / AU Plugin

| Platform | Format | Install |
|----------|--------|---------|
| macOS | VST3 + AU (.pkg) | Double-click installer |
| Windows | VST3 (.exe) | Run installer |

---

## Demo

| Sample | Duration | Description |
|--------|----------|-------------|
| [Landscape #1](samples/drone-01.mp3) | 16s | INT8 quantized |
| [Landscape #2](samples/drone-02.mp3) | 32s | 6-stream generation |
| [Full Pool Demo](samples/drone-03.mp3) | 60s | 566K grains |
| [Quantized vs Original](samples/drone-04-int8.mp3) | 16s | Compression comparison |

---

## Architecture

**MultiNavigator** — Transformer (4 attention heads, 3 layers, 192 hidden). 6 independent stream heads pick grains from the pool.

| Stream | Frequency | Role |
|--------|-----------|------|
| sub | 20–120 Hz | Deep bass |
| drums | 120–500 Hz | Percussion |
| harmonic | 500–2000 Hz | Tonal harmonics |
| texture | 2–4 kHz | Presence |
| noise | 4–8 kHz | Detail |
| air | 8–11 kHz | Upper spectrum |

---

## Links

- [Listen to Demo](https://0penagi.github.io/0MGE/) — audio-reactive player with GLSL shader
- [GitHub](https://github.com/0penAGI/0MGE)
- [HuggingFace](https://huggingface.co/0penAGI/0MGE)

---

## Credits

by **0penAGI**

Neural engine trained on music by **Slut Online** with permission.
