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

# Generate
python3 granular_field.py --pool granular_pool_v2_int16.npz --model granular_multi_v1.pt --bars 60 --multi-stream
```

This generates from Slut Online's music. To generate from your own — just use the app.

---

## Components

### Desktop App (`app.py`)

PySide6 GUI. Select folder → Generate. Wraps the full pipeline.

### Neural Engine (`granular_field.py`)

CLI for the full pipeline: scan → extract → cluster → train → generate.

```bash
# Generate from your music (app does this automatically)
python3 granular_field.py --bars 60 --multi-stream

# Or use pre-trained demo pool
python3 granular_field.py --pool granular_pool_v2_int16.npz --model granular_multi_v1.pt --bars 60 --multi-stream
```

### VST3/AU Plugin (`vst/`)

Real-time granular processor for DAWs. Separate from the neural engine — takes incoming audio and chops it into grains live. 32 voices, 9 knobs.

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

- [GitHub](https://github.com/0penAGI/0MGE)
- [HuggingFace](https://huggingface.co/0penAGI/0MGE)

---

## Credits

by **0penAGI**

Neural engine trained on music by **Slut Online** with permission.
