# 0MGE — Pre-built Downloads

Just download and use. No building required.

## macOS

### Standalone App (drag & drop)
1. Download `macOS/0MGE.app`
2. Drag it to `/Applications/`
3. Double-click to open
4. First launch: right-click → Open (macOS Gatekeeper)

**What it does:** Scans your music folder, builds a grain pool, trains a neural network, generates new sound. All locally.

### VST3 / AU Plugin (for DAW)
Copy to your DAW's plugin folder:

```bash
# User folder
cp -R macOS/VST3/0MGE.vst3 ~/Library/Audio/Plug-Ins/VST3/
cp -R macOS/AU/0MGE.component ~/Library/Audio/Plug-Ins/Components/

# Or system-wide (admin password)
sudo cp -R macOS/VST3/0MGE.vst3 /Library/Audio/Plug-Ins/VST3/
sudo cp -R macOS/AU/0MGE.component /Library/Audio/Plug-Ins/Components/
```

Restart your DAW.

## Uninstall
```bash
rm -rf /Applications/0MGE.app
rm -rf ~/Library/Audio/Plug-Ins/VST3/0MGE.vst3
rm -rf ~/Library/Audio/Plug-Ins/Components/0MGE.component
```
