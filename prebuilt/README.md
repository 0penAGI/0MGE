# 0MGE — Pre-built Plugins

Manual install: copy the plugin to your DAW's plugin folder.

## macOS

### VST3 (works in most DAWs)
```bash
cp -R macOS/VST3/0MGE.vst3 ~/Library/Audio/Plug-Ins/VST3/
```

### AU (Logic Pro only)
```bash
cp -R macOS/AU/0MGE.component ~/Library/Audio/Plug-Ins/Components/
```

### System-wide (requires admin)
```bash
sudo cp -R macOS/VST3/0MGE.vst3 /Library/Audio/Plug-Ins/VST3/
sudo cp -R macOS/AU/0MGE.component /Library/Audio/Plug-Ins/Components/
```

Then restart your DAW.

## Windows

### VST3
Copy `Windows/0MGE.vst3` to:
- `C:\Program Files\VST3\` (system-wide)
- Or your DAW's custom VST folder

## Uninstall
Delete the plugin files:
```bash
rm -rf ~/Library/Audio/Plug-Ins/VST3/0MGE.vst3
rm -rf ~/Library/Audio/Plug-Ins/Components/0MGE.component
```
