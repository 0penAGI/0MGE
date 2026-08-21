#!/bin/bash
# 0MGE macOS Installer Builder
# Creates two .pkg files: plugins (VST3+AU) and app (standalone)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
BUILD_DIR="$ROOT_DIR/installers/macos/build"
PLUGIN_DIR="$ROOT_DIR/vst/build/ZeroMGE_Project_artefacts/Release"

VERSION="1.0.0"

echo "🔧 Building 0MGE macOS installers v${VERSION}..."

# Clean
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Check if plugins exist
if [ ! -d "$PLUGIN_DIR/VST3/0MGE.vst3" ]; then
    echo "❌ VST3 not found. Build plugin first:"
    echo "   cd vst && mkdir build && cd build && cmake -G Xcode .. && cmake --build . --config Release"
    exit 1
fi

# === 1. PLUGINS .pkg (VST3 + AU) ===
echo "📦 Building plugins installer..."
PLUGINS_PKG_DIR="$BUILD_DIR/plugins-pkg"
mkdir -p "$PLUGINS_PKG_DIR/Library/Audio/Plug-Ins/VST3"
mkdir -p "$PLUGINS_PKG_DIR/Library/Audio/Plug-Ins/Components"

cp -R "$PLUGIN_DIR/VST3/0MGE.vst3" "$PLUGINS_PKG_DIR/Library/Audio/Plug-Ins/VST3/"
if [ -d "$PLUGIN_DIR/AU/0MGE.component" ]; then
    cp -R "$PLUGIN_DIR/AU/0MGE.component" "$PLUGINS_PKG_DIR/Library/Audio/Plug-Ins/Components/"
fi

cat > "$PLUGINS_PKG_DIR/postinstall" << 'EOF'
#!/bin/bash
echo ""
echo "✅ 0MGE Plugins installed!"
echo "   VST3: /Library/Audio/Plug-Ins/VST3/0MGE.vst3"
echo "   AU:   /Library/Audio/Plug-Ins/Components/0MGE.component"
echo "   Restart your DAW to see the plugin."
echo ""
EOF
chmod +x "$PLUGINS_PKG_DIR/postinstall"

pkgbuild --root "$PLUGINS_PKG_DIR" \
    --identifier "com.0penagi.0mge.plugins" \
    --version "$VERSION" \
    --install-location "/" \
    --scripts "$PLUGINS_PKG_DIR" \
    "$BUILD_DIR/0MGE-plugins-component.pkg"

# === 2. APP .pkg (Standalone) ===
APP_PKG_DIR="$BUILD_DIR/app-pkg"
APP_BUILT=false

if [ -d "$PLUGIN_DIR/Standalone/0MGE.app" ]; then
    echo "📦 Building app installer..."
    mkdir -p "$APP_PKG_DIR/Applications"
    cp -R "$PLUGIN_DIR/Standalone/0MGE.app" "$APP_PKG_DIR/Applications/"

    cat > "$APP_PKG_DIR/postinstall" << 'EOF'
#!/bin/bash
echo ""
echo "✅ 0MGE App installed!"
echo "   App: /Applications/0MGE.app"
echo "   Open it, select your music folder, hit Generate."
echo ""
EOF
    chmod +x "$APP_PKG_DIR/postinstall"

    pkgbuild --root "$APP_PKG_DIR" \
        --identifier "com.0penagi.0mge.app" \
        --version "$VERSION" \
        --install-location "/" \
        --scripts "$APP_PKG_DIR" \
        "$BUILD_DIR/0MGE-app-component.pkg"
    APP_BUILT=true
fi

# === 3. DISTRIBUTION XML (both choices) ===
if [ "$APP_BUILT" = true ]; then
cat > "$BUILD_DIR/distribution.xml" << EOF
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>0MGE - 0penAGI Neural Granular Engine</title>
    <options customize="never" require-scripts="true" hostArchitectures="x86_64,arm64"/>
    <domains enable_localSystem="true"/>
    <choices-outline>
        <line choice="plugins"/>
        <line choice="app"/>
    </choices-outline>
    <choice id="plugins" title="VST3 + AU Plugin" description="Granular processor for your DAW (Logic, Ableton, Reaper, etc)">
        <pkg-ref id="com.0penagi.0mge.plugins"/>
    </choice>
    <choice id="app" title="Standalone App" description="Desktop app to scan your music, train, and generate new sound">
        <pkg-ref id="com.0penagi.0mge.app"/>
    </choice>
    <pkg-ref id="com.0penagi.0mge.plugins" version="${VERSION}">0MGE-plugins-component.pkg</pkg-ref>
    <pkg-ref id="com.0penagi.0mge.app" version="${VERSION}">0MGE-app-component.pkg</pkg-ref>
</installer-gui-script>
EOF
else
cat > "$BUILD_DIR/distribution.xml" << EOF
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>0MGE - 0penAGI Neural Granular Engine</title>
    <options customize="never" require-scripts="true" hostArchitectures="x86_64,arm64"/>
    <domains enable_localSystem="true"/>
    <choices-outline>
        <line choice="plugins"/>
    </choices-outline>
    <choice id="plugins" title="VST3 + AU Plugin" description="Granular processor for your DAW">
        <pkg-ref id="com.0penagi.0mge.plugins"/>
    </choice>
    <pkg-ref id="com.0penagi.0mge.plugins" version="${VERSION}">0MGE-plugins-component.pkg</pkg-ref>
</installer-gui-script>
EOF
fi

# === 4. FINAL .pkg ===
productbuild --distribution "$BUILD_DIR/distribution.xml" \
    --package-path "$BUILD_DIR" \
    "$BUILD_DIR/0MGE-${VERSION}-macOS.pkg"

# Copy to release
RELEASE_DIR="$ROOT_DIR/release/macos"
mkdir -p "$RELEASE_DIR"
cp "$BUILD_DIR/0MGE-${VERSION}-macOS.pkg" "$RELEASE_DIR/"

echo ""
echo "✅ Installer: $RELEASE_DIR/0MGE-${VERSION}-macOS.pkg"
echo "   Size: $(du -sh "$RELEASE_DIR/0MGE-${VERSION}-macOS.pkg" | cut -f1)"
if [ "$APP_BUILT" = true ]; then
    echo "   Choices: VST3+AU Plugin | Standalone App"
else
    echo "   Choices: VST3+AU Plugin only (app not built)"
fi
echo "   Install: double-click → pick what you need → admin password"
