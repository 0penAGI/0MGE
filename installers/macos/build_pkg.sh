#!/bin/bash
# 0MGE macOS Installer Builder
# Creates .pkg that installs VST3 + AU to system folders

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$ROOT_DIR/installers/macos/build"
PKG_DIR="$BUILD_DIR/0MGE-pkg"
PLUGIN_DIR="$ROOT_DIR/vst/build"

VERSION="1.0.0"
PKG_NAME="0MGE-${VERSION}-macOS.pkg"

echo "🔧 Building 0MGE macOS installer v${VERSION}..."

# Clean
rm -rf "$BUILD_DIR"
mkdir -p "$PKG_DIR"

# Check if plugins exist
if [ ! -d "$PLUGIN_DIR/0MGE.vst3" ]; then
    echo "❌ VST3 not found. Build plugin first:"
    echo "   cd vst && mkdir build && cd build && cmake -G Xcode .. && cmake --build . --config Release"
    exit 1
fi

# Copy plugins to package staging
mkdir -p "$PKG_DIR/Library/Audio/Plug-Ins/VST3"
mkdir -p "$PKG_DIR/Library/Audio/Plug-Ins/Components"

cp -R "$PLUGIN_DIR/0MGE.vst3" "$PKG_DIR/Library/Audio/Plug-Ins/VST3/"
if [ -d "$PLUGIN_DIR/0MGE.component" ]; then
    cp -R "$PLUGIN_DIR/0MGE.component" "$PKG_DIR/Library/Audio/Plug-Ins/Components/"
fi

# Also include standalone app if it exists
if [ -d "$PLUGIN_DIR/0MGE.app" ]; then
    mkdir -p "$PKG_DIR/Applications"
    cp -R "$PLUGIN_DIR/0MGE.app" "$PKG_DIR/Applications/"
fi

# Create post-install script
cat > "$PKG_DIR/postinstall" << 'EOF'
#!/bin/bash
echo "✅ 0MGE installed!"
echo "   VST3: /Library/Audio/Plug-Ins/VST3/0MGE.vst3"
echo "   AU:   /Library/Audio/Plug-Ins/Components/0MGE.component"
echo "   Restart your DAW to see the plugin."
EOF
chmod +x "$PKG_DIR/postinstall"

# Build component package
pkgbuild --root "$PKG_DIR" \
    --identifier "com.0penagi.0mge" \
    --version "$VERSION" \
    --install-location "/" \
    --scripts "$PKG_DIR" \
    "$BUILD_DIR/0MGE-component.pkg"

# Build product archive
cat > "$BUILD_DIR/distribution.xml" << EOF
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>0MGE - 0penAGI Music Granular Engine</title>
    <options customize="never" require-scripts="true" hostArchitectures="x86_64,arm64"/>
    <domains enable_localSystem="true"/>
    <choices-outline>
        <line choice="0MGE"/>
    </choices-outline>
    <choice id="0MGE" title="0MGE Plugin">
        <pkg-ref id="com.0penagi.0mge"/>
    </choice>
    <pkg-ref id="com.0penagi.0mge" version="${VERSION}">0MGE-component.pkg</pkg-ref>
</installer-gui-script>
EOF

productbuild --distribution "$BUILD_DIR/distribution.xml" \
    --package-path "$BUILD_DIR" \
    "$BUILD_DIR/$PKG_NAME"

# Copy to release
RELEASE_DIR="$ROOT_DIR/release/macos"
mkdir -p "$RELEASE_DIR"
cp "$BUILD_DIR/$PKG_NAME" "$RELEASE_DIR/"

echo ""
echo "✅ Installer: $RELEASE_DIR/$PKG_NAME"
echo "   Size: $(du -sh "$RELEASE_DIR/$PKG_NAME" | cut -f1)"
echo "   Install: double-click .pkg → requires admin password"
