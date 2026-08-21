#!/bin/bash
# 0MGE — One-click bootstrap (macOS / Linux)
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/venv"
REQ="$DIR/requirements.txt"

# Hide terminal on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    osascript -e 'tell application "System Events" to set visible of process "Terminal" to false' 2>/dev/null || true
fi

echo "0MGE — Music Granular Engine"
echo ""

# Check Python
if command -v python3 &>/dev/null; then
    PY=$(command -v python3)
elif command -v python &>/dev/null; then
    PY=$(command -v python)
else
    echo "Python not found."
    echo ""
    echo "Install Python 3.10+ from:"
    echo "  https://www.python.org/downloads/"
    echo ""
    read -p "Press Enter to open python.org... "
    open "https://www.python.org/downloads/" 2>/dev/null || xdg-open "https://www.python.org/downloads/" 2>/dev/null || true
    exit 1
fi

# Check Python version (need 3.10+)
PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PY" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PY" -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "Python $PY_VER found, but 3.10+ required."
    echo "Install from: https://www.python.org/downloads/"
    exit 1
fi
echo "Python $PY_VER OK"

# Create venv
if [ ! -d "$VENV" ]; then
    echo "Creating venv..."
    "$PY" -m venv "$VENV"
fi
source "$VENV/bin/activate"

# Install deps (quiet on subsequent runs)
echo "Checking dependencies..."
pip install --quiet --upgrade pip 2>/dev/null
pip install --quiet -r "$REQ" 2>/dev/null

echo ""
echo "Ready!"
echo ""

# Launch app
python "$DIR/app.py"
