# -*- mode: python ; coding: utf-8 -*-
# 0MGE Neural Engine — desktop app bundle (macOS .app)

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['granular_field', 'mutagen.id3'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
        'yt_dlp', 'yt_dlp_ejs', 'pygame', 'pygame_widgets',
        'pymorphy2', 'pymorphy3', 'language_tool_python',
        'lightning', 'pytorch_lightning', 'torchvision', 'torchaudio',
        'torchdiffeq', 'torchsde', 'torchquantum', 'torchpack',
        'torchmetrics', 'torch_augmentations', 'torch_pitch_shift',
        'bitsandbytes', 'clip_anytorch', 'open_clip_torch',
        'opencv', 'cv2', 'pytesseract', 'julius', 'llama_cpp_python',
        'ipython', 'jupyter', 'notebook', 'jupyter_client', 'jupyter_core',
        'pyTelegramBotAPI', 'python_telegram_bot', 'pytoniq', 'pytonlib',
        'pytonconnect', 'peewee', 'aiogram',
        # heavy junk dragged in by hook-contrib from this shared venv:
        'transformers', 'tokenizers', 'nltk', 'playwright', 'sphinx',
        'matplotlib', 'pyarrow', 'astropy', 'boto3', 'botocore', 'babel',
        'plotly', 'jieba', 'sudachipy', 'sudachidict_core', 'mecab_python3',
'jedi', 'IPython', 'Cython', 'MeCab', 'pybind11',
        'statsmodels',
        'tensorflow', 'keras', 'jax', 'flax', 'sounddevice',
        'onnxruntime', 'imageio_ffmpeg', 'av', 'faiss', 'spacy',
        'pandas', 'skimage', 'grpc', 'cryptography', 'pyspark',
        'openai', 'anthropic', 'django', 'flask', 'fastapi',
        'PySide6.QtQml', 'PySide6.QtQuick', 'PySide6.QtQuickWidgets',
        'PySide6.QtQmlModels', 'PySide6.QtQmlWorkerScript',
        'PySide6.QtVirtualKeyboard', 'PySide6.QtVirtualKeyboardQml',
        'PySide6.QtPdf', 'PySide6.QtPdfWidgets', 'PySide6.QtSvg',
        'PySide6.QtOpenGL', 'PySide6.QtOpenGLWidgets',
        'PySide6.QtDBus', 'PySide6.QtConcurrent',
        'PySide6.QtMultimediaWidgets', 'PySide6.QtCharts',
        'PySide6.QtDesigner',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='0MGE',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='0MGE.icns',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='0MGE',
)

app = BUNDLE(
    coll,
    name='0MGE.app',
    icon='0MGE.icns',
    bundle_identifier='com.0penagi.0mge.neural',
)