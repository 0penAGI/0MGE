"""
0MGE — 0 Music Granular Engine
Minimal desktop UI. One button. One result. Zero friction. by Slut Online.

Rewritten on PySide6 (Qt). Qt ships its own rendering engine and does not
depend on the OS-provided Tcl/Tk, so the "blank window" issue from
customtkinter-on-deprecated-system-Tk cannot happen here.

Install (inside your existing venv, nothing else changes):
    pip install PySide6 soundfile numpy
"""
import os, sys, json, subprocess, hashlib
import numpy as np
from pathlib import Path

from PySide6.QtCore import Qt, QObject, Signal, QThread, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFrame, QProgressBar, QComboBox, QSlider, QCheckBox, QLineEdit, QStyle
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

DIR = Path(__file__).parent
OUT = DIR / "granular_output"
CACHE = DIR / "scan_index.json"
ENGINE = DIR / "granular_field.py"
SETTINGS = DIR / "app_settings.json"

SCAN_DIRS = [
    Path.home() / "Music",
    Path.home() / "Documents",
    Path.home() / "Downloads",
    Path.home() / "Desktop",
    Path.home() / "Documents" / "Ableton",
    Path.home() / "Music" / "Ableton",
]
for v in Path("/Volumes").iterdir() if Path("/Volumes").exists() else []:
    if v.name not in ("Macintosh HD", "Preboot", "VM"):
        SCAN_DIRS.append(v)
SKIP_DIRS = {"Factory Packs", "User Library", "Live Recordings"}
AUDIO_EXTS = {".wav", ".aiff", ".aif", ".flac", ".mp3", ".ogg", ".m4a"}

BG = "#f4f4f2"
SURFACE = "#ffffff"
GLASS = "#ffffff"
GLASS_LIGHT = "#eeeeec"
ACCENT = "#111111"
ACCENT_DIM = "#666666"
TEXT = "#111111"
TEXT_DIM = "#777777"
SUCCESS = "#222222"
WARN = "#111111"


def load_scan_index():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def save_scan_index(idx):
    CACHE.write_text(json.dumps(idx))


def load_settings():
    if SETTINGS.exists():
        return json.loads(SETTINGS.read_text())
    return {}


def save_settings(s):
    SETTINGS.write_text(json.dumps(s))


def scan_audio(scan_dirs, skip=SKIP_DIRS, exts=AUDIO_EXTS):
    old = load_scan_index()
    idx = {}
    files = []
    for d in scan_dirs:
        if not d.exists():
            continue
        for root, dirs, fnames in os.walk(str(d)):
            dirs[:] = [dd for dd in dirs if dd not in skip]
            for fname in fnames:
                fp = Path(root) / fname
                if fp.suffix.lower() not in exts:
                    continue
                stat = fp.stat()
                key = str(fp)
                cached = old.get(key)
                if cached and cached.get("mtime") == stat.st_mtime:
                    idx[key] = cached
                else:
                    h = hashlib.md5(fp.read_bytes()).hexdigest()
                    idx[key] = {"path": key, "mtime": stat.st_mtime,
                                "size": stat.st_size, "md5": h}
                files.append(key)
    save_scan_index(idx)
    return files


STYLESHEET = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-family: 'Helvetica Neue', 'SF Pro Text', Arial; }}
QFrame#Surface {{ background: rgba(255,255,255,235); border: none; border-radius: 0px; }}
QLabel#Header {{ font-family: 'SF Mono'; font-size: 30px; font-weight: 700; color: {TEXT}; }}
QLabel#Sub {{ background: transparent; border: none; padding: 0; margin: 0; font-family: 'SF Mono'; font-size: 10px; letter-spacing: 2px; color: {TEXT_DIM}; }}
QLabel#Dim {{ background: transparent; border: none; padding: 0; margin: 0; color: {TEXT_DIM}; font-size: 12px; }}
QLabel#SectionLabel {{ background: transparent; border: none; padding: 0; margin: 0; font-family: 'SF Mono'; font-size: 9px; font-weight: 700; letter-spacing: 2px; color: {TEXT_DIM}; }}
QLabel#Pill {{ background: transparent; color: #555555; border: none; font-family: 'SF Mono'; font-size: 10px; font-weight: 700; padding: 0; border-radius: 0px; }}
QLabel#OutName {{ background: transparent; border: none; padding: 0; margin: 0; font-family: 'SF Mono'; font-size: 13px; color: {TEXT}; }}
QLabel#OutInfo {{ background: transparent; border: none; padding: 0; margin: 0; color: {TEXT_DIM}; font-size: 11px; }}
QPushButton#Generate {{ background: #111111; color: #ffffff; font-size: 16px; font-weight: 700; border: none; border-radius: 0px; padding: 16px; outline: none; }}
QPushButton#Generate:hover {{ background: #2a2a2a; }}
QPushButton#Generate:disabled {{ background: #aaaaaa; color: #eeeeee; }}
QPushButton#Flat {{ background: transparent; color: #222222; border: none; border-radius: 0px; font-size: 11px; padding: 7px 12px; }}
QPushButton#Flat:hover {{ background: #e8e8e6; }}
QProgressBar {{ background: #dededb; border: none; border-radius: 0px; max-height: 5px; }}
QProgressBar::chunk {{ background: #111111; border-radius: 0px; }}
QComboBox {{ background: transparent; color: #111111; border: none; border-radius: 0px; padding: 5px 0; }}
QComboBox:hover {{ background: transparent; border: none; }}
QComboBox:focus {{ background: transparent; border: none; outline: none; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox::down-arrow {{ image: none; width: 0px; height: 0px; }}
QComboBox QAbstractItemView {{ background: #ffffff; color: #111111; border: none; outline: none; padding: 4px; selection-background-color: #eeeeec; selection-color: #111111; border-radius: 0px; }}
QComboBox QAbstractItemView::item {{ background: #ffffff; color: #111111; padding: 7px 10px; border: none; }}
QComboBox QAbstractItemView::item:selected {{ background: #eeeeec; color: #111111; border: none; }}
QLineEdit {{ background: transparent; color: #111111; border: none; border-radius: 0px; padding: 5px 0; }}
QLineEdit:focus {{ background: transparent; border: none; outline: none; }}
QLabel#SettingsLabel {{ background: transparent; border: none; padding: 0; color: #666666; }}
QCheckBox {{ background: transparent; border: none; color: #666666; font-size: 12px; }}
QSlider {{ background: transparent; border: none; }}
QSlider::groove:horizontal {{ background: #d5d5d2; height: 4px; border-radius: 0px; }}
QSlider::handle:horizontal {{ background: #111111; width: 12px; margin: -4px 0; border-radius: 0px; }}
QLabel#WaveHint {{ color: #777777; font-family: 'SF Mono'; font-size: 10px; }}
"""


class WaveformWidget(QWidget):
    """Native QPainter waveform — no external canvas widget needed."""
    clicked = Signal()

    def __init__(self):
        super().__init__()
        self.samples = None
        self.playback_position = 0.0
        self.generation_phase = 0.0
        self.generating = False
        self.setMinimumHeight(150)

    def set_audio(self, audio):
        w = max(self.width(), 660)
        n = len(audio)
        step = max(1, n // w)
        peaks = []
        for x in range(0, min(w, n // step)):
            i = x * step
            chunk = audio[i:i + step]
            peaks.append(float(np.max(np.abs(chunk))) if len(chunk) > 0 else 0.0)
        self.samples = peaks
        self.update()

    def clear(self):
        self.samples = None
        self.update()

    def set_playback_position(self, position):
        self.playback_position = max(0.0, min(1.0, position))
        self.update()

    def set_generating(self, active):
        self.generating = active
        if not active:
            self.generation_phase = 0.0
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.samples:
            self.clicked.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(SURFACE))
        w, h = self.width(), self.height()
        mid = h / 2

        if self.generating:
            # Living granular field: moving layered sine envelopes, no fake audio data.
            import math
            phase = self.generation_phase
            for x in range(0, w, 3):
                t = x / max(1, w)
                env = 0.25 + 0.75 * (0.5 + 0.5 * math.sin(t * 12.0 + phase))
                wave = math.sin(t * 38.0 + phase * 2.1) * 0.55 + math.sin(t * 83.0 - phase * 1.4) * 0.25
                bar_h = int((h * 0.16 + h * 0.20 * abs(wave)) * env)
                p.setPen(QPen(QColor("#999999"), 1))
                p.drawLine(x, int(mid - bar_h), x, int(mid + bar_h))
            p.setPen(QPen(QColor("#777777"), 1))
            p.drawText(self.rect().adjusted(18, 0, -18, 0), Qt.AlignBottom | Qt.AlignLeft, "GENERATING · GRANULAR FIELD")
            p.end()
            return

        if not self.samples:
            p.setPen(QColor(TEXT_DIM))
            p.setFont(QFont("SF Mono", 10, QFont.Bold))
            p.drawText(self.rect(), Qt.AlignCenter, "CLICK GENERATE")
            p.end()
            return

        accent = QPen(QColor("#222222"), 1)
        dim = QPen(QColor("#cccccc"), 1)
        for x, peak in enumerate(self.samples):
            bar_h = int(peak * (mid - 8))
            p.setPen(accent if peak < 0.8 else QPen(QColor("#111111"), 1))
            p.drawLine(x, int(mid - bar_h), x, int(mid + bar_h))

        # Playback progress: dark track + bright playhead + subtle glow band.
        if self.playback_position > 0.0:
            px = int(self.playback_position * max(1, w - 1))
            p.setPen(QPen(QColor("#777777"), 1))
            p.drawLine(px, 8, px, h - 8)
            p.setPen(QPen(QColor("#111111"), 2))
            p.drawLine(px, 8, px, h - 8)
            p.setBrush(QColor("#111111"))
            p.setPen(Qt.NoPen)
            p.drawEllipse(px - 4, 4, 8, 8)
        p.end()


class ScanWorker(QObject):
    finished = Signal(int)

    def run(self):
        files = scan_audio(SCAN_DIRS)
        self.finished.emit(len(files))


class GenerateWorker(QObject):
    progress = Signal(int, str)
    done = Signal(object)
    error = Signal(str)

    def __init__(self, bars, bpm, temp, seed, multi, closed, train_multi):
        super().__init__()
        self.bars, self.bpm, self.temp = bars, bpm, temp
        self.seed, self.multi, self.closed, self.train_multi = seed, multi, closed, train_multi

    def run(self):
        try:
            args = [sys.executable, str(ENGINE),
                    "--bars", str(self.bars), "--bpm", str(self.bpm),
                    "--temperature", str(self.temp), "--generate-only"]
            if self.seed is not None:
                args.extend(["--seed", str(self.seed)])
            if self.multi:
                args.append("--multi-stream")
            if self.closed:
                args.extend(["--closed-loop", "8"])
            if self.train_multi:
                args.append("--train-multi")

            self.progress.emit(10, "")

            proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    cwd=str(DIR))
            for line in proc.stdout:
                l = line.strip()
                if "GENERATING" in l or "MULTI-STREAM" in l:
                    self.progress.emit(30, "Generating...")
                elif "Mastering" in l:
                    self.progress.emit(80, "Mastering...")
                elif "✅" in l:
                    self.progress.emit(95, "Done!")
            proc.wait()

            wavs = sorted(OUT.glob("*.wav"), key=lambda f: f.stat().st_mtime)
            latest = wavs[-1] if wavs else None
            self.done.emit(latest)
        except Exception as e:
            self.error.emit(str(e))


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("0MGE")
        self.setFixedSize(720, 780)
        self.setStyleSheet(STYLESHEET)

        self.generating = False
        self.last_output = None
        self.settings_open = False
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)
        self.player.positionChanged.connect(self._on_playback_position)
        self.player.durationChanged.connect(self._on_playback_duration)
        self._playback_duration = 0
        self._generation_timer = None
        self._generation_phase = 0.0
        self._display_progress = 0.0
        self._target_progress = 0.0
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(16)
        self._progress_timer.timeout.connect(self._animate_progress)

        self._build_ui()
        self._load_settings()
        self._check_pool()
        self._run_scan()
    def _animate_progress(self):
        delta = self._target_progress - self._display_progress
        if abs(delta) < 0.15:
            self._display_progress = self._target_progress
            if self._display_progress >= 100:
                self._progress_timer.stop()
        else:
            # Smooth ease-out toward the real engine state.
            self._display_progress += delta * 0.055
        self.progress.setValue(int(self._display_progress))

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(30, 30, 30, 30)
        root.setSpacing(10)

        header = QVBoxLayout()
        header.setSpacing(3)
        title = QLabel("0MGE")
        title.setObjectName("Header")
        title.setAlignment(Qt.AlignCenter)
        sub = QLabel("MUSIC GRANULAR ENGINE")
        sub.setObjectName("Sub")
        sub.setAlignment(Qt.AlignCenter)
        header.addWidget(title)
        header.addWidget(sub)
        root.addLayout(header)

        self.status_label = QLabel(""); self.status_label.setObjectName("Dim")
        root.addWidget(self.status_label)

        # pills block removed

        self.gen_btn = QPushButton("Generate")
        self.gen_btn.setObjectName("Generate")
        self.gen_btn.clicked.connect(self._on_generate)
        root.addWidget(self.gen_btn)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100)
        root.addWidget(self.progress)

        field_lbl = QLabel("GENERATED FIELD"); field_lbl.setObjectName("SectionLabel")
        root.addWidget(field_lbl)
        self.waveform = WaveformWidget()
        self.waveform.setToolTip("Click to play / pause")
        root.addWidget(self.waveform)
        self.waveform.clicked.connect(self._toggle_playback)

        out_lbl = QLabel("OUTPUT"); out_lbl.setObjectName("SectionLabel")
        root.addWidget(out_lbl)

        out_card = QFrame(); out_card.setObjectName("Surface")
        # Remove any border, background, padding, or frameShape settings here (leave only objectName)
        out_row = QHBoxLayout(out_card)
        out_col = QVBoxLayout()
        wav_lbl = QLabel("WAV"); wav_lbl.setObjectName("SectionLabel")
        self.output_name = QLabel(""); self.output_name.setObjectName("OutName")
        self.output_info = QLabel(""); self.output_info.setObjectName("OutInfo")
        out_col.addWidget(wav_lbl); out_col.addWidget(self.output_name); out_col.addWidget(self.output_info)
        out_row.addLayout(out_col)
        out_row.addStretch()
        finder_btn = QPushButton("Show in Finder"); finder_btn.setObjectName("Flat")
        finder_btn.clicked.connect(self._open_folder)
        out_row.addWidget(finder_btn, alignment=Qt.AlignVCenter)
        root.addWidget(out_card)

        self.settings_btn = QPushButton("Settings  ▸"); self.settings_btn.setObjectName("Flat")
        self.settings_btn.clicked.connect(self._toggle_settings)
        root.addWidget(self.settings_btn, alignment=Qt.AlignLeft)

        self.settings_frame = QFrame(); self.settings_frame.setObjectName("Surface")
        self._build_settings()
        self.settings_frame.hide()
        root.addWidget(self.settings_frame)

        root.addStretch()
        footer = QLabel("0MGE · LOCAL AUDIO INTELLIGENCE by 0penAGI"); footer.setObjectName("Sub")
        root.addWidget(footer)
    def _on_playback_duration(self, duration):
        self._playback_duration = duration

    def _on_playback_position(self, position):
        if self._playback_duration > 0:
            self.waveform.set_playback_position(position / self._playback_duration)
        else:
            self.waveform.set_playback_position(0.0)

    def _animate_generation(self):
        if not self.generating:
            return
        self._generation_phase += 0.16
        self.waveform.generation_phase = self._generation_phase
        self.waveform.update()

    def _toggle_playback(self):
        if not self.last_output or not self.last_output.exists():
            return
        if self.player.source().toLocalFile() != str(self.last_output):
            self.waveform.set_playback_position(0.0)
            self.player.setSource(str(self.last_output))
            self.player.play()
        elif self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _build_settings(self):
        col = QVBoxLayout(self.settings_frame)

        row1 = QHBoxLayout()
        bars_label = QLabel("Bars"); bars_label.setObjectName("SettingsLabel")
        row1.addWidget(bars_label)
        self.bars_box = QComboBox()
        self.bars_box.addItems(["15", "30", "60", "120"])
        self.bars_box.setCurrentText("60")
        self.bars_box.setMinimumWidth(70)
        self.bars_box.setCursor(Qt.PointingHandCursor)
        row1.addWidget(self.bars_box)

        bpm_label = QLabel("BPM"); bpm_label.setObjectName("SettingsLabel")
        row1.addWidget(bpm_label)
        self.bpm_box = QComboBox()
        self.bpm_box.addItems(["80", "90", "100", "110", "120", "130", "140"])
        self.bpm_box.setCurrentText("120")
        self.bpm_box.setMinimumWidth(70)
        self.bpm_box.setCursor(Qt.PointingHandCursor)
        row1.addWidget(self.bpm_box)

        temp_label = QLabel("Temp"); temp_label.setObjectName("SettingsLabel")
        row1.addWidget(temp_label)
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(30, 150)
        self.temp_slider.setValue(80)
        row1.addWidget(self.temp_slider)
        row1.addStretch()
        col.addLayout(row1)

        row2 = QHBoxLayout()
        self.multi_check = QCheckBox("Multi-stream (6 layers)"); self.multi_check.setChecked(True)
        self.closed_check = QCheckBox("Closed-loop")
        row2.addWidget(self.multi_check); row2.addWidget(self.closed_check)
        row2.addStretch()
        col.addLayout(row2)

        row3 = QHBoxLayout()
        seed_label = QLabel("Seed"); seed_label.setObjectName("SettingsLabel")
        row3.addWidget(seed_label)
        self.seed_edit = QLineEdit()
        self.seed_edit.setFixedWidth(70)
        row3.addWidget(self.seed_edit)
        row3.addStretch()
        col.addLayout(row3)

        row4 = QHBoxLayout()
        self.train_multi_check = QCheckBox("Train MultiNavigator")
        row4.addWidget(self.train_multi_check)
        row4.addStretch()
        col.addLayout(row4)

    def _toggle_settings(self):
        self.settings_open = not self.settings_open
        self.settings_frame.setVisible(self.settings_open)
        self.settings_btn.setText("Settings  ▾" if self.settings_open else "Settings  ▸")

    def _load_settings(self):
        s = load_settings()
        if s.get("bars"): self.bars_box.setCurrentText(str(s["bars"]))
        if s.get("bpm"): self.bpm_box.setCurrentText(str(s["bpm"]))
        if s.get("temp"): self.temp_slider.setValue(int(float(s["temp"]) * 100))
        if "multi" in s: self.multi_check.setChecked(bool(s["multi"]))
        if "closed" in s: self.closed_check.setChecked(bool(s["closed"]))
        if s.get("seed"): self.seed_edit.setText(str(s["seed"]))

    def _save_settings(self):
        save_settings({
            "bars": self.bars_box.currentText(),
            "bpm": self.bpm_box.currentText(),
            "temp": self.temp_slider.value() / 100.0,
            "multi": self.multi_check.isChecked(),
            "closed": self.closed_check.isChecked(),
            "seed": self.seed_edit.text(),
        })

    def _check_pool(self):
        pool = DIR / "granular_pool_lite.npz"
        model = DIR / "granular_navigator_v2.pt"
        if pool.exists() and model.exists():
            d = np.load(pool, allow_pickle=True)
            n = {k: len(d[f"{k}_feats"]) for k in ["micro", "meso", "macro"]}
            total = sum(n.values())
            self.status_label.setText(f"Pool ready: {total:,} grains")
        else:
            self.status_label.setText("Ready — first generate will build pool (~2 min)")

    def _run_scan(self):
        self._scan_thread = QThread()
        self._scan_worker = ScanWorker()
        self._scan_worker.moveToThread(self._scan_thread)
        self._scan_thread.started.connect(self._scan_worker.run)
        self._scan_worker.finished.connect(self._on_scan_done)
        self._scan_worker.finished.connect(self._scan_thread.quit)
        self._scan_thread.start()

    def _on_scan_done(self, n):
        self.status_label.setText(f"Pool ready · {n} files scanned")

    def _on_generate(self):
        if self.generating:
            return
        self._save_settings()
        self.generating = True
        self.gen_btn.setText("Generating...")
        self.gen_btn.setEnabled(False)
        self._display_progress = 0.0
        self._target_progress = 0.0
        self.progress.setValue(0)
        self._progress_timer.start()
        self.output_name.setText("Working...")
        self.output_info.setText("")
        self.waveform.clear()
        self.waveform.set_generating(True)
        if self._generation_timer is None:
            self._generation_timer = QTimer(self)
            self._generation_timer.timeout.connect(self._animate_generation)
        self._generation_timer.start(33)

        bars = int(self.bars_box.currentText())
        bpm = int(self.bpm_box.currentText())
        temp = self.temp_slider.value() / 100.0
        seed_str = self.seed_edit.text().strip()
        seed = int(seed_str) if seed_str.isdigit() else None
        multi = self.multi_check.isChecked()
        closed = self.closed_check.isChecked()
        train_multi = self.train_multi_check.isChecked()

        self._gen_thread = QThread()
        self._gen_worker = GenerateWorker(bars, bpm, temp, seed, multi, closed, train_multi)
        self._gen_worker.moveToThread(self._gen_thread)
        self._gen_thread.started.connect(self._gen_worker.run)
        self._gen_worker.progress.connect(self._on_gen_progress)
        self._gen_worker.done.connect(self._on_gen_done)
        self._gen_worker.error.connect(self._on_gen_error)
        self._gen_worker.done.connect(self._gen_thread.quit)
        self._gen_worker.error.connect(self._gen_thread.quit)
        self._gen_thread.start()

        self._bars, self._bpm = bars, bpm

    def _on_gen_progress(self, val, text):
        self._target_progress = float(val)
        if text:
            self.output_name.setText(text)

    def _on_gen_done(self, latest):
        self._target_progress = 100.0
        self._progress_timer.start()
        self.waveform.set_generating(False)
        if self._generation_timer:
            self._generation_timer.stop()
        if latest:
            self.last_output = latest
            self.waveform.set_playback_position(0.0)
            size_mb = latest.stat().st_size / 1024 / 1024
            dur = self._bars * 4 * 60 / self._bpm
            self.output_name.setText(latest.name)
            self.output_info.setText(f"{dur:.0f}s · {size_mb:.1f} MB · {self._bars} bars @ {self._bpm} BPM")
            try:
                import soundfile as sf
                audio, sr = sf.read(str(latest))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                self.waveform.set_audio(audio)
            except Exception:
                pass
            # self._open_folder_select(latest)  # Removed auto open Finder
        else:
            self.output_name.setText("No output")
        self.generating = False
        self.gen_btn.setText("Generate")
        self.gen_btn.setEnabled(True)

    def _on_gen_error(self, msg):
        self._target_progress = 0.0
        self._progress_timer.stop()
        self._display_progress = 0.0
        self.progress.setValue(0)
        self.waveform.set_generating(False)
        if self._generation_timer:
            self._generation_timer.stop()
        self.output_name.setText(f"Error: {msg}")
        self.generating = False
        self.gen_btn.setText("Generate")
        self.gen_btn.setEnabled(True)

    def _open_folder(self):
        if self.last_output:
            self._open_folder_select(self.last_output)
        else:
            subprocess.run(["open", str(OUT)])

    def _open_folder_select(self, path):
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", str(path)])
        elif sys.platform == "win32":
            subprocess.run(["explorer", "/select,", str(path)])
        else:
            subprocess.run(["xdg-open", str(path.parent)])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec())
