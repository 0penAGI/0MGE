#!/usr/bin/env python3
"""
GENOME — Spectral Genome Scanner & Track Generator (v2)

Что изменилось относительно v1 ("тупая склейка"):

  1. BEAT-ALIGNED EXTRACTION
     Раньше фрагмент вырезался со случайного сэмпла внутри файла — без
     привязки к сетке долей. Теперь для каждого извлекаемого куска
     детектируются реальные beat-times в локальном окне файла, и кусок
     берётся строго от доли до доли (bars*4 битов) — ритмически
     когерентный фрагмент, а не рандомный срез.

  2. TEMPO NORMALIZATION
     Каждый извлечённый (уже бит-выровненный) кусок тайм-стретчится
     (phase vocoder, librosa.effects.time_stretch) до точной длины бара
     в целевом BPM трека. Раньше все куски считались одной длины
     (bar_samples от глобального медианного BPM) независимо от реального
     темпа источника — это и давало ощущение "нарезки".

  3. HARMONIC / KEY MATCHING
     Оценка тональности (Krumhansl-Schmuckler профили корреляции с
     хромой) для каждого фрагмента и для целевой тональности жанра.
     Кусок питч-шифтится на минимальное количество полутонов, чтобы
     попасть в целевую тональность (сдвиг капается, чтобы не убить
     звук). Раньше тональность не учитывалась вообще.

  4. STRUCTURE-AWARE ARC
     Вместо чистого марковского блуждания по архетипам — макро-план
     секций (intro → build → drop → break → drop → outro) с целевым
     уровнем энергии на каждую секцию. Выбор архетипа — взвешенная
     смесь: близость к целевой энергии секции, тембральная схожесть с
     предыдущим архетипом, совместимость тональности, и штраф за
     повтор недавно использованных источников (anti-repeat decay).

  5. CLICK-FREE TRANSITIONS
     Границы фрагментов подрезаются к ближайшему zero-crossing,
     кроссфейд — equal-power (cos/sin), а не линейный.
"""

import os
import json
import time
import warnings
from pathlib import Path

import numpy as np
import librosa
from scipy.io.wavfile import write
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
SCAN_DIRS = [
    os.path.expanduser("~/Documents/Ableton"),
    os.path.expanduser("~/Music/Ableton"),
    os.path.expanduser("~/jam Project"),
    os.path.expanduser("~/Music"),
]
SKIP_DIRS = {"Factory Packs", "User Library", "Live Recordings"}
AUDIO_EXTS = {".wav", ".aiff", ".aif", ".flac", ".mp3", ".ogg", ".m4a"}
TARGET_SR = 22050
MAX_DURATION_SCAN = 60
MAX_DURATION_GENERATE = 180
MAX_FILES_FOR_GENOME = 300
GENOME_JSON = "genome.json"
N_GENOME_CLUSTERS = 12
N_FFT = 2048
HOP = 512
MAX_PITCH_SHIFT_SEMITONES = 4          # cap to avoid audible artifacts
BEAT_SEARCH_WINDOW_SEC = 12            # local window loaded to find a beat grid
ZERO_CROSS_SEARCH = 300                # samples to search for click-free cut

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

SECTION_PLAN = [
    # (name, fraction_of_total_duration, target_energy 0..1)
    ("intro", 0.14, 0.15),
    ("build", 0.18, 0.55),
    ("drop1", 0.20, 0.95),
    ("break", 0.14, 0.25),
    ("drop2", 0.24, 1.00),
    ("outro", 0.10, 0.12),
]


# ──────────────────────────────────────────────────────────────
# 1. SCANNER — recursive find + dedup
# ──────────────────────────────────────────────────────────────
def md5_file_fast(path):
    import subprocess
    try:
        result = subprocess.run(["md5", "-q", path], capture_output=True,
                                 text=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return None


def scan_audio_files(dirs):
    seen_hashes = {}
    files = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, fnames in os.walk(d):
            rel = os.path.relpath(root, d)
            if any(skip in rel for skip in SKIP_DIRS):
                continue
            for fname in fnames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in AUDIO_EXTS:
                    continue
                fpath = os.path.join(root, fname)
                h = md5_file_fast(fpath)
                if h and h not in seen_hashes:
                    seen_hashes[h] = fpath
                    files.append(fpath)
    return files


# ──────────────────────────────────────────────────────────────
# 2. FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────
def load_audio(path, sr=TARGET_SR, max_dur=MAX_DURATION_SCAN, offset=0.0):
    try:
        y, _ = librosa.load(path, sr=sr, mono=True, duration=max_dur, offset=offset)
        if len(y) < sr * 0.5:
            return None
        return y
    except Exception:
        return None


def estimate_key(chroma_vec):
    """Krumhansl-Schmuckler key estimate from a 12-bin chroma vector.
    Returns (pitch_class 0-11, mode 'major'/'minor', confidence -1..1)."""
    v = chroma_vec - np.mean(chroma_vec)
    if np.allclose(v, 0):
        return 0, "major", 0.0
    best = (-2.0, 0, "major")
    for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        p = profile - np.mean(profile)
        p_norm = np.linalg.norm(p)
        for shift in range(12):
            rotated = np.roll(p, shift)
            denom = (np.linalg.norm(v) * p_norm)
            corr = float(np.dot(v, rotated) / denom) if denom > 1e-9 else 0.0
            if corr > best[0]:
                best = (corr, shift, mode)
    return best[1], best[2], best[0]


def key_shift_semitones(src_key, tgt_key, cap=MAX_PITCH_SHIFT_SEMITONES):
    """Minimal chromatic distance (in semitones, signed) from src to tgt pitch class."""
    diff = (tgt_key - src_key) % 12
    if diff > 6:
        diff -= 12
    if abs(diff) > cap:
        return 0  # too far — leave uncorrected rather than mangle the audio
    return diff


def extract_features(y, sr=TARGET_SR):
    feats = {}

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    feats["stft_mean"] = np.mean(S_db, axis=1)
    feats["stft_std"] = np.std(S_db, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    feats["chroma_mean"] = np.mean(chroma, axis=1)
    feats["chroma_std"] = np.std(chroma, axis=1)
    key_idx, key_mode, key_conf = estimate_key(feats["chroma_mean"])
    feats["key_idx"] = key_idx
    feats["key_mode"] = key_mode
    feats["key_conf"] = key_conf

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP)
    feats["mfcc_mean"] = np.mean(mfcc, axis=1)
    feats["mfcc_std"] = np.std(mfcc, axis=1)

    sc = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    sr_feat = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    feats["spectral_centroid_mean"] = np.mean(sc)
    feats["spectral_centroid_std"] = np.std(sc)
    feats["spectral_rolloff_mean"] = np.mean(sr_feat)

    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    feats["rms_mean"] = np.mean(rms)
    feats["rms_std"] = np.std(rms)
    feats["rms_dynamic_range"] = np.max(rms) - np.min(rms)

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]
    feats["zcr_mean"] = np.mean(zcr)

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats["tempo"] = float(np.atleast_1d(tempo)[0])
    except Exception:
        feats["tempo"] = 120.0

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    feats["onset_density"] = float(np.sum(onset_env > np.mean(onset_env)) / len(onset_env))
    feats["onset_mean"] = float(np.mean(onset_env))

    harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harm, sr=sr)
    feats["tonnetz_mean"] = np.mean(tonnetz, axis=1)

    flat = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP)[0]
    feats["flatness_mean"] = np.mean(flat)

    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    feats["bandwidth_mean"] = np.mean(bw)

    vec = np.concatenate([
        feats["stft_mean"], feats["chroma_mean"], feats["mfcc_mean"],
        np.array([
            feats["spectral_centroid_mean"], feats["spectral_centroid_std"],
            feats["spectral_rolloff_mean"], feats["rms_mean"], feats["rms_std"],
            feats["rms_dynamic_range"], feats["zcr_mean"], feats["tempo"],
            feats["onset_density"], feats["onset_mean"],
            feats["flatness_mean"], feats["bandwidth_mean"],
        ]),
        feats["tonnetz_mean"],
    ])
    feats["vector"] = vec
    return feats


# ──────────────────────────────────────────────────────────────
# 3. GENOME — cluster into archetypes (+ dominant key vote)
# ──────────────────────────────────────────────────────────────
def build_genome(files, sr=TARGET_SR):
    if len(files) > MAX_FILES_FOR_GENOME:
        print(f"\n⚠️ {len(files)} files found, sampling {MAX_FILES_FOR_GENOME} for genome")
        indices = np.random.choice(len(files), MAX_FILES_FOR_GENOME, replace=False)
        files = [files[i] for i in sorted(indices)]
    print(f"\n🧬 GENOME SCANNER — {len(files)} unique files")
    print("=" * 60)

    genome = {"files": [], "features": [], "vectors": [],
              "clusters": None, "archetypes": None, "dominant_key": None}

    for i, fpath in enumerate(files):
        pct = (i + 1) / len(files) * 100
        print(f"\r  [{pct:5.1f}%] {os.path.basename(fpath)[:50]}", end="", flush=True)
        y = load_audio(fpath, sr)
        if y is None:
            continue
        feats = extract_features(y, sr)
        genome["files"].append(fpath)
        genome["features"].append(feats)
        genome["vectors"].append(feats["vector"])

    print(f"\n\n✅ Loaded {len(genome['files'])} files")
    if len(genome["vectors"]) < 2:
        print("⚠️ Too few files for clustering")
        return genome

    vectors = np.array(genome["vectors"])
    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(vectors)
    n_components = min(32, vectors_scaled.shape[0], vectors_scaled.shape[1])
    pca = PCA(n_components=n_components)
    vectors_pca = pca.fit_transform(vectors_scaled)

    n_clusters = min(N_GENOME_CLUSTERS, len(genome["files"]))
    Z = linkage(vectors_pca, method="ward")
    cluster_labels = fcluster(Z, n_clusters, criterion="maxclust")
    genome["clusters"] = cluster_labels.tolist()

    archetypes = {}
    for c in range(1, n_clusters + 1):
        idxs = [i for i, lbl in enumerate(cluster_labels) if lbl == c]
        if not idxs:
            continue
        mean_vec = np.mean([genome["vectors"][i] for i in idxs], axis=0)
        tempos = [genome["features"][i]["tempo"] for i in idxs]
        energies = [genome["features"][i]["rms_mean"] for i in idxs]
        bright = [genome["features"][i]["spectral_centroid_mean"] for i in idxs]
        chroma_avg = np.mean([genome["features"][i]["chroma_mean"] for i in idxs], axis=0)
        arch_key, arch_mode, arch_conf = estimate_key(chroma_avg)

        archetypes[c] = {
            "mean_vector": mean_vec.tolist(),
            "mean_tempo": float(np.mean(tempos)),
            "mean_energy": float(np.mean(energies)),
            "mean_brightness": float(np.mean(bright)),
            "key_idx": arch_key,
            "key_mode": arch_mode,
            "key_conf": arch_conf,
            "file_indices": idxs,
            "size": len(idxs),
        }
    genome["archetypes"] = archetypes

    # Dominant key of the whole genome, confidence-weighted vote
    votes = {}
    for f in genome["features"]:
        k = (f["key_idx"], f["key_mode"])
        votes[k] = votes.get(k, 0.0) + max(f["key_conf"], 0.0)
    if votes:
        dom = max(votes.items(), key=lambda kv: kv[1])[0]
        genome["dominant_key"] = {"key_idx": dom[0], "key_mode": dom[1]}

    print(f"\n📊 Genome summary:")
    print(f"   Archetypes: {n_clusters}")
    for c, arch in archetypes.items():
        print(f"   Cluster {c}: {arch['size']} files, tempo={arch['mean_tempo']:.0f}bpm, "
              f"energy={arch['mean_energy']:.3f}, key={arch['key_idx']}/{arch['key_mode']}")
    if genome["dominant_key"]:
        print(f"   Dominant key: pitch class {genome['dominant_key']['key_idx']} "
              f"({genome['dominant_key']['key_mode']})")
    return genome


# ──────────────────────────────────────────────────────────────
# 4. SMART EXTRACTION — beat-aligned, tempo/key-corrected fragments
# ──────────────────────────────────────────────────────────────
def _nearest_zero_crossing(y, idx, window=ZERO_CROSS_SEARCH):
    lo = max(0, idx - window)
    hi = min(len(y) - 1, idx + window)
    if hi <= lo:
        return idx
    seg = y[lo:hi]
    signs = np.sign(seg)
    crossings = np.where(np.diff(signs) != 0)[0]
    if len(crossings) == 0:
        return idx
    best = crossings[np.argmin(np.abs(crossings - (idx - lo)))]
    return lo + best


def load_beat_aligned_bar(fpath, sr, target_bar_samples, bars=1, bpm_hint=120.0):
    """
    Load a local window of a source file, find its real beat grid, and cut a
    bars*4-beat fragment strictly between two detected beats. Falls back to a
    plain random slice only if beat detection fails (short/percussive files).
    Returns raw (un-stretched) audio and whether it was beat-aligned.
    """
    try:
        dur = librosa.get_duration(path=fpath)
    except Exception:
        return None, False
    if dur < 1.0:
        return None, False

    window_sec = min(BEAT_SEARCH_WINDOW_SEC, dur)
    max_offset = max(0.0, dur - window_sec)
    offset = np.random.uniform(0, max_offset) if max_offset > 0 else 0.0

    try:
        y_local, _ = librosa.load(fpath, sr=sr, mono=True, offset=offset, duration=window_sec)
    except Exception:
        return None, False
    if y_local is None or len(y_local) < sr * 0.5:
        return None, False

    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y_local, sr=sr, start_bpm=bpm_hint)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    except Exception:
        beat_times = np.array([])

    beats_needed = bars * 4
    if len(beat_times) >= beats_needed + 1:
        max_start = len(beat_times) - beats_needed - 1
        i = np.random.randint(0, max_start + 1)
        start_s = beat_times[i]
        end_s = beat_times[i + beats_needed]
        start_i = int(start_s * sr)
        end_i = int(end_s * sr)
        start_i = _nearest_zero_crossing(y_local, start_i)
        end_i = _nearest_zero_crossing(y_local, end_i)
        if end_i - start_i > sr * 0.15:
            return y_local[start_i:end_i], True

    # Fallback: plain slice sized to the target bar duration
    n = min(target_bar_samples, len(y_local))
    start_i = np.random.randint(0, max(1, len(y_local) - n + 1))
    end_i = start_i + n
    return y_local[start_i:end_i], False


def time_stretch_to_length(chunk, target_len):
    if len(chunk) < 4 or target_len < 4:
        return np.pad(chunk, (0, max(0, target_len - len(chunk))))[:target_len]
    rate = len(chunk) / float(target_len)
    rate = float(np.clip(rate, 0.5, 2.0))  # avoid extreme stretch artifacts
    try:
        stretched = librosa.effects.time_stretch(chunk.astype(np.float32), rate=rate)
    except Exception:
        stretched = chunk
    if len(stretched) < target_len:
        stretched = np.pad(stretched, (0, target_len - len(stretched)))
    return stretched[:target_len]


def pitch_correct(chunk, sr, src_key, tgt_key):
    shift = key_shift_semitones(src_key, tgt_key)
    if shift == 0:
        return chunk
    try:
        return librosa.effects.pitch_shift(chunk.astype(np.float32), sr=sr, n_steps=shift)
    except Exception:
        return chunk


def equal_power_crossfade(a_tail, b_head, n):
    n = min(n, len(a_tail), len(b_head))
    if n <= 0:
        return np.array([])
    t = np.linspace(0, np.pi / 2, n)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    return a_tail[:n] * fade_out + b_head[:n] * fade_in


# ──────────────────────────────────────────────────────────────
# 5. STRUCTURE-AWARE TRACK GENERATOR
# ──────────────────────────────────────────────────────────────
def build_section_schedule(total_samples, sr, bar_samples):
    """Expand SECTION_PLAN into a per-bar list of (section_name, target_energy)."""
    total_bars = max(1, total_samples // bar_samples)
    schedule = []
    for name, frac, target_e in SECTION_PLAN:
        n_bars = max(1, int(round(total_bars * frac)))
        schedule.extend([(name, target_e)] * n_bars)
    # pad/trim to exact bar count
    if len(schedule) < total_bars:
        schedule.extend([schedule[-1]] * (total_bars - len(schedule)))
    return schedule[:total_bars]


def generate_track(genome, target_sec=MAX_DURATION_GENERATE, sr=TARGET_SR,
                    output="genome_track.wav", seed=None,
                    w_energy=1.0, w_timbre=0.8, w_key=0.6, w_repeat=1.2,
                    temperature=0.35):
    if seed is not None:
        np.random.seed(seed)

    print(f"\n🎸 GENERATING TRACK (smart mode) — {target_sec}s @ {sr}Hz")
    print("=" * 60)

    archetypes = genome["archetypes"]
    if not archetypes:
        print("❌ No archetypes to work with")
        return None

    arch_ids = list(archetypes.keys())
    arch_vecs = np.array([archetypes[a]["mean_vector"] for a in arch_ids])
    energies = np.array([archetypes[a]["mean_energy"] for a in arch_ids])
    e_min, e_max = float(energies.min()), float(energies.max() + 1e-9)
    energies_norm = (energies - e_min) / (e_max - e_min + 1e-9)

    # timbre similarity between archetypes (for transition smoothness)
    dists = np.linalg.norm(arch_vecs[:, None] - arch_vecs[None, :], axis=2)
    timbre_sim = np.exp(-dists / (dists.mean() + 1e-9))

    tgt_key_info = genome.get("dominant_key") or {"key_idx": 0, "key_mode": "major"}
    tgt_key = tgt_key_info["key_idx"]

    all_tempos = [a["mean_tempo"] for a in archetypes.values() if a["mean_tempo"] > 20]
    bpm = float(np.median(all_tempos)) if all_tempos else 120.0
    beat_samples = int(sr * 60.0 / bpm)
    bar_samples = beat_samples * 4
    crossfade_samples = int(0.12 * sr)
    total_samples = int(target_sec * sr)

    print(f"   Tempo: {bpm:.0f} BPM   Target key: pc={tgt_key} ({tgt_key_info['key_mode']})")
    print(f"   Bar length: {bar_samples / sr:.2f}s")

    schedule = build_section_schedule(total_samples, sr, bar_samples)
    print(f"   Sections: {[(n, schedule.count((n, e))) for n, e in dict.fromkeys(schedule)]}")

    output_audio = np.zeros(total_samples + bar_samples)  # small overrun buffer
    prev_chunk = None
    current_pos = 0
    current_idx = int(np.argmax(energies))  # start from a mid/high energy archetype
    recent_files = {}   # file_idx -> bars since last used (repetition memory)
    REPEAT_MEMORY = 8

    for bar_i, (section_name, target_e) in enumerate(schedule):
        # decay repetition memory
        for k in list(recent_files):
            recent_files[k] += 1
            if recent_files[k] > REPEAT_MEMORY:
                del recent_files[k]

        # score every archetype for this bar's slot
        scores = np.zeros(len(arch_ids))
        for j in range(len(arch_ids)):
            e_score = -abs(energies_norm[j] - target_e)
            t_score = timbre_sim[current_idx, j]
            key_dist = abs(key_shift_semitones(archetypes[arch_ids[j]]["key_idx"], tgt_key))
            k_score = -key_dist / 6.0
            scores[j] = w_energy * e_score + w_timbre * t_score + w_key * k_score

        probs = np.exp(scores / max(temperature, 1e-3))
        probs /= probs.sum()
        current_idx = int(np.random.choice(len(arch_ids), p=probs))
        arch_id = arch_ids[current_idx]
        arch = archetypes[arch_id]

        # pick a source file, penalizing recently used ones
        candidates = arch["file_indices"]
        weights = np.array([1.0 / (1 + recent_files.get(fi, REPEAT_MEMORY)) for fi in candidates])
        weights /= weights.sum()
        file_idx = int(np.random.choice(candidates, p=weights))
        recent_files[file_idx] = 0
        fpath = genome["files"][file_idx]

        raw_chunk, aligned = load_beat_aligned_bar(fpath, sr, bar_samples, bars=1, bpm_hint=bpm)
        if raw_chunk is None or len(raw_chunk) < sr * 0.1:
            continue

        chunk = time_stretch_to_length(raw_chunk, bar_samples)

        src_key = genome["features"][file_idx]["key_idx"] if file_idx < len(genome["features"]) else tgt_key
        chunk = pitch_correct(chunk, sr, src_key, tgt_key)

        rms = np.sqrt(np.mean(chunk ** 2)) + 1e-10
        chunk = chunk / rms
        target_rms = 0.05 + 0.25 * target_e
        chunk = chunk * target_rms

        env = np.ones(len(chunk))
        attack = max(1, int(0.015 * len(chunk)))
        release = max(1, int(0.04 * len(chunk)))
        env[:attack] = np.linspace(0, 1, attack)
        env[-release:] = np.linspace(1, 0, release)
        chunk *= env

        if prev_chunk is not None and crossfade_samples > 0:
            faded = equal_power_crossfade(prev_chunk[-crossfade_samples:], chunk, crossfade_samples)
            write_start = current_pos - len(faded)
            if write_start >= 0:
                output_audio[write_start:write_start + len(faded)] = faded
            rest = chunk[len(faded):]
            end = min(current_pos + len(rest), len(output_audio))
            n = end - current_pos
            if n > 0:
                output_audio[current_pos:end] += rest[:n]
        else:
            end = min(current_pos + len(chunk), len(output_audio))
            n = end - current_pos
            output_audio[current_pos:end] += chunk[:n]

        prev_chunk = chunk.copy()
        current_pos += bar_samples

        if bar_i % 8 == 0:
            print(f"  [{section_name:6s} bar {bar_i}] {current_pos / sr:.1f}s  "
                  f"arch={arch_id} aligned={aligned} key_shift={key_shift_semitones(src_key, tgt_key):+d}")

        if current_pos >= total_samples:
            break

    output_audio = output_audio[:total_samples]

    print("🎛️ Post-processing...")
    peak = np.max(np.abs(output_audio))
    if peak > 1e-6:
        output_audio = output_audio / peak * 0.85
    output_audio = np.tanh(1.4 * output_audio)

    fade_len = int(0.3 * sr)
    output_audio[:fade_len] *= np.linspace(0, 1, fade_len)
    output_audio[-fade_len:] *= np.linspace(1, 0, fade_len)

    audio_16bit = (output_audio * 32767).astype(np.int16)
    write(output, sr, audio_16bit)
    print(f"✅ Saved: {output} ({target_sec}s)")

    with open(output.replace(".wav", "_info.json"), "w") as f:
        json.dump({
            "duration_sec": target_sec, "bpm": bpm,
            "target_key": tgt_key_info, "archetypes_used": len(archetypes),
            "genome_files": len(genome["files"]),
        }, f, indent=2)

    return output_audio


# ──────────────────────────────────────────────────────────────
# 6. FULL PIPELINE
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# Load genome from cache if available
# ──────────────────────────────────────────────────────────────
def load_genome_from_cache(path=GENOME_JSON):
    """Load cached genome metadata. Returns None when no usable cache exists."""
    try:
        with open(path, "r") as f:
            meta = json.load(f)
        if not meta.get("files") or not meta.get("archetypes"):
            return None
        # Older genome.json files may not contain the archetype vectors needed
        # by the generator. Treat them as stale and rebuild the genome once.
        for arch in meta["archetypes"].values():
            if "mean_vector" not in arch or "file_indices" not in arch:
                print("⚠️ Cached genome is stale/incomplete; rebuilding genome")
                return None
        genome = {
            "files": [],
            "features": [],
            "vectors": [],
            "clusters": meta.get("clusters"),
            "archetypes": {int(k): v for k, v in meta["archetypes"].items()},
            "dominant_key": meta.get("dominant_key"),
        }
        # The cached metadata currently stores basenames, so resolve them against
        # the configured scan directories. Files that disappeared are ignored.
        by_name = {}
        for d in SCAN_DIRS:
            if not os.path.isdir(d):
                continue
            for root, _, fnames in os.walk(d):
                rel = os.path.relpath(root, d)
                if any(skip in rel for skip in SKIP_DIRS):
                    continue
                for fname in fnames:
                    by_name.setdefault(fname, os.path.join(root, fname))

        cached_to_current = {}
        for i, name in enumerate(meta["files"]):
            if name in by_name:
                cached_to_current[i] = len(genome["files"])
                genome["files"].append(by_name[name])
                genome["features"].append({"key_idx": genome["dominant_key"]["key_idx"] if genome["dominant_key"] else 0})

        # Archetype file_indices must point into the compacted current file list.
        for arch in genome["archetypes"].values():
            arch["file_indices"] = [cached_to_current[i] for i in arch.get("file_indices", []) if i in cached_to_current]
            arch["size"] = len(arch["file_indices"])

        genome["archetypes"] = {k: v for k, v in genome["archetypes"].items() if v["file_indices"]}
        if not genome["archetypes"]:
            return None
        print(f"🧬 Loaded cached genome: {len(genome['files'])} files, {len(genome['archetypes'])} archetypes")
        return genome
    except Exception as exc:
        print(f"⚠️ Could not load cached genome: {exc}")
        return None


def run_full_pipeline(scan_dirs=SCAN_DIRS, target_sec=MAX_DURATION_GENERATE,
                       output="genome_track.wav", rescan=False):
    t0 = time.time()

    genome = None if rescan else load_genome_from_cache()
    if genome is None:
        print("🔍 Scanning for audio files...")
        files = scan_audio_files(scan_dirs)
        print(f"   Found {len(files)} unique files (after dedup)")
        if not files:
            print("❌ No audio files found!")
            return
        genome = build_genome(files)

        genome_meta = {
            "total_files": len(genome["files"]),
            "files": [os.path.basename(f) for f in genome["files"]],
            "clusters": genome["clusters"],
            "dominant_key": genome["dominant_key"],
            "archetypes": {
                str(k): {kk: vv for kk, vv in v.items() if kk != "mean_vector"} | {"mean_vector": v["mean_vector"]}
                for k, v in (genome["archetypes"] or {}).items()
            },
        }
        with open(GENOME_JSON, "w") as f:
            json.dump(genome_meta, f, indent=2)
        print(f"\n📄 Genome metadata saved: {GENOME_JSON}")
    else:
        print("⚡ Using cached genome, skipping full scan")

    generate_track(genome, target_sec=target_sec, output=output)
    print(f"\n⏱️ Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GENOME v2 — smart spectral generator")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--output", type=str, default="genome_track.wav")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rescan", action="store_true", help="force a full audio scan and rebuild the genome")
    args = parser.parse_args()

    if args.scan_only:
        files = scan_audio_files(SCAN_DIRS)
        genome = build_genome(files)
        print(f"\nDone. {len(genome['files'])} files scanned.")
    else:
        run_full_pipeline(target_sec=args.duration, output=args.output, rescan=args.rescan)
