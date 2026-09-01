"""
GRANULAR MUSIC FIELD v3
2389 треков → иерархия гранул → multi-stream Navigator → granular engine → spectral field → multi-scale critic
"""
import os, json, time, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import librosa
import soundfile as sf
from scipy.io.wavfile import write
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
warnings.filterwarnings("ignore")

SCAN_DIRS = [
    os.path.expanduser("~/Documents/Ableton"),
    os.path.expanduser("~/Music/Ableton"),
    os.path.expanduser("~/jam Project"),
    os.path.expanduser("~/Music"),
]
VISUAL_SCAN_DIRS = SCAN_DIRS + [
    os.path.expanduser("~/Pictures"),
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~/Downloads"),
]
SKIP_DIRS = {"Factory Packs", "User Library", "Live Recordings"}
AUDIO_EXTS = {".wav", ".aiff", ".aif", ".flac", ".mp3", ".ogg", ".m4a"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

VIS_MICRO_SIZE = 16
VIS_MESO_SIZE = 32
VIS_MACRO_SIZE = 64
VIS_FEAT_DIM = 22
VIS_CANVAS_SIZE = 512
VIS_N_CLUSTERS = 512
SR = 22050
N_FFT = 1024
HOP = 256

MICRO_FRAMES = 5
MESO_FRAMES = 26
MACRO_FRAMES = 259

FEAT_DIM = 22
N_CLUSTERS = 1024
STATE_DIM = 48
CONTEXT_LEN = 12
HIDDEN_DIM = 192
N_HEADS = 4
N_LAYERS = 3
BATCH_SIZE = 128
TRAIN_STEPS = 8000
LR = 3e-4
MAX_MICRO_PER_FILE = 300
MAX_MESO_PER_FILE = 100
MAX_MACRO_PER_FILE = 30

N_STREAMS = 6
N_ATTRACTORS = 6  # one per stream — learned global direction for long-range coherence
STREAM_NAMES = ["sub", "drums", "harmonic", "texture", "noise", "air"]
STREAM_BANDS = [
    (20, 120),      # sub — deep bass
    (120, 500),     # drums — punch, body
    (500, 2000),    # harmonic — fundamental harmonics
    (2000, 4000),   # texture — presence
    (4000, 8000),   # noise — brightness
    (8000, 11000),  # air — top shimmer (limited by 22050 SR)
]
# Reference balance: sub 77%, drums 46%, harm 20%, text 10%, noise 6%, air 2%
# Generated harm/text were +8-9 dB too hot vs reference → harsh
STREAM_WEIGHTS = [3.0, 2.0, 0.5, 0.25, 0.15, 0.05]
# Deterministic pan per stream — model can't learn spatial from mono features, so we assign it directly
# Tight fog: small offsets, not binary L/R. Sub/drums mono center, upper streams fill the field subtly
STREAM_PANS = [0.0, 0.0, -0.5, 0.5, -0.2, 0.2]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FILELIST_CACHE = "genome_filelist.json"
POOL_CACHE = "granular_pool_v2.npz"
POOL_CACHE_LIGHT = "granular_pool_lite.npz"
MODEL_CACHE = "granular_navigator_v2.pt"
MODEL_MULTI_CACHE = "granular_multi_v1.pt"
VIS_POOL_CACHE = "visual_pool.npz"
VIS_MODEL_CACHE = "granular_multi_visual_v1.pt"
OUT = "granular_output"
os.makedirs(OUT, exist_ok=True)

def md5_fast(p):
    import subprocess
    try: return subprocess.run(["md5","-q",p],capture_output=True,text=True,timeout=10).stdout.strip()
    except: return None

def scan_audio(dirs):
    if os.path.exists(FILELIST_CACHE):
        with open(FILELIST_CACHE) as f: cached = json.load(f)
        files = [p for p in cached if os.path.exists(p)]
        if files: print(f"📦 Filelist cache: {len(files)}"); return files
    seen = {}
    for d in dirs:
        if not os.path.isdir(d): continue
        for root, _, fnames in os.walk(d):
            rel = os.path.relpath(root, d)
            if any(s in rel for s in SKIP_DIRS): continue
            for fn in fnames:
                if os.path.splitext(fn)[1].lower() not in AUDIO_EXTS: continue
                fp = os.path.join(root, fn)
                h = md5_fast(fp)
                if h and h not in seen: seen[h] = fp
    files = list(seen.values())
    with open(FILELIST_CACHE, "w") as f: json.dump(files, f)
    return files


def scan_images(dirs, max_files=2000):
    seen = {}
    for d in dirs:
        if not os.path.isdir(d): continue
        for root, _, fnames in os.walk(d):
            rel = os.path.relpath(root, d)
            if any(s in rel for s in SKIP_DIRS): continue
            for fn in fnames:
                if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS: continue
                fp = os.path.join(root, fn)
                try:
                    size = os.path.getsize(fp)
                except OSError:
                    continue
                if size < 2000 or size > 15 * 1024 * 1024:
                    continue
                h = md5_fast(fp)
                if h and h not in seen:
                    seen[h] = fp
                if len(seen) >= max_files:
                    return list(seen.values())
    return list(seen.values())


def extract_feat_from_stft(S_slice):
    """Feature extraction from LINEAR magnitude STFT slice [n_fft_half, n_frames]."""
    S = S_slice.astype(np.float64) + 1e-10
    n_freqs, n_frames = S.shape
    feat = []

    mean_spectrum = S.mean(axis=1)

    # Spectral stats
    feat.append(float(np.mean(mean_spectrum)))
    feat.append(float(np.std(mean_spectrum)))

    # Spectral centroid (linear magnitudes as weights)
    freqs = np.linspace(0, 1, n_freqs)
    total = np.sum(mean_spectrum)
    feat.append(float(np.sum(freqs * mean_spectrum) / (total + 1e-10)))

    # Spectral bandwidth
    centroid = feat[-1]
    feat.append(float(np.sqrt(np.sum((freqs - centroid)**2 * mean_spectrum) / (total + 1e-10))))

    # Spectral flatness (geometric mean / arithmetic mean — valid for positive values)
    log_spectrum = np.log(mean_spectrum)
    feat.append(float(np.exp(np.mean(log_spectrum)) / (np.mean(mean_spectrum) + 1e-10)))

    # Spectral rolloff
    cumsum = np.cumsum(mean_spectrum)
    rolloff = np.searchsorted(cumsum, 0.85 * cumsum[-1]) / n_freqs
    feat.append(float(rolloff))

    # Band energies (6 mel-like bins from linear spectrum)
    mel_bins = 6
    for b in range(mel_bins):
        lo = int(n_freqs * b / mel_bins)
        hi = int(n_freqs * (b + 1) / mel_bins)
        feat.append(float(np.mean(mean_spectrum[lo:hi])))

    # Temporal stats (energy over time frames)
    frame_energy = S.mean(axis=0)
    feat.append(float(np.mean(frame_energy)))
    feat.append(float(np.std(frame_energy)))
    feat.append(float(np.max(frame_energy) - np.min(frame_energy)))
    feat.append(float(np.std(np.diff(frame_energy)) + 1e-10))

    # Spectral flux
    flux = np.diff(S, axis=1)
    feat.append(float(np.mean(np.maximum(flux, 0))))
    feat.append(float(np.mean(np.abs(flux))))

    # Band energy ratios
    lo_f = int(n_freqs * 0.1); mid_f = int(n_freqs * 0.4)
    e_lo = np.mean(mean_spectrum[:lo_f])
    e_mid = np.mean(mean_spectrum[lo_f:mid_f])
    e_hi = np.mean(mean_spectrum[mid_f:])
    total_e = e_lo + e_mid + e_hi + 1e-10
    feat.append(float(e_lo / total_e))
    feat.append(float(e_mid / total_e))
    feat.append(float(e_hi / total_e))

    # Low freq ratio
    feat.append(float(np.sum(mean_spectrum[:lo_f]) / (total + 1e-10)))

    return np.array(feat[:FEAT_DIM], dtype=np.float32)


def extract_all(files, save_audio=True):
    print(f"\n🧬 Extracting grains from {len(files)} files...")
    t0 = time.time()

    micro_feats, micro_audio = [], []
    meso_feats, meso_audio = [], []
    macro_feats, macro_audio = [], []
    micro_sources, meso_sources, macro_sources = [], [], []
    trajectories = []
    errors = 0

    for i, fp in enumerate(files):
        if (i+1) % 100 == 0 or i+1 == len(files):
            e = time.time()-t0; r = (i+1)/(e+0.001)
            print(f"\r  [{i+1}/{len(files)}] μ={len(micro_feats)} σ={len(meso_feats)} Ω={len(macro_feats)} err={errors} {r:.1f}f/s  ", end="", flush=True)
        try:
            y, fsr = sf.read(fp, dtype="float32", always_2d=False)
            if y.ndim > 1: y = np.mean(y, axis=1)
            if fsr != SR: y = librosa.resample(y, orig_sr=fsr, target_sr=SR)
            if len(y) < SR * 2: errors += 1; continue
            if len(y) > SR * 90: y = y[:SR * 90]

            S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
            n_frames = S.shape[1]
            traj = []

            # Micro
            hop_m = max(1, MICRO_FRAMES // 2)
            cnt = 0
            for s in range(0, n_frames - MICRO_FRAMES + 1, hop_m):
                if cnt >= MAX_MICRO_PER_FILE: break
                sl = S[:, s:s+MICRO_FRAMES]
                if np.max(sl) < 0.001: continue
                feat = extract_feat_from_stft(sl)
                if np.any(np.isnan(feat)) or np.all(feat == 0): continue
                smp_start = s * HOP
                smp_end = min(smp_start + MICRO_FRAMES * HOP, len(y))
                chunk = y[smp_start:smp_end]
                if len(chunk) < MICRO_FRAMES * HOP:
                    chunk = np.pad(chunk, (0, MICRO_FRAMES * HOP - len(chunk)))
                micro_feats.append(feat)
                if save_audio: micro_audio.append(chunk)
                micro_sources.append((fp, int(smp_start), MICRO_FRAMES * HOP))
                traj.append((0, len(micro_feats)-1))
                cnt += 1

            # Meso
            hop_s = max(1, MESO_FRAMES // 2)
            cnt = 0
            for s in range(0, n_frames - MESO_FRAMES + 1, hop_s):
                if cnt >= MAX_MESO_PER_FILE: break
                sl = S[:, s:s+MESO_FRAMES]
                if np.max(sl) < 0.001: continue
                feat = extract_feat_from_stft(sl)
                if np.any(np.isnan(feat)) or np.all(feat == 0): continue
                smp_start = s * HOP
                smp_end = min(smp_start + MESO_FRAMES * HOP, len(y))
                chunk = y[smp_start:smp_end]
                if len(chunk) < MESO_FRAMES * HOP:
                    chunk = np.pad(chunk, (0, MESO_FRAMES * HOP - len(chunk)))
                meso_feats.append(feat)
                if save_audio: meso_audio.append(chunk)
                meso_sources.append((fp, int(smp_start), MESO_FRAMES * HOP))
                traj.append((1, len(meso_feats)-1))
                cnt += 1

            # Macro
            hop_l = max(1, MACRO_FRAMES // 2)
            cnt = 0
            for s in range(0, n_frames - MACRO_FRAMES + 1, hop_l):
                if cnt >= MAX_MACRO_PER_FILE: break
                sl = S[:, s:s+MACRO_FRAMES]
                if np.max(sl) < 0.001: continue
                feat = extract_feat_from_stft(sl)
                if np.any(np.isnan(feat)) or np.all(feat == 0): continue
                smp_start = s * HOP
                smp_end = min(smp_start + MACRO_FRAMES * HOP, len(y))
                chunk = y[smp_start:smp_end]
                if len(chunk) < MACRO_FRAMES * HOP:
                    chunk = np.pad(chunk, (0, MACRO_FRAMES * HOP - len(chunk)))
                macro_feats.append(feat)
                if save_audio: macro_audio.append(chunk)
                macro_sources.append((fp, int(smp_start), MACRO_FRAMES * HOP))
                traj.append((2, len(macro_feats)-1))
                cnt += 1

            if traj: trajectories.append(traj)
        except Exception as e:
            errors += 1

    elapsed = time.time()-t0
    print(f"\n  ✅ {elapsed:.1f}s — micro={len(micro_feats)} meso={len(meso_feats)} macro={len(macro_feats)}")

    result = {
        "micro_feats": np.array(micro_feats, dtype=np.float32) if micro_feats else np.zeros((0,FEAT_DIM),np.float32),
        "meso_feats": np.array(meso_feats, dtype=np.float32) if meso_feats else np.zeros((0,FEAT_DIM),np.float32),
        "macro_feats": np.array(macro_feats, dtype=np.float32) if macro_feats else np.zeros((0,FEAT_DIM),np.float32),
        "micro_sources": micro_sources,
        "meso_sources": meso_sources,
        "macro_sources": macro_sources,
        "trajectories": trajectories,
    }
    if save_audio:
        result["micro_audio"] = np.array(micro_audio, dtype=np.float32) if micro_audio else np.zeros((0,MICRO_FRAMES*HOP),np.float32)
        result["meso_audio"] = np.array(meso_audio, dtype=np.float32) if meso_audio else np.zeros((0,MESO_FRAMES*HOP),np.float32)
        result["macro_audio"] = np.array(macro_audio, dtype=np.float32) if macro_audio else np.zeros((0,MACRO_FRAMES*HOP),np.float32)
    return result


def build_clusters(pool):
    print("🔬 Clustering grains...")
    all_f = np.concatenate([pool["micro_feats"], pool["meso_feats"], pool["macro_feats"]])
    if len(all_f) > 80000:
        idx = np.random.choice(len(all_f), 80000, replace=False)
        fit_f = all_f[idx]
    else:
        fit_f = all_f
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=1024, n_init=3, random_state=42)
    kmeans.fit(fit_f)
    all_labels = kmeans.predict(all_f)
    n_m = len(pool["micro_feats"])
    n_s = len(pool["meso_feats"])
    clusters = {
        "micro": all_labels[:n_m],
        "meso": all_labels[n_m:n_m+n_s],
        "macro": all_labels[n_m+n_s:],
    }
    print(f"  ✅ {N_CLUSTERS} clusters")
    return clusters


# ══════════════════════════════════════════════════════════════
# NAVIGATOR
# ══════════════════════════════════════════════════════════════
class Navigator(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, state_dim=STATE_DIM, hidden=HIDDEN_DIM,
                 ctx=CONTEXT_LEN, n_clusters=N_CLUSTERS):
        super().__init__()
        self.feat_enc = nn.Linear(feat_dim, state_dim)
        self.pos = nn.Parameter(torch.randn(1, ctx, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=N_HEADS,
            dim_feedforward=hidden*2, dropout=0.1, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.proj = nn.Linear(state_dim, hidden)
        self.cluster_head = nn.Linear(hidden, n_clusters)
        self.level_head = nn.Linear(hidden, 3)
        self.params_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 8), nn.Tanh())
        self.cond_proj = nn.Linear(4, hidden)

    def forward(self, states, cond=None):
        B, K, _ = states.shape
        z = self.proj(self.feat_enc(states)) + self.pos[:, :K, :]
        z = self.transformer(z)[:, -1, :]
        if cond is not None:
            z = z + self.cond_proj(cond)
        return self.cluster_head(z), self.level_head(z), self.params_head(z)

    @torch.no_grad()
    def step(self, states, temp=0.8, cond=None):
        self.eval()
        if states.dim() == 2: states = states.unsqueeze(0)
        cl, lv, pr = self.forward(states, cond=cond)
        c = F.softmax(cl.squeeze(0)/temp, dim=-1)
        l = F.softmax(lv.squeeze(0)/temp, dim=-1)
        cluster = torch.multinomial(c, 1).item()
        level = torch.multinomial(l, 1).item()
        p = pr.squeeze(0).cpu().numpy()
        return cluster, level, p


# ══════════════════════════════════════════════════════════════
# MULTI-STREAM NAVIGATOR
# ══════════════════════════════════════════════════════════════
class MultiNavigator(nn.Module):
    """6 independent streams: sub, drums, harmonic, texture, noise, air.
    Shared backbone transformer + per-stream audio heads + visual head."""

    def __init__(self, feat_dim=FEAT_DIM, state_dim=STATE_DIM, hidden=HIDDEN_DIM,
                 ctx=CONTEXT_LEN, n_clusters=N_CLUSTERS, n_streams=N_STREAMS,
                 vis_feat_dim=VIS_FEAT_DIM, vis_n_clusters=VIS_N_CLUSTERS):
        super().__init__()
        self.n_streams = n_streams
        self.feat_enc = nn.Linear(feat_dim, state_dim)
        self.pos = nn.Parameter(torch.randn(1, ctx, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=N_HEADS,
            dim_feedforward=hidden*2, dropout=0.1, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.proj = nn.Linear(state_dim, hidden)

        self.stream_cluster = nn.Linear(hidden, n_clusters)
        self.stream_level = nn.Linear(hidden, 3)
        self.stream_params = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 8), nn.Tanh())
        self.stream_density = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1), nn.Sigmoid())
        self.stream_pan = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1), nn.Tanh())

        self.cross_stream = nn.MultiheadAttention(hidden, num_heads=2, batch_first=True)
        self.stream_embed = nn.Parameter(torch.randn(1, n_streams, hidden) * 0.05)
        self.cond_proj = nn.Linear(4, hidden)

        # ── Visual head: same backbone, separate visual prediction ──────
        self.v_feat_enc = nn.Linear(vis_feat_dim, state_dim)
        self.v_cross = nn.MultiheadAttention(hidden, num_heads=2, batch_first=True)
        self.v_stream_embed = nn.Parameter(torch.randn(1, 1, hidden) * 0.05)
        self.v_cluster_head = nn.Linear(hidden, vis_n_clusters)
        self.v_blend_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 4), nn.Tanh())

        # ── Attractor field: learned global state per stream ──────────
        # Unlike ctx_z (sliding window, local), attractors persist across the
        # full generation. They pull the sound toward a learned direction —
        # the z0 hypothesis: a global hint layered on top of local coherence.
        self.attractor_embed = nn.Parameter(torch.randn(n_streams, hidden) * 0.05)
        self.attractor_proj = nn.Linear(hidden, hidden)
        self.attractor_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.attractor_update = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.Tanh())
        # running attractor state per stream: updated during generation, frozen during training
        self.register_buffer("attractor_state", torch.zeros(n_streams, hidden))

    def forward(self, states, stream_idx=0, cond=None, vis_states=None):
        B, K, _ = states.shape
        z = self.proj(self.feat_enc(states)) + self.pos[:, :K, :]
        z = self.transformer(z)
        ctx_z = z[:, -1, :]

        # stream_idx can be int, scalar tensor, or batch tensor [B]
        if isinstance(stream_idx, int):
            se = self.stream_embed[:, stream_idx:stream_idx+1].expand(B, -1, -1)
            as_ = self.attractor_state[stream_idx:stream_idx+1].expand(B, -1, -1)
        elif stream_idx.dim() == 0:
            se = self.stream_embed[:, stream_idx:stream_idx+1].expand(B, -1, -1)
            as_ = self.attractor_state[stream_idx:stream_idx+1].expand(B, -1, -1)
        else:
            se = self.stream_embed.squeeze(0)[stream_idx].unsqueeze(1)
            as_ = self.attractor_state[stream_idx].unsqueeze(1)

        crossed, _ = self.cross_stream(se, z, z)

        # attractor pull: how much the global state influences this step
        ag = self.attractor_gate(torch.cat([ctx_z, as_.squeeze(1)], dim=-1))
        a_pull = ag * self.attractor_proj(as_.squeeze(1))

        h = (ctx_z + crossed.squeeze(1) + a_pull) / 3

        if cond is not None:
            h = h + self.cond_proj(cond)

        # ── Visual forward: separate pathway from backbone ──────────
        v_h = None
        if vis_states is not None:
            vK = vis_states.shape[1]
            vz = self.proj(self.v_feat_enc(vis_states)) + self.pos[:, :vK, :]
            vz = self.transformer(vz)
            v_ctx = vz[:, -1, :]
            vse = self.v_stream_embed.expand(B, -1, -1)
            vcrossed, _ = self.v_cross(vse, vz, vz)
            v_h = (v_ctx + vcrossed.squeeze(1) + a_pull) / 3

        audio_out = (self.stream_cluster(h), self.stream_level(h),
                     self.stream_params(h), self.stream_density(h), self.stream_pan(h))
        if v_h is not None:
            vis_out = (self.v_cluster_head(v_h), self.v_blend_head(v_h))
            return audio_out, vis_out
        return audio_out, None

    def update_attractors(self, states, stream_idx, temp=0.8):
        """Update attractor states from the current context — called during generation."""
        with torch.no_grad():
            if states.dim() == 2: states = states.unsqueeze(0)
            z = self.proj(self.feat_enc(states))
            z = self.transformer(z)
            ctx_z = z[:, -1, :]  # [B, hidden]
            if isinstance(stream_idx, int):
                as_ = self.attractor_state[stream_idx:stream_idx+1].expand_as(ctx_z)  # [B, hidden]
                new_state = self.attractor_update(torch.cat([ctx_z, as_], dim=-1))
                self.attractor_state[stream_idx] = 0.9 * self.attractor_state[stream_idx] + 0.1 * new_state.mean(0)
            else:
                idx = stream_idx if isinstance(stream_idx, list) else stream_idx.tolist()
                n = len(idx)
                as_ = self.attractor_state[idx]  # [n, hidden]
                ctx_z_exp = ctx_z.expand(n, -1)  # [n, hidden] — broadcast single context to all streams
                new_state = self.attractor_update(torch.cat([ctx_z_exp, as_], dim=-1))
                for i, s in enumerate(idx):
                    self.attractor_state[s] = 0.9 * self.attractor_state[s] + 0.1 * new_state[i]

    @torch.no_grad()
    def step(self, states, stream_idx=0, temp=0.8, cond=None, vis_states=None):
        self.eval()
        if states.dim() == 2: states = states.unsqueeze(0)
        audio_out, vis_out = self.forward(states, stream_idx=stream_idx, cond=cond, vis_states=vis_states)
        cl, lv, pr, dn, pn = audio_out
        c = F.softmax(cl.squeeze(0) / temp, dim=-1)
        l = F.softmax(lv.squeeze(0) / temp, dim=-1)
        result = {
            "cluster": torch.multinomial(c, 1).item(),
            "level": torch.multinomial(l, 1).item(),
            "params": pr.squeeze(0).cpu().numpy(),
            "density": float(dn.squeeze(0).cpu()),
            "pan": float(pn.squeeze(0).cpu()),
        }
        if vis_out is not None:
            vcl, vbl = vis_out
            vc = F.softmax(vcl.squeeze(0) / temp, dim=-1)
            result["v_cluster"] = torch.multinomial(vc, 1).item()
            result["v_blend"] = vbl.squeeze(0).cpu().numpy()
        return result


# ══════════════════════════════════════════════════════════════
# SPECTRAL FIELD — attraction/repulsion between streams
# ══════════════════════════════════════════════════════════════
class SpectralField:
    """Tracks energy in frequency bands per stream.
    Streams in underused bands get attraction, crowded bands get repulsion."""

    def __init__(self, sr=SR, n_bands=6):
        from scipy.signal import butter
        self.sr = sr
        self.bands = STREAM_BANDS
        self.band_filters = []
        for lo, hi in self.bands:
            sos = butter(4, [lo/(sr/2), min(hi/(sr/2), 0.99)], btype='band', output='sos')
            self.band_filters.append(sos)
        self.energy = np.zeros(n_bands, dtype=np.float64)
        self.target_energy = 1.0 / n_bands

    def analyze(self, audio):
        from scipy.signal import sosfilt
        for i, sos in enumerate(self.band_filters):
            mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio
            filtered = sosfilt(sos, mono)
            self.energy[i] = np.sqrt(np.mean(filtered**2) + 1e-10)

    def get_bias(self, stream_idx):
        if stream_idx >= len(self.bands): return 0.0
        ratio = self.energy[stream_idx] / (np.mean(self.energy) + 1e-10)
        if ratio < 0.5:
            return 0.15
        elif ratio > 2.0:
            return -0.1
        return 0.0

    def reset(self):
        self.energy[:] = 0


# ══════════════════════════════════════════════════════════════
# MULTI-SCALE CRITIC
# ══════════════════════════════════════════════════════════════
class MultiScaleCritic:
    """3 scales: short STFT (transients), long STFT (form), onset analysis (rhythm)."""

    def __init__(self, sr=SR):
        self.sr = sr

    def score(self, audio):
        mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio
        s1, d1 = self._score_short(mono)
        s2, d2 = self._score_long(mono)
        s3, d3 = self._score_rhythm(mono)
        total = 0.35 * s1 + 0.35 * s2 + 0.30 * s3
        return total, {**d1, **d2, **d3}

    def _score_short(self, mono):
        chunk = mono[:N_FFT] if len(mono) >= N_FFT else mono
        S = np.abs(np.fft.rfft(chunk.astype(np.float64)))
        freqs = np.linspace(0, self.sr/2, len(S))
        centroid = np.sum(freqs * S) / (np.sum(S) + 1e-10)
        sc = float(np.clip(1.0 - abs(centroid - 2000) / 5000, 0, 1))
        flatness = float(np.exp(np.mean(np.log(S + 1e-10))) / (np.mean(S) + 1e-10))
        fl = float(np.clip(1.0 - flatness, 0, 1))
        return (sc + fl) / 2, {"short_centroid": sc, "short_flatness": fl}

    def _score_long(self, mono):
        chunk = mono[:min(len(mono), self.sr * 4)]
        rms = np.sqrt(np.mean(chunk**2) + 1e-10)
        eng = float(np.clip(rms / 0.1, 0, 1))
        block = self.sr // 2
        blocks = [np.sqrt(np.mean(chunk[i:i+block]**2) + 1e-10)
                  for i in range(0, len(chunk) - block, block)]
        if len(blocks) > 1:
            std = np.std(blocks) / (np.mean(blocks) + 1e-10)
            dyn = float(np.clip(1.0 - std, 0, 1))
        else:
            dyn = 0.5
        return (eng + dyn) / 2, {"long_energy": eng, "long_dynamics": dyn}

    def _score_rhythm(self, mono):
        from scipy.signal import find_peaks
        if len(mono) < self.sr: return 0.5, {"rhythm": 0.5}
        envelope = np.abs(mono)
        block = self.sr // 10
        env_blocks = [np.mean(envelope[i:i+block]**2) for i in range(0, len(envelope)-block, block)]
        env_arr = np.array(env_blocks)
        if np.max(env_arr) < 1e-8: return 0.3, {"rhythm": 0.3}
        env_norm = env_arr / (np.max(env_arr) + 1e-10)
        peaks, props = find_peaks(env_norm, height=0.3, distance=2)
        if len(peaks) > 2:
            isis = np.diff(peaks) if len(peaks) > 1 else np.array([1])
            regularity = 1.0 - np.std(isis) / (np.mean(isis) + 1e-10)
            regularity = float(np.clip(regularity, 0, 1))
            density = float(np.clip(len(peaks) / (len(env_arr) + 1e-10), 0, 1))
            score = 0.6 * regularity + 0.4 * density
        else:
            score = 0.2
            regularity = 0.0
            density = 0.0
        return score, {"rhythm_reg": regularity, "rhythm_density": density}


# ══════════════════════════════════════════════════════════════
# GRANULAR ENGINE
class GranularEngine:
    def __init__(self, pool, clusters):
        self.has_audio = "micro_audio" in pool
        self.sizes = {0: MICRO_FRAMES * HOP, 1: MESO_FRAMES * HOP, 2: MACRO_FRAMES * HOP}
        if self.has_audio:
            self.chunks = {0: pool["micro_audio"], 1: pool["meso_audio"], 2: pool["macro_audio"]}
        else:
            self.sources = {0: pool["micro_sources"], 1: pool["meso_sources"], 2: pool["macro_sources"]}
            self._file_cache = {}
        self.cluster_map = {}
        lm = {"micro": 0, "meso": 1, "macro": 2}
        for ln, ids in clusters.items():
            for j, cid in enumerate(ids):
                cid = int(cid)
                if cid not in self.cluster_map: self.cluster_map[cid] = []
                self.cluster_map[cid].append((lm[ln], j))

    def _get_file(self, fp):
        if fp in self._file_cache:
            return self._file_cache[fp]
        try:
            y, fsr = sf.read(fp, dtype="float32", always_2d=False)
        except Exception:
            # source file missing/locked — cache a quiet placeholder so the
            # whole generation doesn't die on one bad file
            y = np.zeros(SR, dtype=np.float32)
            self._file_cache[fp] = y
            return y
        if y.ndim > 1: y = np.mean(y, axis=1)
        if fsr != SR: y = librosa.resample(y, orig_sr=fsr, target_sr=SR)
        if len(self._file_cache) < 200:
            self._file_cache[fp] = y
        return y

    def _load_grain(self, level, idx):
        if self.has_audio:
            return self.chunks[level][idx].copy()
        fp, smp_start, n_samps = self.sources[level][idx]
        y = self._get_file(fp)
        chunk = y[smp_start:min(smp_start + n_samps, len(y))]
        if len(chunk) < n_samps:
            chunk = np.pad(chunk, (0, n_samps - len(chunk)))
        return chunk.copy()

    def synthesize(self, steps, total_samples):
        audio = np.zeros((2, total_samples), dtype=np.float32)
        wt = np.zeros(total_samples, dtype=np.float32)
        n = len(steps)
        if n == 0: return audio

        micro_hop = self.sizes[0] // 4
        meso_hop = self.sizes[1] // 2

        for i, s in enumerate(steps):
            center = i * (total_samples // n)

            for level, grain_size, hop, amp_scale in [
                (0, self.sizes[0], micro_hop, 0.4),
                (1, self.sizes[1], meso_hop, 0.6),
                (2, self.sizes[2], self.sizes[2], 1.0),
            ]:
                cluster = int(s["cluster"])
                grains = self.cluster_map.get(cluster, [])
                chosen = None
                for ln, gi in grains:
                    if ln == level:
                        chosen = (ln, gi); break
                if chosen is None and grains:
                    chosen = grains[0]
                if chosen is None: continue

                ln, gi = chosen
                chunk = self._load_grain(ln, gi)

                pitch = float(np.clip(s["pitch"] * (1.5 if level == 0 else 0.5 if level == 2 else 1.0), -12, 12))
                stretch = float(np.clip(s["stretch"], 0.3, 3.0))
                rate = 2**(pitch/12) * stretch
                if abs(rate-1) > 0.05 and 0.1 < rate < 5:
                    tl = int(len(chunk)/rate)
                    if 10 < tl < len(chunk)*5:
                        chunk = np.interp(np.linspace(0, len(chunk)-1, tl), np.arange(len(chunk)), chunk)

                if s.get("reverse", 0) > 0.5: chunk = chunk[::-1].copy()

                base_amp = float(np.clip(s["amp"], 0.05, 1)) * amp_scale
                pan = float(np.clip(s["pan"], -1, 1))
                gl = np.cos((pan+1)*np.pi/4)
                gr = np.sin((pan+1)*np.pi/4)

                # density controls number of overlapping grains per step
                density_val = s.get("density", 1)
                if density_val <= 0:
                    n_overlaps = 1
                else:
                    n_overlaps = int(np.clip(round(density_val), 1, 6))

                # pos_offset shifts the grain start position
                pos_offset = s.get("pos_offset", 0.0)
                pos_offset_samples = int(pos_offset * grain_size)
                for o in range(n_overlaps):
                    jitter = np.random.randint(-hop//3, hop//3) if o > 0 else 0
                    pos = int(np.clip(center + pos_offset_samples + jitter, 0, max(0, total_samples - len(chunk))))
                    end = min(pos + len(chunk), total_samples)
                    al = end - pos
                    env = np.hanning(al) * base_amp * (0.7 ** o)
                    audio[0, pos:end] += chunk[:al] * env * gl
                    audio[1, pos:end] += chunk[:al] * env * gr
                    wt[pos:end] += env * (gl+gr)/2 + 1e-8

        wt[wt<1e-6]=1
        audio[0]/=wt; audio[1]/=wt
        return audio

    def synthesize_multi(self, stream_steps, total_samples):
        """Hybrid v1+v3: full-band streams summed with weights. No band-pass = no spectral holes.
        Each stream generates full-band audio with its own pan/character, weighted by STREAM_WEIGHTS."""
        combined = np.zeros((2, total_samples), dtype=np.float32)

        for stream_idx, steps in stream_steps:
            if not steps: continue
            stream_audio = self.synthesize(steps, total_samples)

            # Perceptual weighting: sub/drums louder, air quieter
            weight = STREAM_WEIGHTS[stream_idx] if stream_idx < len(STREAM_WEIGHTS) else 1.0

            stream_rms = np.sqrt(np.mean(stream_audio**2) + 1e-10)
            if stream_rms > 1e-6:
                norm_rms = 0.1 * weight
                stream_audio *= norm_rms / stream_rms

            combined += stream_audio

            # Perceptual weighting: sub/drums louder, air quieter
            weight = STREAM_WEIGHTS[stream_idx] if stream_idx < len(STREAM_WEIGHTS) else 1.0

            stream_rms = np.sqrt(np.mean(stream_audio**2) + 1e-10)
            if stream_rms > 1e-6:
                norm_rms = 0.1 * weight
                stream_audio *= norm_rms / stream_rms

            combined += stream_audio

        peak = np.max(np.abs(combined))
        if peak > 0.95:
            combined *= 0.95 / peak

        return combined


# ══════════════════════════════════════════════════════════════
# MASTERING
# ══════════════════════════════════════════════════════════════
def butter_band(data, low, high, sr, order=4):
    from scipy.signal import butter, sosfilt
    nyq = sr / 2
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    sos = butter(order, [low_n, high_n], btype='band', output='sos')
    return sosfilt(sos, data)

def butter_low(data, freq, sr, order=4):
    from scipy.signal import butter, sosfilt
    nyq = sr / 2
    sos = butter(order, min(freq / nyq, 0.999), btype='low', output='sos')
    return sosfilt(sos, data)

def butter_high(data, freq, sr, order=4):
    from scipy.signal import butter, sosfilt
    nyq = sr / 2
    sos = butter(order, max(freq / nyq, 0.001), btype='high', output='sos')
    return sosfilt(sos, data)

def ott_compress(x, threshold_db=-20, ratio=4.0, attack_ms=10, release_ms=100, sr=SR):
    from scipy.signal import lfilter
    threshold = 10 ** (threshold_db / 20)
    attack = np.exp(-1 / (sr * attack_ms / 1000))
    release = np.exp(-1 / (sr * release_ms / 1000))
    env = np.zeros(len(x))
    env[0] = abs(x[0])
    for i in range(1, len(x)):
        coeff = attack if abs(x[i]) > env[i-1] else release
        env[i] = coeff * env[i-1] + (1 - coeff) * abs(x[i])
    gain = np.ones(len(x))
    mask = env > threshold
    gain[mask] = threshold + (env[mask] - threshold) / ratio
    gain[mask] /= env[mask]
    gain = np.clip(gain, 0.01, 1.0)
    return x * gain

def stereo_enhance(stereo, width=0.3):
    mid = (stereo[0] + stereo[1]) / 2
    side = (stereo[0] - stereo[1]) / 2
    side *= (1.0 + width)
    out = np.zeros_like(stereo)
    out[0] = mid + side
    out[1] = mid - side
    return out

def simple_reverb(x, sr=SR, predelay_ms=15, decay=0.3, taps=6):
    """Simple multi-tap reverb for spatial width."""
    out = x.copy().astype(np.float64)
    delays_ms = np.linspace(predelay_ms, predelay_ms + 80, taps)
    gains = np.linspace(1.0, decay, taps)
    for d_ms, g in zip(delays_ms, gains):
        d = int(sr * d_ms / 1000)
        if d < len(x):
            delayed = np.zeros_like(x)
            delayed[d:] = x[:-d] * g
            out += delayed
    return out.astype(np.float32)

def stereo_spread(stereo, sr=SR):
    """Mono lows + subtle reverb for depth. Simple and phase-safe."""
    from scipy.signal import butter as _butter, sosfilt as _sosfilt
    out = stereo.copy().astype(np.float64)

    # Force lows mono (<150Hz) — clean, no phase issues
    sos_lo = _butter(4, min(150/(sr/2), 0.99), btype='low', output='sos')
    lo_l = _sosfilt(sos_lo, out[0])
    lo_r = _sosfilt(sos_lo, out[1])
    mono_lo = (lo_l + lo_r) / 2
    out[0] = out[0] - lo_l + mono_lo
    out[1] = out[1] - lo_r + mono_lo

    # Very subtle side on highs only (>800Hz), 1.1x max
    sos_hi = _butter(4, max(800/(sr/2), 0.001), btype='high', output='sos')
    hi_l = _sosfilt(sos_hi, out[0])
    hi_r = _sosfilt(sos_hi, out[1])
    m = (hi_l + hi_r) / 2
    s = (hi_l - hi_r) / 2
    s *= 1.1
    lo_l = out[0] - _sosfilt(sos_hi, out[0])
    lo_r = out[1] - _sosfilt(sos_hi, out[1])
    out[0] = lo_l + m + s
    out[1] = lo_r + m - s

    # Short reverb, lows excluded
    for ch in range(2):
        rev = simple_reverb(out[ch], sr, predelay_ms=10, decay=0.15, taps=4)
        rev_hi = _sosfilt(sos_hi, rev)
        out[ch] = out[ch] * 0.85 + rev_hi * 0.15

    return out.astype(np.float32)

def saturate(x, drive=0.15):
    driven = x * (1.0 + drive * 3)
    return np.tanh(driven) / np.tanh(1.0 + drive * 3)

def limiter_fast(x, ceiling_db=-0.3, release_ms=40, sr=SR):
    ceiling = 10 ** (ceiling_db / 20)
    attack_samples = int(sr * 0.001)
    release_samples = int(sr * release_ms / 1000)
    attack_coeff = np.exp(-1 / max(attack_samples, 1))
    release_coeff = np.exp(-1 / max(release_samples, 1))
    n = len(x)
    env = np.zeros(n, dtype=np.float64)
    env[0] = abs(x[0])
    for i in range(1, n):
        coeff = attack_coeff if abs(x[i]) > env[i-1] else release_coeff
        env[i] = coeff * env[i-1] + (1 - coeff) * abs(x[i])
    gain = np.ones(n, dtype=np.float64)
    over = env > ceiling
    gain[over] = ceiling / env[over]
    return (x * gain).astype(np.float32)

def master(stereo, sr=SR):
    from scipy.signal import butter as _butter, sosfilt as _sosfilt
    print("🎛️ Mastering...")
    out = stereo.copy().astype(np.float64)

    # 1. Highpass 30Hz
    sos_hp = _butter(2, max(30/(sr/2), 0.001), btype='high', output='sos')
    for ch in range(2):
        out[ch] = _sosfilt(sos_hp, out[ch])

    # 2. OTT on lows (v1 style) — tightens bass
    sos_lo300 = _butter(2, min(300/(sr/2), 0.99), btype='low', output='sos')
    for ch in range(2):
        lo = _sosfilt(sos_lo300, out[ch])
        rest = out[ch] - lo
        out[ch] = rest + ott_compress(lo, threshold_db=-18, ratio=3.0, attack_ms=8, release_ms=120, sr=sr) * 1.2

    # 3. Low shelf boost +6dB
    sos_lo = _butter(4, min(200/(sr/2), 0.99), btype='low', output='sos')
    for ch in range(2):
        lo = _sosfilt(sos_lo, out[ch])
        out[ch] = out[ch] + lo * 1.0

    # 4. High shelf cut -6.5dB above 1kHz
    sos_hi = _butter(4, max(1000/(sr/2), 0.001), btype='high', output='sos')
    hi_factor = 10**(-6.5/20) - 1
    for ch in range(2):
        hi = _sosfilt(sos_hi, out[ch])
        out[ch] = out[ch] + hi * hi_factor

    # 5. Mono below 200Hz
    mono_lo = (_sosfilt(sos_lo, out[0]) + _sosfilt(sos_lo, out[1])) / 2
    for ch in range(2):
        lo_part = _sosfilt(sos_lo, out[ch])
        out[ch] = out[ch] - lo_part + mono_lo

    # 6. Stereo enhance (v1 style) — adds cohesion, not binary L/R
    out = stereo_enhance(out, width=0.25)

    # 7. Saturate (v1 style) — warmth, glues transients
    for ch in range(2):
        out[ch] = saturate(out[ch], drive=0.12)

    # 8. Gain to -12.5 dBFS RMS
    rms_cur = np.sqrt(np.mean(out**2))
    if rms_cur > 0:
        target_rms = 10**(-12.5/20)
        out *= target_rms / rms_cur

    # 9. Limiter
    for ch in range(2):
        out[ch] = limiter_fast(out[ch], ceiling_db=-0.1, release_ms=15, sr=sr)

    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    rms = np.sqrt(np.mean(out**2))
    pk = max(np.max(np.abs(out[0])), np.max(np.abs(out[1])))
    print(f"   RMS: {rms:.3f} ({20*np.log10(rms+1e-10):.1f} dBFS) Peak: {pk:.3f}")
    return out
class STFTCritic:
    def __init__(self, pool, engine=None):
        print("🔬 Building STFT critic...")
        has_audio = "micro_audio" in pool
        n_sample = min(5000, len(pool["micro_feats"]))

        if has_audio:
            chunks = pool["micro_audio"][:n_sample]
        elif engine:
            chunks = []
            for idx in range(min(n_sample, len(pool["micro_sources"]))):
                chunks.append(engine._load_grain(0, idx))
        else:
            chunks = []

        specs = []
        for c in chunks:
            S = np.abs(librosa.stft(c, n_fft=N_FFT, hop_length=HOP))
            S_db = librosa.amplitude_to_db(S+1e-10, ref=np.max)
            specs.append(S_db.mean(axis=1))
        specs = np.array(specs) if specs else np.zeros((1, N_FFT//2+1))
        self.spec_mean = specs.mean(axis=0)
        self.spec_std = specs.std(axis=0) + 1e-8
        print(f"  ✅ Critic from {len(chunks)} grains")

    def score(self, audio):
        if audio.ndim > 1: audio = np.mean(audio, axis=0)
        S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP))
        S_db = librosa.amplitude_to_db(S+1e-10, ref=np.max)
        ms = S_db.mean(axis=1)
        ml = min(len(ms), len(self.spec_mean))
        sd = np.abs(ms[:ml] - self.spec_mean[:ml]).mean()
        spec_s = float(np.exp(-sd / 10))

        frame_e = S_db.mean(axis=0)
        env_s = float(min(np.std(frame_e) / 10, 1.0))
        rms = np.sqrt(np.mean(audio**2)+1e-10)
        eng_s = float(min(rms * 10, 1.0))
        flux = np.diff(ms)
        fl_s = float(min(np.abs(flux).mean() / 3, 1.0))

        total = 0.4*spec_s + 0.2*env_s + 0.2*eng_s + 0.2*fl_s
        return total, {"spectral": spec_s, "envelope": env_s, "energy": eng_s, "flux": fl_s}


# ══════════════════════════════════════════════════════════════
# VISUAL GRAINS — granular synthesis for images
# ══════════════════════════════════════════════════════════════
def extract_visual_feat(patch):
    """22-dim feature vector from an RGB patch (same dim as audio features)."""
    arr = np.array(patch, dtype=np.float64) / 255.0
    feat = []
    # Color stats per channel (6)
    for c in range(3):
        ch = arr[:,:,c]
        feat.append(float(np.mean(ch)))
        feat.append(float(np.std(ch)))
    # Overall brightness + contrast (2)
    gray = np.mean(arr, axis=2)
    feat.append(float(np.mean(gray)))
    feat.append(float(np.std(gray)))
    # Texture: gradient magnitude (2)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    feat.append(float(np.mean(np.abs(gx))))
    feat.append(float(np.mean(np.abs(gy))))
    # Spatial distribution: quadrant brightness (4)
    h, w = gray.shape
    feat.append(float(np.mean(gray[:h//2, :w//2])))
    feat.append(float(np.mean(gray[:h//2, w//2:])))
    feat.append(float(np.mean(gray[h//2:, :w//2])))
    feat.append(float(np.mean(gray[h//2:, w//2:])))
    # Frequency: mean of FFT magnitude per channel (3)
    for c in range(3):
        fft_mag = np.abs(np.fft.fft2(arr[:,:,c]))
        feat.append(float(np.mean(fft_mag) / (np.max(fft_mag) + 1e-10)))
    # Color ratios (3)
    total = np.mean(arr) + 1e-10
    feat.append(float(np.mean(arr[:,:,0]) / total))
    feat.append(float(np.mean(arr[:,:,1]) / total))
    feat.append(float(np.mean(arr[:,:,2]) / total))
    # Edge density (2)
    gx_pad = np.pad(gx, ((0,0),(0,1)))  # (H, W)
    gy_pad = np.pad(gy, ((0,1),(0,0)))  # (H, W)
    edges = np.sqrt(gx_pad**2 + gy_pad**2)
    feat.append(float(np.mean(edges)))
    feat.append(float(np.std(edges)))
    return np.array(feat[:VIS_FEAT_DIM], dtype=np.float32)


class VisualGrainExtractor:
    """Extract visual grains (patches + features) from images."""
    def __init__(self):
        self.micro_size = VIS_MICRO_SIZE
        self.meso_size = VIS_MESO_SIZE
        self.macro_size = VIS_MACRO_SIZE

    def extract(self, image_paths, max_per_image=50):
        micro_feats, micro_patches = [], []
        meso_feats, meso_patches = [], []
        macro_feats, macro_patches = [], []
        errors = 0

        print(f"\n🖼️ Extracting visual grains from {len(image_paths)} images...")
        t0 = time.time()

        for i, fp in enumerate(image_paths):
            if (i+1) % 100 == 0 or i+1 == len(image_paths):
                print(f"\r  [{i+1}/{len(image_paths)}] μ={len(micro_feats)} σ={len(meso_feats)} Ω={len(macro_feats)}", end="", flush=True)
            try:
                img = Image.open(fp).convert("RGB")
                w, h = img.size
                # resize to manageable size
                if max(w, h) > 1024:
                    scale = 1024 / max(w, h)
                    img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                arr = np.array(img)
                cnt = 0

                for size, f_list, p_list in [
                    (self.micro_size, micro_feats, micro_patches),
                    (self.meso_size, meso_feats, meso_patches),
                    (self.macro_size, macro_feats, macro_patches),
                ]:
                    cnt = 0
                    h_img, w_img = arr.shape[:2]
                    stride = max(size // 2, 1)
                    for y in range(0, h_img - size + 1, stride):
                        for x in range(0, w_img - size + 1, stride):
                            if cnt >= max_per_image: break
                            patch = arr[y:y+size, x:x+size]
                            # skip very dark patches
                            if np.mean(patch) < 10: continue
                            feat = extract_visual_feat(
                                Image.fromarray(patch))
                            if np.any(np.isnan(feat)) or np.all(feat == 0): continue
                            f_list.append(feat)
                            p_list.append(patch.astype(np.float32) / 255.0)
                            cnt += 1
            except Exception:
                errors += 1

        elapsed = time.time() - t0
        print(f"\n  ✅ {elapsed:.1f}s — μ={len(micro_feats)} σ={len(meso_feats)} Ω={len(macro_feats)} err={errors}")

        return {
            "micro_feats": np.array(micro_feats, dtype=np.float32) if micro_feats else np.zeros((0, VIS_FEAT_DIM), np.float32),
            "meso_feats": np.array(meso_feats, dtype=np.float32) if meso_feats else np.zeros((0, VIS_FEAT_DIM), np.float32),
            "macro_feats": np.array(macro_feats, dtype=np.float32) if macro_feats else np.zeros((0, VIS_FEAT_DIM), np.float32),
            "micro_patches": np.array(micro_patches, dtype=np.float32) if micro_patches else np.zeros((0, self.micro_size, self.micro_size, 3), np.float32),
            "meso_patches": np.array(meso_patches, dtype=np.float32) if meso_patches else np.zeros((0, self.meso_size, self.meso_size, 3), np.float32),
            "macro_patches": np.array(macro_patches, dtype=np.float32) if macro_patches else np.zeros((0, self.macro_size, self.macro_size, 3), np.float32),
        }


def build_visual_clusters(pool):
    print("🔬 Clustering visual grains...")
    all_f = np.concatenate([pool["micro_feats"], pool["meso_feats"], pool["macro_feats"]])
    if len(all_f) > 50000:
        idx = np.random.choice(len(all_f), 50000, replace=False)
        fit_f = all_f[idx]
    else:
        fit_f = all_f
    n_cl = min(VIS_N_CLUSTERS, len(fit_f))
    kmeans = MiniBatchKMeans(n_clusters=n_cl, batch_size=1024, n_init=3, random_state=42)
    kmeans.fit(fit_f)
    all_labels = kmeans.predict(all_f)
    n_m = len(pool["micro_feats"])
    n_s = len(pool["meso_feats"])
    clusters = {
        "micro": all_labels[:n_m],
        "meso": all_labels[n_m:n_m+n_s],
        "macro": all_labels[n_m+n_s:],
    }
    print(f"  ✅ {n_cl} visual clusters")
    return clusters


def _visual_feat_for_grain(vis_pool, level, grain_idx):
    """Return the feature vector of a visual grain given (level, index)."""
    key = ["micro_feats", "meso_feats", "macro_feats"][level]
    feats = vis_pool[key]
    if feats is not None and len(feats) > 0:
        return feats[min(grain_idx, len(feats) - 1)].astype(np.float32)
    return np.zeros(VIS_FEAT_DIM, dtype=np.float32)


class VisualEngine:
    """Composites visual grains into an image canvas."""
    def __init__(self, pool, clusters):
        self.sizes = {
            0: VIS_MICRO_SIZE, 1: VIS_MESO_SIZE, 2: VIS_MACRO_SIZE,
        }
        self.patches = {
            0: pool["micro_patches"], 1: pool["meso_patches"], 2: pool["macro_patches"],
        }
        self.cluster_map = {}
        lm = {"micro": 0, "meso": 1, "macro": 2}
        for ln, ids in clusters.items():
            for j, cid in enumerate(ids):
                cid = int(cid)
                if cid not in self.cluster_map:
                    self.cluster_map[cid] = []
                self.cluster_map[cid].append((lm[ln], j))

    def render(self, steps, canvas_size=VIS_CANVAS_SIZE):
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.float64)
        wt = np.zeros((canvas_size, canvas_size), dtype=np.float64)

        for s in steps:
            cluster = int(s["cluster"])
            grains = self.cluster_map.get(cluster, [])
            if not grains: continue
            ln, gi = grains[np.random.randint(len(grains))]
            patch = self.patches[ln][gi]

            size = self.sizes[ln]
            pos_x = int(np.clip(s.get("pos_x", 0.5), 0, 1) * (canvas_size - size))
            pos_y = int(np.clip(s.get("pos_y", 0.5), 0, 1) * (canvas_size - size))

            # optional: scale the patch
            target_size = int(size * np.clip(s.get("scale", 1.0), 0.5, 3.0))
            target_size = max(target_size, 4)
            if target_size != size:
                p_img = Image.fromarray((patch * 255).astype(np.uint8))
                p_img = p_img.resize((target_size, target_size), Image.LANCZOS)
                patch = np.array(p_img, dtype=np.float64) / 255.0

            # per-grain contrast stretch — makes dark photo patches read as
            # vivid color tiles instead of near-black squares
            lo = float(patch.min())
            hi = float(patch.max())
            if hi - lo > 0.02:
                patch = (patch - lo) / (hi - lo)

            # alpha blending
            alpha = float(np.clip(s.get("alpha", 0.5), 0.0, 1.0))

            # bounds
            end_y = min(pos_y + target_size, canvas_size)
            end_x = min(pos_x + target_size, canvas_size)
            ph = end_y - pos_y
            pw = end_x - pos_x
            if ph <= 0 or pw <= 0: continue

            p_crop = patch[:ph, :pw]
            canvas[pos_y:end_y, pos_x:end_x] += p_crop * alpha
            wt[pos_y:end_y, pos_x:end_x] += alpha

        # normalize where weighted
        mask = wt > 1e-6
        for c in range(3):
            canvas[:,:,c] = np.where(mask, canvas[:,:,c] / np.maximum(wt, 1e-6), canvas[:,:,c])

        # diagnostics (useful for debugging coverage/brightness)
        if os.environ.get("GF_VIS_DEBUG"):
            print(f"   [render] mask_cov={mask.mean()*100:.1f}% placed_mean="
                  f"{canvas[mask].mean(axis=0) if mask.any() else (0,0,0)}")

        # Fill unweighted (empty) regions with a soft dark base so the canvas
        # is never pure black. Derived from the average of the placed grains
        # (dimmed), giving the composite a continuous "field" behind the grains.
        placed = canvas[mask] if mask.any() else canvas.reshape(-1, 3)
        if len(placed) > 0:
            base = placed.mean(axis=0) * 0.35
        else:
            base = np.array([0.1, 0.1, 0.12])
        filler = np.zeros_like(canvas)
        filler[:,:,0] = base[0]; filler[:,:,1] = base[1]; filler[:,:,2] = base[2]
        for c in range(3):
            canvas[:,:,c] = np.where(mask, canvas[:,:,c], filler[:,:,c])

        canvas = np.clip(canvas * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(canvas)

    def render_grid(self, steps, canvas_size=VIS_CANVAS_SIZE):
        """Debug: render each grain on its own tile in a grid."""
        n = len(steps)
        if n == 0:
            return Image.fromarray(np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8))
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        tile = canvas_size // max(cols, 1)
        grid = Image.new("RGB", (cols * tile, rows * tile))
        for i, s in enumerate(steps):
            cluster = int(s["cluster"])
            grains = self.cluster_map.get(cluster, [])
            if not grains: continue
            ln, gi = grains[0]
            patch = self.patches[ln][gi]
            p_img = Image.fromarray((patch * 255).astype(np.uint8))
            p_img = p_img.resize((tile, tile), Image.NEAREST)
            r, c = divmod(i, cols)
            grid.paste(p_img, (c * tile, r * tile))
        return grid


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════
def extract_params_from_feats(feat_prev, feat_next):
    """Extract relative synthesis params from feature difference between consecutive grains.

    Feature layout (FEAT_DIM=22):
      0: mean_spectrum, 1: std_spectrum, 2: centroid, 3: bandwidth,
      4: flatness, 5: rolloff, 6-11: mel_band[0..5],
      12: mean_frame_energy, 13: std_frame_energy, 14: peak_frame_energy,
      15: std_diff_frame_energy, 16: flux_pos, 17: flux_abs,
      18: ratio_lo, 19: ratio_mid, 20: ratio_hi, 21: low_freq_ratio
    """
    params = np.zeros(8, dtype=np.float32)
    diff = feat_next - feat_prev
    # [0] pitch: spectral centroid shift → semitone-like
    params[0] = float(np.clip(diff[2] * 12, -1, 1))
    # [1] stretch: std_diff_frame_energy → how choppy/fluid
    params[1] = float(np.clip(diff[15] * 10, -1, 1))
    # [2] amp: overall energy ratio
    params[2] = float(np.clip(diff[0] / (abs(feat_prev[0]) + 1e-6), -1, 1))
    # [3] pan: spectral balance shift (mid ratio delta)
    params[3] = float(np.clip(diff[19] * 5, -1, 1))
    # [4] density: flux_abs → how active the texture is
    params[4] = float(np.clip(diff[17] * 10, -1, 1))
    # [5] pos_offset: rolloff shift → temporal position within grain
    params[5] = float(np.clip(diff[5] * 3, -1, 1))
    # [6] reverse: if energy drops sharply
    params[6] = float(1.0 if diff[0] < -0.01 else -1.0)
    # [7] unused (reserved)
    return params


def _level_to_all_f_idx(lev, idx, n_micro, n_meso):
    """Convert (level, index_within_level) to absolute index in all_f concatenation."""
    if lev == 0:
        return min(idx, n_micro - 1)
    elif lev == 1:
        return n_micro + min(idx, n_meso - 1)
    else:
        return n_micro + n_meso + min(idx, 999999)


def _cluster_to_grain(cluster_id, level, engine):
    """Look up a grain in the cluster_map that matches the desired level.
    Returns (level, grain_index) or None."""
    grains = engine.cluster_map.get(cluster_id, [])
    for ln, gi in grains:
        if ln == level:
            return (ln, gi)
    if grains:
        return grains[0]
    return None


def _cluster_feat_for_ctx(cluster_id, engine, all_f):
    """Return the feature vector of a grain from this cluster (for context feedback)."""
    grains = engine.cluster_map.get(cluster_id, [])
    if not grains:
        return all_f[min(cluster_id, len(all_f) - 1)]
    ln, gi = grains[0]
    if engine.has_audio:
        return all_f[min(gi, len(all_f) - 1)]  # fallback
    else:
        sources = engine.sources.get(ln)
        if sources and gi < len(sources):
            fp, smp_start, n_samps = sources[gi]
            try:
                y = engine._get_file(fp)
                chunk = y[smp_start:min(smp_start + n_samps, len(y))]
                S = np.abs(librosa.stft(chunk.astype(np.float64), n_fft=N_FFT, hop_length=HOP))
                return extract_feat_from_stft(S[:, :5])
            except Exception:
                pass
    return all_f[min(cluster_id, len(all_f) - 1)]


def build_training_pairs(pool, clusters):
    print("\n📐 Building training pairs from trajectories...")
    all_f = np.concatenate([pool["micro_feats"], pool["meso_feats"], pool["macro_feats"]])
    n_micro = len(pool["micro_feats"])
    n_meso = len(pool["meso_feats"])
    n_macro = len(pool["macro_feats"])
    pairs = []

    for traj in pool.get("trajectories", []):
        if len(traj) < CONTEXT_LEN + 2: continue
        for k in range(CONTEXT_LEN, len(traj)):
            ctx_feats = []
            for j in range(k - CONTEXT_LEN, k):
                lev, idx = traj[j]
                gi = _level_to_all_f_idx(lev, idx, n_micro, n_meso)
                ctx_feats.append(all_f[gi])

            target_lev, target_idx = traj[k]
            target_all_f_idx = _level_to_all_f_idx(target_lev, target_idx, n_micro, n_meso)

            prev_lev, prev_idx = traj[k-1]
            prev_all_f_idx = _level_to_all_f_idx(prev_lev, prev_idx, n_micro, n_meso)

            params = extract_params_from_feats(all_f[prev_all_f_idx], all_f[target_all_f_idx])

            ln = ["micro","meso","macro"][target_lev]
            cluster_id = int(clusters[ln][min(target_idx, len(clusters[ln])-1)])

            # density from flux_abs (feature 17), pan from ratio_mid delta (feature 19)
            target_feat = all_f[target_all_f_idx]
            prev_feat = all_f[prev_all_f_idx]
            density = float(np.clip(target_feat[17] * 5, 0, 1))
            pan = float(np.clip((target_feat[19] - target_feat[18]) * 3, -1, 1))

            pairs.append({
                "ctx": np.array(ctx_feats, dtype=np.float32),
                "cluster": cluster_id,
                "level": target_lev,
                "params": params,
                "density": density,
                "pan": pan,
            })

    print(f"  ✅ {len(pairs)} pairs")
    return pairs


class PairDS(Dataset):
    def __init__(self, pairs):
        self.p = pairs
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        p = self.p[i]
        return (torch.tensor(p["ctx"]), torch.tensor(p["cluster"]),
                torch.tensor(p["level"]), torch.tensor(p["params"]))


def build_visual_training_pairs(vis_pool, vis_clusters):
    """Build visual training pairs from visual grain trajectories.
    Since images have no temporal order, we synthesize trajectories by
    random walk through feature space — each step predicts the next visual grain."""
    all_f = np.concatenate([vis_pool["micro_feats"], vis_pool["meso_feats"], vis_pool["macro_feats"]])
    n_micro = len(vis_pool["micro_feats"])
    n_meso = len(vis_pool["meso_feats"])

    def _v_idx(lev, idx):
        if lev == 0:
            return min(idx, n_micro - 1)
        elif lev == 1:
            return n_micro + min(idx, n_meso - 1)
        else:
            return n_micro + n_meso + min(idx, len(all_f) - 1)

    pairs = []
    rng = np.random.RandomState(42)

    lm = ["micro", "meso", "macro"]
    # Build per-cluster index for random walks
    for _ in range(min(20000, len(all_f) // 4)):
        lev = rng.randint(0, 3)
        ln = lm[lev]
        labels = vis_clusters[ln]
        n_l = len(labels)
        if n_l == 0: continue
        start = rng.randint(0, n_l)
        cluster_id = int(labels[start])

        ctx = []
        # random context of CONTEXT_LEN feature vectors from same cluster family
        for j in range(CONTEXT_LEN):
            gi = _v_idx(lev, (start + j * 7) % n_l)
            ctx.append(all_f[gi])
        ctx = np.array(ctx, dtype=np.float32)

        # target: another grain from nearby in feature space
        t_idx = (start + rng.randint(1, max(n_l // 10, 2))) % n_l
        target_cluster = int(labels[t_idx])

        # blend params: 4 dims (pos_x, pos_y, scale, alpha)
        prev_f = all_f[_v_idx(lev, start)]
        tgt_f = all_f[_v_idx(lev, t_idx)]
        pos_x = float(np.clip(0.5 + (tgt_f[2] - prev_f[2]) * 2, -1, 1))
        pos_y = float(np.clip(0.5 + (tgt_f[6] - prev_f[6]) * 4, -1, 1))
        scale = float(np.clip(1.0 + (tgt_f[13] - prev_f[13]) * 4, -1, 1))
        alpha = float(np.clip(0.5 + abs(tgt_f[1] - prev_f[1]) * 3, 0, 1))

        pairs.append({
            "v_ctx": ctx,
            "v_cluster": target_cluster,
            "v_blend": np.array([pos_x, pos_y, scale, alpha], dtype=np.float32),
        })

    print(f"  ✅ {len(pairs)} visual pairs")
    return pairs


class MultiPairDS(Dataset):
    """Training dataset for MultiNavigator. Each pair gets a random stream index."""
    def __init__(self, pairs, n_streams=N_STREAMS):
        self.p = pairs
        self.n_streams = n_streams
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        p = self.p[i]
        stream_idx = np.random.randint(0, self.n_streams)
        return (torch.tensor(p["ctx"]), torch.tensor(p["cluster"]),
                torch.tensor(p["level"]), torch.tensor(p["params"]),
                torch.tensor(p.get("density", 0.5), dtype=torch.float32),
                torch.tensor(p.get("pan", 0.0), dtype=torch.float32),
                torch.tensor(stream_idx, dtype=torch.long))


class VisualPairDS(Dataset):
    """Training dataset for the visual head."""
    def __init__(self, pairs):
        self.p = pairs
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        p = self.p[i]
        return (torch.tensor(p["v_ctx"]), torch.tensor(p["v_cluster"]),
                torch.tensor(p["v_blend"]))


def train(model, pairs, n_steps=TRAIN_STEPS):
    ds = PairDS(pairs)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    print(f"\n🔥 TRAINING NAVIGATOR ({n_steps} steps, {DEVICE})")
    print(f"   Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M | Pairs: {len(ds)}")

    model.train()
    losses = []
    t0 = time.time()
    step = 0

    while step < n_steps:
        for batch in loader:
            if step >= n_steps: break
            ctx, tgt_c, tgt_l, tgt_p = batch
            ctx, tgt_c, tgt_l, tgt_p = ctx.to(DEVICE), tgt_c.to(DEVICE), tgt_l.to(DEVICE), tgt_p.to(DEVICE)
            cl, lv, pr = model(ctx)
            loss_c = F.cross_entropy(cl, tgt_c)
            loss_l = 0.5 * F.cross_entropy(lv, tgt_l)
            loss_p = F.mse_loss(torch.tanh(pr), tgt_p)
            loss = loss_c + loss_l + loss_p
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            losses.append(loss.item())
            step += 1
            if step % 500 == 0:
                avg = np.mean(losses[-500:]); e = time.time()-t0
                print(f"  step {step:5d}/{n_steps}  loss={avg:.4f} (c={loss_c:.3f} l={loss_l:.3f} p={loss_p:.3f})  {e:.0f}s  ETA {e/step*(n_steps-step):.0f}s")

    print(f"\n   ✅ {time.time()-t0:.1f}s, loss={np.mean(losses[-100:]):.4f}")
    return model


def train_multi(model, pairs, n_steps=TRAIN_STEPS, vis_pairs=None, vis_freq=1):
    ds = MultiPairDS(pairs)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    vis_loader = None
    vis_iter = None
    if vis_pairs and len(vis_pairs) > 0:
        vis_ds = VisualPairDS(vis_pairs)
        vis_loader = DataLoader(vis_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        vis_iter = iter(vis_loader)
        print(f"   Visual pairs: {len(vis_ds)}")

    print(f"\n🔥 TRAINING MULTI-NAVIGATOR ({n_steps} steps, {DEVICE})")
    print(f"   Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M | Pairs: {len(ds)}")

    model.train()
    losses = []

    def _next_vis_batch():
        nonlocal vis_iter, vis_loader
        try:
            return next(vis_iter)
        except StopIteration:
            if vis_loader is None:
                return None
            vis_iter = iter(vis_loader)
            try:
                return next(vis_iter)
            except StopIteration:
                return None

    t0 = time.time()
    step = 0

    while step < n_steps:
        audio_iter = iter(loader)
        for batch in audio_iter:
            if step >= n_steps: break
            ctx, tgt_c, tgt_l, tgt_p, tgt_dn, tgt_pn, stream_idx = batch
            ctx = ctx.to(DEVICE); tgt_c = tgt_c.to(DEVICE)
            tgt_l = tgt_l.to(DEVICE); tgt_p = tgt_p.to(DEVICE)
            tgt_dn = tgt_dn.to(DEVICE); tgt_pn = tgt_pn.to(DEVICE)
            stream_idx = stream_idx.to(DEVICE)

            vis_in = None
            if vis_iter is not None and step % vis_freq == 0:
                vb = _next_vis_batch()
                if vb is not None:
                    vctx, tgt_vc, tgt_vb = vb[0].to(DEVICE), vb[1].to(DEVICE), vb[2].to(DEVICE)
                    vis_in = vctx

            audio_out, vis_out = model(ctx, stream_idx=stream_idx, vis_states=vis_in)
            cl, lv, pr, dn, pn = audio_out

            loss_c = F.cross_entropy(cl, tgt_c)
            loss_l = 0.5 * F.cross_entropy(lv, tgt_l)
            loss_p = F.mse_loss(pr, tgt_p)
            loss_dn = F.mse_loss(dn.squeeze(-1), tgt_dn)
            loss_pn = F.mse_loss(pn.squeeze(-1), tgt_pn)
            # attractor coherence: push attractor state toward the mean of the target context
            # this teaches attractors to represent meaningful musical directions
            ctx_proj = model.proj(model.feat_enc(ctx)).mean(dim=1)  # [B, hidden]
            loss_at = 0.1 * F.mse_loss(model.attractor_state[stream_idx], ctx_proj)
            loss = loss_c + loss_l + loss_p + 0.5 * loss_dn + 0.5 * loss_pn + loss_at

            # visual head loss
            loss_v = 0.0
            if vis_out is not None:
                vcl, vbl = vis_out
                loss_vc = F.cross_entropy(vcl, tgt_vc)
                loss_vbl = F.mse_loss(vbl, tgt_vb)
                loss_v = 0.5 * loss_vc + 0.5 * loss_vbl
                loss = loss + loss_v

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            losses.append(loss.item())
            step += 1
            if step % 500 == 0:
                avg = np.mean(losses[-500:]); e = time.time()-t0
                print(f"  step {step:5d}/{n_steps}  loss={avg:.4f}  (a={loss_c:.3f} l={loss_l:.3f} p={loss_p:.3f} v={loss_v:.3f})  {e:.0f}s  ETA {e/step*(n_steps-step):.0f}s")

    print(f"\n   ✅ {time.time()-t0:.1f}s, loss={np.mean(losses[-100:]):.4f}")
    return model


def generate_multi(model, engine, critic, pool, n_seconds=16, seed=42, temp=0.8, bars=8, bpm=120,
                   target_stats=None, noise_inject=0.0, vis_engine=None, vis_pool=None):
    """Multi-stream generation: 6 independent streams + spectral field.
    Optionally generates a cover image via the visual head when vis_engine is given."""
    if seed is not None: torch.manual_seed(seed); np.random.seed(seed)
    model.eval()
    dur = int(n_seconds * SR)
    n_steps = max(8, int(n_seconds / 0.5))
    print(f"\n🎵 MULTI-STREAM: {bars} bars, {bpm} BPM, {n_seconds:.1f}s, {n_steps} steps × {N_STREAMS} streams")

    all_f = np.concatenate([pool["micro_feats"], pool["meso_feats"], pool["macro_feats"]])
    field = SpectralField()

    # visual context: if a vis pool is given, build its context features
    v_all_f = None
    v_ctx = None
    if vis_engine is not None and vis_pool is not None:
        v_all_f = np.concatenate([vis_pool["micro_feats"], vis_pool["meso_feats"], vis_pool["macro_feats"]])
        v_ctx = v_all_f[np.random.choice(len(v_all_f), CONTEXT_LEN, replace=True)].copy()
        print(f"   Visual pool: {len(v_all_f)} grains → cover art enabled")

    cond = None
    if target_stats is not None:
        cond = torch.tensor(target_stats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        print(f"   Target: centroid={target_stats[0]:.2f} energy={target_stats[1]:.2f}")

    stream_steps = [[] for _ in range(N_STREAMS)]
    ctx = all_f[np.random.choice(len(all_f), CONTEXT_LEN, replace=True)]

    vis_steps = []

    # reset attractor states for fresh generation
    model.attractor_state.zero_()

    def _halton(i, base):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    for si in range(n_steps):
        if si % 50 == 0: print(f"\r  [{si}/{n_steps}]", end="", flush=True)
        ct = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        if noise_inject > 0:
            ct = ct + torch.randn_like(ct) * noise_inject

        # visual context for this step
        vis_in = None
        if v_ctx is not None:
            vis_in = torch.tensor(v_ctx, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        for s_idx in range(N_STREAMS):
            result = model.step(ct, stream_idx=s_idx, temp=temp, cond=cond, vis_states=vis_in)
            if vis_engine is not None and "v_cluster" in result:
                vb = result.get("v_blend")
                if vb is not None:
                    # Halton low-discrepancy scan guarantees the whole canvas is
                    # visited step by step; the brain only jitters around it.
                    gi = si * N_STREAMS + s_idx
                    hx = _halton(gi + 1, 2)
                    hy = _halton(gi + 1, 3)
                    jx = float(np.clip(vb[0] * 0.5 + 0.5, 0, 1)) * 0.22 - 0.11
                    jy = float(np.clip(vb[1] * 0.5 + 0.5, 0, 1)) * 0.22 - 0.11
                    vis_steps.append({
                        "cluster": result["v_cluster"],
                        "pos_x": float(np.clip(hx + jx, 0, 1)),
                        "pos_y": float(np.clip(hy + jy, 0, 1)),
                        "scale": float(np.clip(1.2 + vb[2] * 1.8, 0.5, 3.0)),
                        "alpha": float(np.clip(vb[3] * 0.5 + 0.5, 0.05, 1.0)),
                    })

            stream_steps[s_idx].append({
                "cluster": result["cluster"],
                "level": result["level"],
                "pitch": float(result["params"][0] * 12),
                "stretch": float(0.5 + result["params"][1] * 1.25),
                "amp": float(0.3 + result["params"][2] * 0.5),
                "pan": STREAM_PANS[s_idx],
                "density": float(np.clip(result["density"] * 6, 1, 6)),
                "pos_offset": float(result["params"][5] * 0.3),
                "reverse": bool(result["params"][6] > 0),
            })

        # attractor update: each stream pulls its global state toward the current context
        model.update_attractors(ct, stream_idx=list(range(N_STREAMS)), temp=temp)

        # Feedback: pick a grain from the cluster, get its features for context
        feedback_feat = _cluster_feat_for_ctx(stream_steps[0][-1]["cluster"], engine, all_f)
        ctx = np.roll(ctx, -1, axis=0)
        ctx[-1] = feedback_feat

        # visual context feedback: roll the visual context, inject a fresh visual feature
        if v_ctx is not None and vis_engine is not None and vis_steps:
            _vc = vis_steps[-1]["cluster"]
            grains = vis_engine.cluster_map.get(_vc, [])
            v_ctx = np.roll(v_ctx, -1, axis=0)
            if grains:
                ln, gi = grains[(si * 7) % len(grains)]
                v_ctx[-1] = _visual_feat_for_grain(vis_pool, ln, gi)
            else:
                v_ctx[-1] = v_all_f[np.random.randint(len(v_all_f))]

    print(f"\r  Generating {N_STREAMS} streams...")
    stereo = engine.synthesize_multi(list(enumerate(stream_steps)), dur)

    field.analyze(stereo)
    stereo = master(stereo)
    mono = np.mean(stereo, axis=0)
    score, bd = critic.score(mono)
    print(f"\n📊 Critic: {score:.3f}")
    for k, v in bd.items(): print(f"   {k}: {v:.3f}")

    fade = int(0.05 * SR)
    if stereo.shape[1] > fade*2:
        stereo[0,:fade] *= np.linspace(0,1,fade); stereo[1,:fade] *= np.linspace(0,1,fade)
        stereo[0,-fade:] *= np.linspace(1,0,fade); stereo[1,-fade:] *= np.linspace(1,0,fade)

    image = None
    if vis_engine is not None and vis_steps:
        image = vis_engine.render(vis_steps, VIS_CANVAS_SIZE)
        print(f"   🖼️ Cover art: {len(vis_steps)} visual grains → {VIS_CANVAS_SIZE}px")

    return stereo, score, image


def generate(model, engine, critic, pool, n_seconds=16, seed=42, temp=0.8, bars=8, bpm=120, closed_loop=0,
             target_stats=None, noise_inject=0.0):
    if seed is not None: torch.manual_seed(seed); np.random.seed(seed)
    model.eval()
    dur = int(n_seconds * SR)
    n_steps = max(8, int(n_seconds / 0.3))
    print(f"\n🎵 GENERATING: {bars} bars, {bpm} BPM, {n_seconds:.1f}s, {n_steps} steps")

    all_f = np.concatenate([pool["micro_feats"], pool["meso_feats"], pool["macro_feats"]])

    if target_stats is not None:
        cond = torch.tensor(target_stats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        print(f"   Target: centroid={target_stats[0]:.2f} energy={target_stats[1]:.2f} flux={target_stats[2]:.2f} low_ratio={target_stats[3]:.2f}")
    else:
        cond = None

    def _gen_steps(ctx_init, n_steps, temp):
        ctx = ctx_init.copy()
        steps = []
        for si in range(n_steps):
            ct = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            if noise_inject > 0:
                ct = ct + torch.randn_like(ct) * noise_inject
            cluster, level, raw = model.step(ct, temp=temp, cond=cond)
            p = np.clip(raw, -1, 1)
            steps.append({
                "cluster": cluster,
                "level": level,
                "pitch": float(p[0] * 12),
                "stretch": float(0.5 + p[1] * 1.25),
                "amp": float(0.3 + p[2] * 0.5),
                "pan": float(p[3]),
                "density": int(np.clip(1 + p[4] * 3, 1, 6)),
                "pos_offset": float(p[5] * 0.3),
                "reverse": bool(p[6] > 0),
            })
            feedback_feat = _cluster_feat_for_ctx(cluster, engine, all_f)
            ctx = np.roll(ctx, -1, axis=0)
            ctx[-1] = feedback_feat
        return steps, ctx

    if closed_loop > 0:
        n_segs = closed_loop
        seg_steps = n_steps // n_segs
        chunk_dur = max(int(2.0 * SR), dur // n_segs)
        overlap = max(int(0.5 * SR), chunk_dur // 4)
        n_cands = 3

        print(f"   Closed-loop: {n_segs} segments × {n_cands} candidates")

        ctx = all_f[np.random.choice(len(all_f), CONTEXT_LEN, replace=True)]
        all_steps = []
        audio = np.zeros((2, dur), dtype=np.float32)
        wt = np.zeros(dur, dtype=np.float32)

        for seg_i in range(n_segs):
            print(f"\r   Segment {seg_i+1}/{n_segs}", end="", flush=True)
            best_score = -1
            best_steps = None
            best_audio = None
            best_wt = None

            for ci in range(n_cands):
                torch.manual_seed(seed + seg_i * 100 + ci)
                np.random.seed(seed + seg_i * 100 + ci)
                seg_steps_list, _ = _gen_steps(ctx, seg_steps, temp)

                seg_dur = min(chunk_dur, dur - seg_i * chunk_dur // n_segs)
                if seg_dur <= 0: continue
                seg_stereo = engine.synthesize(seg_steps_list, seg_dur)

                mid = seg_dur // 2
                score_win = int(1.5 * SR)
                s = max(0, mid - score_win // 2)
                e = min(seg_dur, mid + score_win // 2)
                mono_chunk = np.mean(seg_stereo[:, s:e], axis=0)
                sc, _ = critic.score(mono_chunk)

                if sc > best_score:
                    best_score = sc
                    best_steps = seg_steps_list
                    best_audio = seg_stereo
                    best_wt = np.zeros(seg_dur, dtype=np.float32)
                    wt_tmp = np.zeros(seg_dur, dtype=np.float32)

            if best_audio is not None:
                pos = int(seg_i * dur / n_segs)
                al = min(len(best_audio[0]), dur - pos)
                fade = min(int(0.3 * SR), al // 4)
                win = np.ones(al)
                if fade > 0 and al > fade * 2:
                    win[:fade] = np.linspace(0, 1, fade)
                    win[-fade:] = np.linspace(1, 0, fade)

                audio[0, pos:pos+al] += best_audio[0, :al] * win
                audio[1, pos:pos+al] += best_audio[1, :al] * win
                wt[pos:pos+al] += win
                all_steps.extend(best_steps)

                ctx = np.roll(ctx, -1, axis=0)
                ctx[-1] = _cluster_feat_for_ctx(best_steps[-1]["cluster"], engine, all_f)

        print(f"\r   {len(all_steps)} events, best scores per segment")
        wt_row = wt[np.newaxis, :]
        wt_safe = np.where(wt_row > 1e-6, wt_row, 1.0)
        audio /= wt_safe
        stereo = audio
    else:
        ctx = all_f[np.random.choice(len(all_f), CONTEXT_LEN, replace=True)]
        steps = []
        for si in range(n_steps):
            if si % 20 == 0: print(f"\r  [{si}/{n_steps}]", end="", flush=True)
            ct = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            if noise_inject > 0:
                ct = ct + torch.randn_like(ct) * noise_inject
            cluster, level, raw = model.step(ct, temp=temp, cond=cond)
            p = np.clip(raw, -1, 1)
            steps.append({
                "cluster": cluster, "level": level,
                "pitch": float(p[0] * 12),
                "stretch": float(0.5 + p[1] * 1.25),
                "amp": float(0.3 + p[2] * 0.5),
                "pan": float(p[3]),
                "density": int(np.clip(1 + p[4] * 3, 1, 6)),
                "pos_offset": float(p[5] * 0.3),
                "reverse": bool(p[6] > 0),
            })
            feedback_feat = _cluster_feat_for_ctx(cluster, engine, all_f)
            ctx = np.roll(ctx, -1, axis=0)
            ctx[-1] = feedback_feat
        all_steps = steps
        print(f"\n  {len(steps)} events")
        stereo = engine.synthesize(steps, dur)

    stereo = master(stereo)
    mono = np.mean(stereo, axis=0)
    score, bd = critic.score(mono)
    print(f"\n📊 Critic: {score:.3f}")
    for k, v in bd.items(): print(f"   {k}: {v:.3f}")

    fade = int(0.05 * SR)
    if stereo.shape[1] > fade*2:
        stereo[0,:fade] *= np.linspace(0,1,fade); stereo[1,:fade] *= np.linspace(0,1,fade)
        stereo[0,-fade:] *= np.linspace(1,0,fade); stereo[1,-fade:] *= np.linspace(1,0,fade)

    return stereo, score


def attach_cover_to_wav(wav_path, image):
    """Embed a cover image into the generated WAV as an ID3v2.3 APIC frame
    carried in a standard RIFF 'ID3 ' chunk.

    Readers that look for ID3 tags on WAV (Music.app/iTunes, QuickLook,
    foobar, TagLib-based tools, Windows Explorer thumbnails) will show the
    artwork with the file. The audio data itself is untouched by the
    insertion; the RIFF size header is updated accordingly.
    """
    try:
        from mutagen.id3 import ID3, APIC
        import io, struct
    except Exception:
        print("   (mutagen not available — cover left as separate PNG)")
        return False
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")

        tag = ID3()
        tag.add(APIC(encoding=3, mime="image/png", type=3, desc="Cover",
                     data=buf.getvalue()))
        tio = io.BytesIO()
        tag.save(fileobj=tio, v2_version=3)
        tag_bytes = tio.getvalue()

        with open(wav_path, "rb") as f:
            raw = bytearray(f.read())
        if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
            print("   (not a RIFF WAVE — cover left as separate PNG)")
            return False
        riff_size = struct.unpack("<I", raw[4:8])[0]

        chunk = b"ID3 " + struct.pack("<I", len(tag_bytes)) + tag_bytes
        if len(chunk) % 2:
            chunk += b"\x00"

        raw[12:12] = chunk
        raw[4:8] = struct.pack("<I", riff_size + len(chunk))
        with open(wav_path, "wb") as f:
            f.write(raw)
        print(f"   🖼️ Cover embedded into WAV ({len(tag_bytes)} bytes APIC chunk)")
        return True
    except Exception as e:
        print(f"   (cover embed failed: {e})")
        return False


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bars", type=int, default=8)
    p.add_argument("--bpm", type=int, default=120)
    p.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    p.add_argument("--generate-only", action="store_true")
    p.add_argument("--full-pool", action="store_true", help="Build 11GB pool with raw audio (needed for training)")
    p.add_argument("--closed-loop", type=int, default=0, help="N segments for closed-loop critic selection (e.g. 8)")
    p.add_argument("--multi-stream", action="store_true", help="Use 6-stream generation (sub/drums/harmonic/texture/noise/air)")
    p.add_argument("--target-centroid", type=float, default=None, help="Target spectral centroid (0-1)")
    p.add_argument("--target-energy", type=float, default=None, help="Target energy level (0-1)")
    p.add_argument("--target-flux", type=float, default=None, help="Target spectral flux (0-1)")
    p.add_argument("--target-low", type=float, default=None, help="Target low-freq ratio (0-1)")
    p.add_argument("--noise-inject", type=float, default=0.0, help="Noise injection std (0-0.5)")
    p.add_argument("--train-multi", action="store_true", help="Train MultiNavigator (6-stream model)")
    p.add_argument("--pool", type=str, default=None, help="Path to grain pool .npz (skips scan/build)")
    p.add_argument("--model", type=str, default=None, help="Path to trained model .pt (skips training)")
    p.add_argument("--visual", action="store_true", help="Enable visual/cover-art generation alongside audio")
    p.add_argument("--train-visual", action="store_true", help="Train the visual head (needs image pool)")
    p.add_argument("--vis-pool", type=str, default=None, help="Path to visual grain pool .npz (skips image scan)")
    args = p.parse_args()
    t0 = time.time()

    cache = args.pool if args.pool else (POOL_CACHE if args.full_pool else POOL_CACHE_LIGHT)

    if os.path.exists(cache):
        print(f"📦 Loading pool: {cache}")
        d = np.load(cache, allow_pickle=True)
        pool = {}
        for k in d.files:
            pool[k] = d[k].tolist() if k in ("trajectories", "micro_sources", "meso_sources", "macro_sources") else d[k]
        # dequantize INT16 audio back to float32 for synthesis/critic
        for k in ["micro_audio", "meso_audio", "macro_audio"]:
            if k in pool and pool[k].dtype == np.int16:
                pool[k] = pool[k].astype(np.float32) / 32767.0
                print(f"   dequantized {k} -> float32")
    else:
        files = scan_audio(SCAN_DIRS)
        print(f"   {len(files)} files")
        if not files: return
        save_audio = args.full_pool
        pool = extract_all(files, save_audio=save_audio)
        save_args = dict(
            micro_feats=pool["micro_feats"],
            meso_feats=pool["meso_feats"],
            macro_feats=pool["macro_feats"],
            micro_sources=np.array(pool["micro_sources"], dtype=object),
            meso_sources=np.array(pool["meso_sources"], dtype=object),
            macro_sources=np.array(pool["macro_sources"], dtype=object),
            trajectories=np.array(pool["trajectories"], dtype=object),
        )
        if save_audio:
            save_args["micro_audio"] = pool["micro_audio"]
            save_args["meso_audio"] = pool["meso_audio"]
            save_args["macro_audio"] = pool["macro_audio"]
        np.savez(cache, **save_args)
        print(f"💾 Saved: {os.path.getsize(cache)/1e9:.1f} GB")

    print(f"\n📊 μ={len(pool['micro_feats'])} σ={len(pool['meso_feats'])} Ω={len(pool['macro_feats'])}")
    print(f"   Audio in pool: {'yes' if 'micro_audio' in pool else 'no (on-demand)'}")

    clusters = build_clusters(pool)
    engine = GranularEngine(pool, clusters)
    critic = STFTCritic(pool, engine=engine)

    pairs = build_training_pairs(pool, clusters)

    # ── Visual pool (optional) ──────────────────────────────────────
    vis_pool = None
    vis_clusters = None
    vis_engine = None
    vis_pairs = []
    if args.visual or args.train_visual:
        vis_cache = args.vis_pool if args.vis_pool else VIS_POOL_CACHE
        if os.path.exists(vis_cache):
            print(f"📦 Loading visual pool: {vis_cache}")
            d = np.load(vis_cache, allow_pickle=True)
            vis_pool = {k: d[k] for k in d.files}
        else:
            image_files = scan_images(VISUAL_SCAN_DIRS)
            print(f"   {len(image_files)} images found")
            if image_files:
                extractor = VisualGrainExtractor()
                vis_pool = extractor.extract(image_files)
                np.savez(vis_cache,
                    micro_feats=vis_pool["micro_feats"],
                    meso_feats=vis_pool["meso_feats"],
                    macro_feats=vis_pool["macro_feats"],
                    micro_patches=vis_pool["micro_patches"],
                    meso_patches=vis_pool["meso_patches"],
                    macro_patches=vis_pool["macro_patches"])
                print(f"💾 Visual pool saved: {os.path.getsize(vis_cache)/1e6:.1f} MB")
        if vis_pool is not None and len(vis_pool.get("micro_feats", [])) > 0:
            vis_clusters = build_visual_clusters(vis_pool)
            vis_engine = VisualEngine(vis_pool, vis_clusters)
            vis_pairs = build_visual_training_pairs(vis_pool, vis_clusters)

    if not args.generate_only and pairs:
        if args.train_multi:
            model_ms = MultiNavigator().to(DEVICE)
            if os.path.exists(MODEL_MULTI_CACHE):
                print(f"📦 Loading multi-model (strict=False for attractor init): {MODEL_MULTI_CACHE}")
                model_ms.load_state_dict(torch.load(MODEL_MULTI_CACHE, map_location=DEVICE, weights_only=False)["model_state"], strict=False)
            model_ms = train_multi(model_ms, pairs, n_steps=args.train_steps,
                vis_pairs=vis_pairs if args.train_visual else None)
            torch.save({"model_state": model_ms.state_dict()}, MODEL_MULTI_CACHE)
        else:
            model = Navigator().to(DEVICE)
            if os.path.exists(MODEL_CACHE):
                print(f"📦 Loading model: {MODEL_CACHE}")
                model.load_state_dict(torch.load(MODEL_CACHE, map_location=DEVICE, weights_only=False)["model_state"])
            model = train(model, pairs, n_steps=args.train_steps)
            torch.save({"model_state": model.state_dict()}, MODEL_CACHE)

    target_stats = None
    if any(x is not None for x in [args.target_centroid, args.target_energy, args.target_flux, args.target_low]):
        target_stats = [
            args.target_centroid or 0.5,
            args.target_energy or 0.5,
            args.target_flux or 0.5,
            args.target_low or 0.5,
        ]

    multi_model_path = args.model if args.model else MODEL_MULTI_CACHE
    model_ms = MultiNavigator().to(DEVICE)
    if os.path.exists(multi_model_path):
        print(f"📦 Loading multi-model: {multi_model_path}")
        model_ms.load_state_dict(torch.load(multi_model_path, map_location=DEVICE, weights_only=False)["model_state"], strict=False)

    single_model_path = args.model if args.model else MODEL_CACHE
    model = Navigator().to(DEVICE)
    if os.path.exists(single_model_path):
        print(f"📦 Loading model: {single_model_path}")
        model.load_state_dict(torch.load(single_model_path, map_location=DEVICE, weights_only=False)["model_state"])

    n_sec = args.bars * 4 * 60.0 / args.bpm
    image = None

    if args.multi_stream:
        stereo, score, image = generate_multi(model_ms, engine, critic, pool, n_seconds=n_sec,
            seed=args.seed, temp=args.temperature, bars=args.bars, bpm=args.bpm,
            target_stats=target_stats, noise_inject=args.noise_inject,
            vis_engine=vis_engine if args.visual else None,
            vis_pool=vis_pool if args.visual else None)
    else:
        stereo, score = generate(model, engine, critic, pool, n_seconds=n_sec,
            seed=args.seed, temp=args.temperature, bars=args.bars, bpm=args.bpm,
            closed_loop=args.closed_loop, target_stats=target_stats,
            noise_inject=args.noise_inject)
        image = None

    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT, f"granular_{args.bars}bars_{ts}.wav")
    stereo_16 = np.clip(stereo.T * 32767, -32767, 32767).astype(np.int16)
    write(out, SR, stereo_16)
    dur = stereo.shape[1]/SR
    rms = np.sqrt(np.mean(stereo**2))
    print(f"\n✅ {out} ({dur:.1f}s stereo)")
    print(f"   RMS: {rms:.3f} ({20*np.log10(rms+1e-10):.1f} dBFS) Peak: {np.max(np.abs(stereo)):.3f}")
    print(f"   Critic: {score:.3f}")
    if image is not None:
        img_out = os.path.join(OUT, f"cover_{args.bars}bars_{ts}.png")
        image.save(img_out)
        print(f"🖼️ ✅ {img_out}")
        attach_cover_to_wav(out, image)
    print(f"⏱️ {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
