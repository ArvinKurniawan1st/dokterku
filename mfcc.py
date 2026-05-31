import os
import json
import numpy as np
import librosa
from tqdm import tqdm

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
DATASET_FOLDER = "dataset"
OUTPUT_FILE    = "features.npz"
INFO_FILE      = "mfcc_info.txt"

# Audio
SAMPLE_RATE    = 16000   # Hz
DURASI_MAX     = 2.0     # detik (potong/pad ke panjang ini)

# Pre-emphasis
PRE_EMPH_COEF  = 0.97    # koefisien pre-emphasis standar

# Framing
FRAME_LENGTH   = 0.025   # 25 ms
FRAME_STEP     = 0.010   # 10 ms (overlap 15 ms)

# MFCC
N_MFCC        = 40       # jumlah koefisien
N_FFT         = 512      # ukuran FFT
N_MELS        = 128      # jumlah mel filterbank

KATA_GEJALA = [
    "demam", "batuk", "pusing", "mual", "sesak",
    "nyeri", "lemas", "bersin", "diare", "muntah",
    "gatal", "bengkak", "panas", "keringat", "berdebar",
    "kebas", "kram", "sakit", "tenggorokan", "hidung"
]

# ─────────────────────────────────────────────
# STEP 1 — LOAD & PAD/TRIM AUDIO
# ─────────────────────────────────────────────

def load_audio(path, sr=SAMPLE_RATE, durasi_max=DURASI_MAX):
    """
    Load file .wav, resample ke sr, lalu:
    - Trim silence di awal/akhir
    - Pad dengan nol jika terlalu pendek
    - Potong jika terlalu panjang
    """
    audio, _ = librosa.load(path, sr=sr, mono=True)

    # Trim silence (top_db=20 dB threshold)
    audio, _ = librosa.effects.trim(audio, top_db=30)

    # Target panjang dalam sampel
    target_len = int(durasi_max * sr)

    if len(audio) < target_len:
        # Pad kanan dengan nol
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    else:
        # Potong ke target_len
        audio = audio[:target_len]

    return audio


# ─────────────────────────────────────────────
# STEP 2 — PRE-EMPHASIS
# ─────────────────────────────────────────────

def pre_emphasis(audio, coef=PRE_EMPH_COEF):
    """
    Filter pre-emphasis: y[t] = x[t] - coef * x[t-1]
    Tujuan: memperkuat frekuensi tinggi yang melemah saat rekaman,
    membuat spektrum lebih rata → MFCC lebih stabil.
    """
    return np.append(audio[0], audio[1:] - coef * audio[:-1])


# ─────────────────────────────────────────────
# STEP 3 — MFCC + DELTA + DELTA-DELTA
# ─────────────────────────────────────────────

def ekstrak_mfcc_delta(audio, sr=SAMPLE_RATE):
    """
    Ekstrak MFCC 40 koefisien + delta + delta-delta menggunakan librosa.
    librosa secara internal melakukan:
      framing → hamming window → FFT → mel filterbank → log → DCT → MFCC

    Return shape: (120, T)
      - 40 MFCC statik
      - 40 delta (turunan pertama — velocity)
      - 40 delta-delta (turunan kedua — acceleration)
    T = jumlah frame
    """
    hop_length    = int(FRAME_STEP * sr)      # sampel per hop
    win_length    = int(FRAME_LENGTH * sr)    # sampel per frame

    # Ekstrak 40 MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=hop_length,
        win_length=win_length,
        window="hamming",
        n_mels=N_MELS,
        fmin=0,
        fmax=sr // 2,
    )
    # mfcc shape: (40, T)

    # Delta — turunan pertama (perubahan antar frame)
    delta = librosa.feature.delta(mfcc, order=1, width=9)

    # Delta-delta — turunan kedua (percepatan perubahan)
    delta2 = librosa.feature.delta(mfcc, order=2, width=9)

    # Gabungkan → (120, T)
    mfcc_combined = np.vstack([mfcc, delta, delta2])

    return mfcc_combined


# ─────────────────────────────────────────────
# STEP 4 — NORMALISASI Z-SCORE PER UTTERANCE
# ─────────────────────────────────────────────

def normalisasi_zscore(mfcc_combined):
    """
    Normalisasi Z-score per utterance (bukan per dataset).
    Setiap koefisien dinormalisasi independen:
      z = (x - mean) / (std + epsilon)

    Tujuan: menghilangkan perbedaan skala antar speaker dan kondisi rekaman.
    Shape input/output: (120, T) → (120, T)
    """
    epsilon = 1e-8  # hindari pembagian nol
    mean = mfcc_combined.mean(axis=1, keepdims=True)   # (120, 1)
    std  = mfcc_combined.std(axis=1, keepdims=True)    # (120, 1)
    return (mfcc_combined - mean) / (std + epsilon)


# ─────────────────────────────────────────────
# STEP 5 — AGREGASI STATISTIK → VEKTOR 1D
# ─────────────────────────────────────────────

def agregasi_statistik(mfcc_norm):
    """
    Ubah matriks (120, T) menjadi vektor 1D untuk input model klasifikasi.

    Strategi: mean + std per koefisien per frame
      - mean shape: (120,)
      - std  shape: (120,)
      → concat → (240,)

    Ini menangkap nilai rata-rata dan variabilitas tiap koefisien
    sepanjang durasi ucapan.
    """
    mean_vec = mfcc_norm.mean(axis=1)   # (120,)
    std_vec  = mfcc_norm.std(axis=1)    # (120,)
    return np.concatenate([mean_vec, std_vec])  # (240,)


# ─────────────────────────────────────────────
# PIPELINE PENUH — 1 FILE AUDIO
# ─────────────────────────────────────────────

def proses_satu_file(path):
    """
    Jalankan pipeline lengkap untuk satu file .wav.
    Return: vektor fitur numpy (240,)
    """
    audio        = load_audio(path)
    audio        = pre_emphasis(audio)
    mfcc_delta   = ekstrak_mfcc_delta(audio)
    mfcc_norm    = normalisasi_zscore(mfcc_delta)
    fitur        = agregasi_statistik(mfcc_norm)
    return fitur


# ─────────────────────────────────────────────
# PROSES SELURUH DATASET
# ─────────────────────────────────────────────

def proses_dataset():
    """
    Iterasi semua file di dataset/, ekstrak fitur, simpan ke .npz
    """
    X = []          # list fitur, tiap elemen (240,)
    y = []          # list label integer
    file_paths = [] # untuk debugging
    gagal = []      # file yang gagal diproses

    label_names = KATA_GEJALA
    label_to_idx = {kata: i for i, kata in enumerate(label_names)}

    print("\n" + "=" * 55)
    print("  DOKTERKU — Pipeline Ekstraksi MFCC")
    print("=" * 55)
    print(f"  Dataset folder : {DATASET_FOLDER}/")
    print(f"  Kelas          : {len(label_names)} kata")
    print(f"  MFCC           : {N_MFCC} koefisien + delta + delta-delta")
    print(f"  Vektor output  : 240 dimensi (mean + std)")
    print(f"  Output file    : {OUTPUT_FILE}")
    print("=" * 55 + "\n")

    for kata in label_names:
        folder_kelas = os.path.join(DATASET_FOLDER, kata)
        if not os.path.exists(folder_kelas):
            print(f"  [!] Folder tidak ditemukan: {folder_kelas}")
            continue

        wav_files = sorted([
            f for f in os.listdir(folder_kelas) if f.endswith(".wav")
        ])

        if not wav_files:
            print(f"  [!] Tidak ada file .wav di: {folder_kelas}")
            continue

        print(f"  Memproses '{kata}' ({len(wav_files)} file)...")

        for fname in tqdm(wav_files, desc=f"    {kata}", ncols=55, leave=False):
            path = os.path.join(folder_kelas, fname)
            try:
                fitur = proses_satu_file(path)
                X.append(fitur)
                y.append(label_to_idx[kata])
                file_paths.append(path)
            except Exception as e:
                gagal.append((path, str(e)))

        print(f"    ✓ {len(wav_files)} sampel selesai")

    # Konversi ke numpy array
    X = np.array(X, dtype=np.float32)   # (N, 240)
    y = np.array(y, dtype=np.int32)     # (N,)

    return X, y, label_names, file_paths, gagal


# ─────────────────────────────────────────────
# SIMPAN & LAPORAN
# ─────────────────────────────────────────────

def simpan_hasil(X, y, label_names):
    """Simpan fitur ke .npz"""
    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y,
        label_names=np.array(label_names)
    )
    print(f"\n  ✓ Fitur disimpan ke: {OUTPUT_FILE}")
    print(f"    X shape : {X.shape}  (sampel × dimensi fitur)")
    print(f"    y shape : {y.shape}  (label integer)")


def cetak_laporan(X, y, label_names, gagal):
    """Cetak dan simpan laporan ringkasan ekstraksi."""
    lines = []
    lines.append("=" * 55)
    lines.append("  LAPORAN EKSTRAKSI MFCC — DOKTERKU")
    lines.append("=" * 55)
    lines.append(f"  Total sampel berhasil : {len(X)}")
    lines.append(f"  Total sampel gagal    : {len(gagal)}")
    lines.append(f"  Dimensi fitur         : {X.shape[1]}")
    lines.append(f"    → 40 MFCC × (mean+std) = 80")
    lines.append(f"    → 40 Δ    × (mean+std) = 80")
    lines.append(f"    → 40 ΔΔ   × (mean+std) = 80")
    lines.append(f"    → Total                = 240")
    lines.append("")
    lines.append("  Distribusi per kelas:")
    lines.append("  " + "-" * 35)
    for i, kata in enumerate(label_names):
        count = int(np.sum(y == i))
        bar = "█" * count
        lines.append(f"  {kata:<15} {bar}  ({count} sampel)")
    lines.append("  " + "-" * 35)

    if gagal:
        lines.append("\n  File gagal diproses:")
        for path, err in gagal:
            lines.append(f"  - {path}: {err}")

    lines.append("=" * 55)
    report = "\n".join(lines)

    print("\n" + report)
    with open(INFO_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  ✓ Laporan disimpan ke: {INFO_FILE}")


def demo_satu_file():
    """
    Demo: proses 1 file pertama yang ditemukan dan print tiap step.
    Berguna untuk verifikasi pipeline sebelum proses dataset penuh.
    """
    print("\n" + "=" * 55)
    print("  MODE DEMO — Verifikasi Pipeline 1 File")
    print("=" * 55)

    # Cari file pertama
    path_demo = None
    for kata in KATA_GEJALA:
        folder = os.path.join(DATASET_FOLDER, kata)
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith(".wav")]
            if files:
                path_demo = os.path.join(folder, sorted(files)[0])
                break

    if path_demo is None:
        print("  [!] Tidak ada file .wav ditemukan di dataset/")
        print("      Jalankan rekam_dataset.py terlebih dahulu.")
        return

    print(f"\n  File : {path_demo}\n")

    # Step by step
    audio_raw = load_audio(path_demo)
    print(f"  [1] Load audio          : shape={audio_raw.shape}, "
          f"durasi={len(audio_raw)/SAMPLE_RATE:.2f}s, "
          f"min={audio_raw.min():.4f}, max={audio_raw.max():.4f}")

    audio_pre = pre_emphasis(audio_raw)
    print(f"  [2] Pre-emphasis        : shape={audio_pre.shape}, "
          f"coef={PRE_EMPH_COEF}")

    mfcc_delta = ekstrak_mfcc_delta(audio_pre)
    print(f"  [3] MFCC+Δ+ΔΔ          : shape={mfcc_delta.shape}  "
          f"(120 koefisien × {mfcc_delta.shape[1]} frame)")

    mfcc_norm = normalisasi_zscore(mfcc_delta)
    print(f"  [4] Normalisasi Z-score : shape={mfcc_norm.shape}, "
          f"mean≈{mfcc_norm.mean():.4f}, std≈{mfcc_norm.std():.4f}")

    fitur = agregasi_statistik(mfcc_norm)
    print(f"  [5] Agregasi statistik  : shape={fitur.shape}  ← vektor input model")

    print("\n  ✓ Pipeline berjalan normal!")
    print("=" * 55)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n  DOKTERKU — Ekstraksi Fitur MFCC")
    print("  ================================")
    print("  [1] Demo pipeline (1 file, verifikasi step-by-step)")
    print("  [2] Proses seluruh dataset")
    print("  [q] Keluar")
    pilihan = input("\n  Pilih: ").strip().lower()

    if pilihan == "1":
        demo_satu_file()

    elif pilihan == "2":
        X, y, label_names, file_paths, gagal = proses_dataset()
        if len(X) == 0:
            print("\n  [!] Tidak ada fitur yang berhasil diekstrak.")
            print("      Pastikan folder dataset/ sudah terisi file .wav.")
            return
        simpan_hasil(X, y, label_names)
        cetak_laporan(X, y, label_names, gagal)
        print("\n  Selesai! File features.npz siap digunakan untuk training.")
        print("  Cara load di script training:")
        print("    data = np.load('features.npz', allow_pickle=True)")
        print("    X = data['X']       # shape (N, 240)")
        print("    y = data['y']       # shape (N,)")
        print("    labels = data['label_names']")

    else:
        print("  Keluar.")


if __name__ == "__main__":
    main()