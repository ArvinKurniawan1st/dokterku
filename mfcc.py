import os
import numpy as np
import librosa
from tqdm import tqdm

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
DATASET_FOLDER = "dataset"
OUTPUT_FILE    = "features.npz"
INFO_FILE      = "mfcc_info.txt"

SAMPLE_RATE   = 16000
DURASI_MAX    = 2.0
PRE_EMPH_COEF = 0.97
FRAME_LENGTH  = 0.025
FRAME_STEP    = 0.010
N_MFCC        = 40
N_FFT         = 512
N_MELS        = 128

KATA_GEJALA = [
    "demam", "batuk", "pusing", "mual", "sesak",
    "nyeri", "lemas", "bersin", "diare", "muntah",
    "gatal", "bengkak", "panas", "keringat", "berdebar",
    "kebas", "kram", "sakit", "tenggorokan", "hidung"
]

# ─────────────────────────────────────────────
# STEP 1 — LOAD & PAD/TRIM
# ─────────────────────────────────────────────

def load_audio(path, sr=SAMPLE_RATE, durasi_max=DURASI_MAX):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    # Trim silence lebih agresif (top_db=30) untuk buang silence panjang
    audio, _ = librosa.effects.trim(audio, top_db=30)
    target_len = int(durasi_max * sr)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    else:
        audio = audio[:target_len]
    return audio

# ─────────────────────────────────────────────
# STEP 2 — PRE-EMPHASIS
# ─────────────────────────────────────────────

def pre_emphasis(audio, coef=PRE_EMPH_COEF):
    return np.append(audio[0], audio[1:] - coef * audio[:-1])

# ─────────────────────────────────────────────
# STEP 3 — MFCC + DELTA + DELTA-DELTA
# ─────────────────────────────────────────────

def ekstrak_mfcc_delta(audio, sr=SAMPLE_RATE):
    hop_length = int(FRAME_STEP * sr)
    win_length = int(FRAME_LENGTH * sr)

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=hop_length,
        win_length=win_length, window="hamming",
        n_mels=N_MELS, fmin=0, fmax=sr // 2,
    )
    delta  = librosa.feature.delta(mfcc, order=1, width=9)
    delta2 = librosa.feature.delta(mfcc, order=2, width=9)
    return np.vstack([mfcc, delta, delta2])  # (120, T)

# ─────────────────────────────────────────────
# STEP 4 — AGREGASI STATISTIK → VEKTOR 1D
# ─────────────────────────────────────────────

def agregasi_statistik(mfcc_combined):
    mean_vec = mfcc_combined.mean(axis=1)  # (120,)
    std_vec  = mfcc_combined.std(axis=1)   # (120,)
    return np.concatenate([mean_vec, std_vec])  # (240,)

# ─────────────────────────────────────────────
# PIPELINE PENUH — 1 FILE AUDIO
# ─────────────────────────────────────────────

def proses_satu_file(path):
    audio      = load_audio(path)
    audio      = pre_emphasis(audio)
    mfcc_delta = ekstrak_mfcc_delta(audio)
    fitur      = agregasi_statistik(mfcc_delta)
    return fitur

# ─────────────────────────────────────────────
# PROSES SELURUH DATASET
# ─────────────────────────────────────────────

def proses_dataset():
    X, y, file_paths, gagal = [], [], [], []
    label_to_idx = {kata: i for i, kata in enumerate(KATA_GEJALA)}

    print("\n" + "=" * 55)
    print("  DOKTERKU — Pipeline Ekstraksi MFCC (FIXED)")
    print("=" * 55)
    print(f"  MFCC : {N_MFCC} koef + delta + delta-delta")
    print(f"  Fitur: 240 dimensi (mean+std, tanpa Z-score per utterance)")
    print("=" * 55 + "\n")

    for kata in KATA_GEJALA:
        folder_kelas = os.path.join(DATASET_FOLDER, kata)
        if not os.path.exists(folder_kelas):
            print(f"  [!] Folder tidak ditemukan: {folder_kelas}")
            continue

        wav_files = sorted([f for f in os.listdir(folder_kelas)
                            if f.endswith(".wav")])
        if not wav_files:
            print(f"  [!] Tidak ada .wav di: {folder_kelas}")
            continue

        for fname in tqdm(wav_files, desc=f"  {kata:<15}", ncols=55, leave=False):
            path = os.path.join(folder_kelas, fname)
            try:
                fitur = proses_satu_file(path)
                X.append(fitur)
                y.append(label_to_idx[kata])
                file_paths.append(path)
            except Exception as e:
                gagal.append((path, str(e)))

        print(f"  ✓ {kata:<15} {len(wav_files)} sampel selesai")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y, file_paths, gagal

# ─────────────────────────────────────────────
# VALIDASI FITUR
# ─────────────────────────────────────────────

def validasi_fitur(X, y):
    print("\n" + "─" * 55)
    print("  VALIDASI FITUR")
    print("─" * 55)

    std_per_fitur = X.std(axis=0)
    n_konstan     = (std_per_fitur < 1e-6).sum()

    print(f"  Rentang nilai      : [{X.min():.4f}, {X.max():.4f}]")
    print(f"  Std per fitur mean : {std_per_fitur.mean():.4f}")
    print(f"  Fitur konstan      : {n_konstan} / {X.shape[1]}")

    # Cek jarak antar centroid
    label_names  = KATA_GEJALA
    centroids    = np.array([X[y == i].mean(axis=0)
                             for i in range(len(label_names))])
    from numpy.linalg import norm
    jarak_list = []
    for i in range(len(label_names)):
        for j in range(i+1, len(label_names)):
            jarak_list.append(norm(centroids[i] - centroids[j]))
    print(f"  Jarak centroid min : {min(jarak_list):.4f}")
    print(f"  Jarak centroid max : {max(jarak_list):.4f}")

    if n_konstan == X.shape[1]:
        print("\n  ❌ FITUR MASIH NOL SEMUA — ada masalah lain, hubungi developer")
        return False
    elif n_konstan > 10:
        print(f"\n  ⚠️  {n_konstan} fitur masih konstan — mungkin beberapa audio masih silence")
    else:
        print("\n  ✅ Fitur terlihat sehat dan bervariasi!")
        print("     Lanjutkan ke: python training_model.py")
    return True

# ─────────────────────────────────────────────
# SIMPAN & LAPORAN
# ─────────────────────────────────────────────

def simpan_hasil(X, y):
    np.savez_compressed(OUTPUT_FILE, X=X, y=y,
                        label_names=np.array(KATA_GEJALA))
    print(f"\n  ✓ Disimpan ke: {OUTPUT_FILE}")
    print(f"    X shape : {X.shape}")
    print(f"    y shape : {y.shape}")

def demo_satu_file():
    print("\n" + "=" * 55)
    print("  MODE DEMO — Verifikasi Pipeline 1 File")
    print("=" * 55)

    path_demo = None
    for kata in KATA_GEJALA:
        folder = os.path.join(DATASET_FOLDER, kata)
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith(".wav")]
            if files:
                path_demo = os.path.join(folder, sorted(files)[0])
                break

    if path_demo is None:
        print("  [!] Tidak ada file .wav ditemukan.")
        return

    print(f"\n  File : {path_demo}\n")

    audio_raw  = load_audio(path_demo)
    print(f"  [1] Load + trim    : {len(audio_raw)} sampel "
          f"({len(audio_raw)/SAMPLE_RATE:.2f}s)")

    audio_pre  = pre_emphasis(audio_raw)
    print(f"  [2] Pre-emphasis   : coef={PRE_EMPH_COEF}")

    mfcc_delta = ekstrak_mfcc_delta(audio_pre)
    print(f"  [3] MFCC+Δ+ΔΔ     : shape={mfcc_delta.shape}")
    print(f"      Min={mfcc_delta.min():.3f}  Max={mfcc_delta.max():.3f}  "
          f"Std={mfcc_delta.std():.3f}")

    fitur = agregasi_statistik(mfcc_delta)
    print(f"  [4] Agregasi       : shape={fitur.shape}")
    print(f"      Min={fitur.min():.4f}  Max={fitur.max():.4f}  "
          f"Std={fitur.std():.4f}")
    print(f"      mean_vec[:3] = {fitur[:3]}")
    print(f"      std_vec[:3]  = {fitur[120:123]}")

    if fitur.std() < 1e-4:
        print("\n  ⚠️  Fitur masih flat — audio mungkin hanya silence")
    else:
        print("\n  ✅ Pipeline OK — fitur bervariasi, siap diproses")
    print("=" * 55)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n  DOKTERKU — Ekstraksi MFCC")
    print("  [1] Demo pipeline (1 file)")
    print("  [2] Proses seluruh dataset")
    print("  [q] Keluar")
    pilihan = input("\n  Pilih: ").strip().lower()

    if pilihan == "1":
        demo_satu_file()

    elif pilihan == "2":
        X, y, _, gagal = proses_dataset()
        if len(X) == 0:
            print("\n  [!] Tidak ada fitur berhasil diekstrak.")
            return

        valid = validasi_fitur(X, y)
        if valid:
            simpan_hasil(X, y)
            if gagal:
                print(f"\n  ⚠️  {len(gagal)} file gagal:")
                for p, e in gagal:
                    print(f"     {p}: {e}")
            print("\n  Selesai! Jalankan: python training_model.py")
    else:
        print("  Keluar.")

if __name__ == "__main__":
    main()