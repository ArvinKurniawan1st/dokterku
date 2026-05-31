"""
DOKTERKU — Augmentasi Data Audio
==================================
Memperbanyak dataset dari 500 → 2.000 sampel tanpa rekam ulang.

Teknik augmentasi (4 versi per file asli):
  1. Gaussian noise    — tambah derau acak (simulasi noise mikrofon)
  2. Pitch shift up    — naikkan nada +2 semitone (suara lebih tinggi)
  3. Pitch shift down  — turunkan nada -2 semitone (suara lebih rendah)
  4. Time stretch slow — perlambat 0.9× (ucapan lebih lambat)
  5. Time stretch fast — percepat 1.1× (ucapan lebih cepat)

Dari 1 file asli → 5 file augmentasi = total 6× lipat
500 file asli × 6 = 3.000 file

Output disimpan di folder yang SAMA dengan aslinya:
  dataset/demam/demam_001.wav         ← asli (tidak diubah)
  dataset/demam/demam_001_aug_noise.wav
  dataset/demam/demam_001_aug_pu.wav
  dataset/demam/demam_001_aug_pd.wav
  dataset/demam/demam_001_aug_slow.wav
  dataset/demam/demam_001_aug_fast.wav

Instalasi:
    pip install librosa numpy scipy soundfile tqdm

Cara pakai:
    python augmentasi_data.py
"""

import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
DATASET_FOLDER  = "dataset"
SAMPLE_RATE     = 16000
DURASI_MAX      = 2.0          # detik — sama dengan ekstraksi_mfcc.py

# Parameter augmentasi
NOISE_FACTOR    = 0.004        # amplitudo noise Gaussian (0.002–0.008)
PITCH_UP        = 2            # semitone naik
PITCH_DOWN      = -2           # semitone turun
STRETCH_SLOW    = 0.9          # 0.9× = lebih lambat
STRETCH_FAST    = 1.1          # 1.1× = lebih cepat

KATA_GEJALA = [
    "demam", "batuk", "pusing", "mual", "sesak",
    "nyeri", "lemas", "bersin", "diare", "muntah",
    "gatal", "bengkak", "panas", "keringat", "berdebar",
    "kebas", "kram", "sakit", "tenggorokan", "hidung"
]

# ─────────────────────────────────────────────
# LOAD + PAD/TRIM (sama persis dengan ekstraksi)
# ─────────────────────────────────────────────

def load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=30)
    target   = int(DURASI_MAX * SAMPLE_RATE)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)), mode="constant")
    else:
        audio = audio[:target]
    return audio.astype(np.float32)


def simpan(audio, path):
    """Simpan audio float32 ke .wav 16kHz."""
    # Clip ke [-1, 1] sebelum simpan agar tidak clipping
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")


# ─────────────────────────────────────────────
# FUNGSI AUGMENTASI
# ─────────────────────────────────────────────

def aug_noise(audio, factor=NOISE_FACTOR):
    """
    Tambahkan Gaussian noise.
    Simulasi: noise mikrofon, lingkungan tidak sempurna.
    factor kecil (0.002–0.008) = tidak terdengar oleh manusia
    tapi cukup membuat model lebih robust.
    """
    noise = np.random.normal(0, factor, len(audio)).astype(np.float32)
    return audio + noise


def aug_pitch_shift(audio, n_steps, sr=SAMPLE_RATE):
    """
    Geser nada ±n_steps semitone tanpa mengubah durasi.
    Simulasi: variasi tinggi suara antar speaker.
    n_steps = +2 → suara lebih tinggi (perempuan)
    n_steps = -2 → suara lebih rendah (laki-laki berat)
    """
    return librosa.effects.pitch_shift(
        audio, sr=sr, n_steps=n_steps
    ).astype(np.float32)


def aug_time_stretch(audio, rate, sr=SAMPLE_RATE):
    """
    Ubah kecepatan ucapan tanpa mengubah nada.
    rate < 1.0 → lebih lambat (0.9 = 90% kecepatan asli)
    rate > 1.0 → lebih cepat (1.1 = 110% kecepatan asli)
    Setelah stretch, pad/trim kembali ke DURASI_MAX.
    """
    stretched = librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)
    target    = int(DURASI_MAX * sr)
    if len(stretched) < target:
        stretched = np.pad(stretched, (0, target - len(stretched)), mode="constant")
    else:
        stretched = stretched[:target]
    return stretched


# ─────────────────────────────────────────────
# CEK APAKAH FILE AUG SUDAH ADA
# ─────────────────────────────────────────────

def sudah_ada(path_asli):
    """
    Return True jika semua 5 file augmentasi untuk path_asli sudah ada.
    Berguna untuk resume jika script dihentikan di tengah jalan.
    """
    base = path_asli.replace(".wav", "")
    suffixes = ["_aug_noise", "_aug_pu", "_aug_pd", "_aug_slow", "_aug_fast"]
    return all(os.path.exists(base + s + ".wav") for s in suffixes)


# ─────────────────────────────────────────────
# PROSES SATU KELAS
# ─────────────────────────────────────────────

def augmentasi_kelas(kata, folder_kelas, skip_existing=True):
    """
    Augmentasi semua file .wav asli di folder_kelas.
    File asli = file yang TIDAK mengandung '_aug_' di namanya.
    """
    semua_files = sorted(os.listdir(folder_kelas))
    # Ambil hanya file asli (bukan hasil augmentasi sebelumnya)
    files_asli  = [f for f in semua_files
                   if f.endswith(".wav") and "_aug_" not in f]

    if not files_asli:
        print(f"  [!] {kata}: tidak ada file asli ditemukan")
        return 0, 0

    diproses = 0
    diskip   = 0

    for fname in tqdm(files_asli, desc=f"  {kata:<15}", ncols=58, leave=False):
        path_asli = os.path.join(folder_kelas, fname)
        base_name = fname.replace(".wav", "")

        # Resume: skip jika semua aug sudah ada
        if skip_existing and sudah_ada(path_asli):
            diskip += 1
            continue

        try:
            audio = load_audio(path_asli)

            # ── 1. Gaussian noise ──
            simpan(
                aug_noise(audio),
                os.path.join(folder_kelas, base_name + "_aug_noise.wav")
            )

            # ── 2. Pitch shift up (+2 semitone) ──
            simpan(
                aug_pitch_shift(audio, PITCH_UP),
                os.path.join(folder_kelas, base_name + "_aug_pu.wav")
            )

            # ── 3. Pitch shift down (-2 semitone) ──
            simpan(
                aug_pitch_shift(audio, PITCH_DOWN),
                os.path.join(folder_kelas, base_name + "_aug_pd.wav")
            )

            # ── 4. Time stretch slow (0.9×) ──
            simpan(
                aug_time_stretch(audio, STRETCH_SLOW),
                os.path.join(folder_kelas, base_name + "_aug_slow.wav")
            )

            # ── 5. Time stretch fast (1.1×) ──
            simpan(
                aug_time_stretch(audio, STRETCH_FAST),
                os.path.join(folder_kelas, base_name + "_aug_fast.wav")
            )

            diproses += 1

        except Exception as e:
            print(f"\n  [!] Gagal: {fname} → {e}")

    return diproses, diskip


# ─────────────────────────────────────────────
# HITUNG STATISTIK DATASET
# ─────────────────────────────────────────────

def hitung_statistik(sebelum=True):
    """Hitung total file asli dan augmentasi per kelas."""
    total_asli = 0
    total_aug  = 0
    print(f"\n  {'Kelas':<15} {'Asli':>6} {'Aug':>6} {'Total':>7}")
    print("  " + "-" * 38)
    for kata in KATA_GEJALA:
        folder = os.path.join(DATASET_FOLDER, kata)
        if not os.path.exists(folder):
            print(f"  {kata:<15} {'—':>6}")
            continue
        files    = [f for f in os.listdir(folder) if f.endswith(".wav")]
        asli     = len([f for f in files if "_aug_" not in f])
        aug      = len([f for f in files if "_aug_" in f])
        total_asli += asli
        total_aug  += aug
        print(f"  {kata:<15} {asli:>6} {aug:>6} {asli+aug:>7}")
    print("  " + "-" * 38)
    print(f"  {'TOTAL':<15} {total_asli:>6} {total_aug:>6} {total_asli+total_aug:>7}")
    return total_asli, total_aug


# ─────────────────────────────────────────────
# HAPUS SEMUA FILE AUGMENTASI (reset)
# ─────────────────────────────────────────────

def hapus_augmentasi():
    """Hapus semua file _aug_*.wav dari seluruh dataset."""
    total = 0
    for kata in KATA_GEJALA:
        folder = os.path.join(DATASET_FOLDER, kata)
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if "_aug_" in f and f.endswith(".wav"):
                os.remove(os.path.join(folder, f))
                total += 1
    print(f"  ✓ {total} file augmentasi dihapus.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 58)
    print("  DOKTERKU — Augmentasi Data Audio")
    print("=" * 58)
    print(f"  Teknik  : noise | pitch ±{PITCH_UP} semitone | stretch {STRETCH_SLOW}×/{STRETCH_FAST}×")
    print(f"  Output  : 1 file asli → 5 file augmentasi → 6× lipat")
    print("=" * 58)

    if not os.path.exists(DATASET_FOLDER):
        print(f"\n  [!] Folder '{DATASET_FOLDER}/' tidak ditemukan.")
        print("      Jalankan rekam_dataset.py terlebih dahulu.\n")
        return

    print("\n  [1] Augmentasi semua kelas (skip yang sudah ada)")
    print("  [2] Augmentasi ulang dari awal (hapus & buat ulang semua)")
    print("  [3] Lihat statistik dataset saat ini")
    print("  [4] Hapus semua file augmentasi")
    print("  [q] Keluar")
    pilihan = input("\n  Pilih: ").strip().lower()

    if pilihan == "3":
        print("\n  Statistik dataset saat ini:")
        hitung_statistik()
        return

    if pilihan == "4":
        konfirm = input("  Yakin hapus semua file augmentasi? (y/n): ").strip().lower()
        if konfirm == "y":
            hapus_augmentasi()
        return

    if pilihan not in ["1", "2"]:
        print("  Keluar.")
        return

    # Pilihan 2: hapus dulu
    if pilihan == "2":
        print("\n  Menghapus file augmentasi lama...")
        hapus_augmentasi()

    skip_existing = (pilihan == "1")

    # Statistik sebelum
    print("\n  Statistik SEBELUM augmentasi:")
    n_asli, n_aug_before = hitung_statistik()

    print(f"\n  Mulai augmentasi {n_asli} file asli...\n")

    total_diproses = 0
    total_diskip   = 0

    for kata in KATA_GEJALA:
        folder_kelas = os.path.join(DATASET_FOLDER, kata)
        if not os.path.exists(folder_kelas):
            continue
        diproses, diskip = augmentasi_kelas(kata, folder_kelas, skip_existing)
        total_diproses  += diproses
        total_diskip    += diskip
        status = f"✓ {diproses} file diproses"
        if diskip:
            status += f" ({diskip} diskip, sudah ada)"
        print(f"  {kata:<15} {status}")

    # Statistik sesudah
    print("\n" + "─" * 58)
    print("  Statistik SETELAH augmentasi:")
    n_asli2, n_aug_after = hitung_statistik()

    total_sebelum = n_asli + n_aug_before
    total_sesudah = n_asli2 + n_aug_after
    print(f"\n  Total file diproses : {total_diproses}")
    print(f"  Total file diskip   : {total_diskip}")
    print(f"  Dataset sebelum     : {total_sebelum} file")
    print(f"  Dataset sesudah     : {total_sesudah} file")
    print(f"  Faktor lipat        : {total_sesudah/max(n_asli,1):.1f}×")

    print("\n" + "=" * 58)
    print("  SELESAI!")
    print("  Langkah berikutnya:")
    print("  1. python ekstraksi_mfcc.py   (ekstrak ulang semua fitur)")
    print("  2. python training_model.py   (training ulang model)")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    main()