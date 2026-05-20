import os
import time
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
SAMPLE_RATE    = 16000   
DURASI_REKAM   = 2      
JUMLAH_SAMPEL  = 25      
DATASET_FOLDER = "dataset"

KATA_GEJALA = [
    "demam", "batuk", "pusing", "mual", "sesak",
    "nyeri", "lemas", "bersin", "diare", "muntah",
    "gatal", "bengkak", "panas", "keringat", "berdebar",
    "kebas", "kram", "sakit", "tenggorokan", "hidung"
]

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def bersihkan_layar():
    os.system("cls" if os.name == "nt" else "clear")


def hitung_sampel_ada(folder_kelas):
    """Hitung file .wav yang sudah ada di folder kelas."""
    if not os.path.exists(folder_kelas):
        return 0
    return len([f for f in os.listdir(folder_kelas) if f.endswith(".wav")])


def rekam_audio(durasi, sample_rate):
    """Rekam audio dari mikrofon, return numpy array."""
    audio = sd.rec(
        int(durasi * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )
    sd.wait()  
    return audio.flatten()


def simpan_wav(audio, path, sample_rate):
    """Simpan audio ke file .wav."""
    wavfile.write(path, sample_rate, audio)


def tampilkan_progress_bar(selesai, total, lebar=30):
    """Tampilkan progress bar sederhana."""
    persen = selesai / total
    terisi = int(lebar * persen)
    bar = "█" * terisi + "░" * (lebar - terisi)
    return f"[{bar}] {selesai}/{total}"


def rekam_satu_kelas(kata, folder_kelas, mulai_dari, jumlah_target):
    """
    Loop rekam sampel untuk satu kata sampai jumlah_target tercapai.
    Bisa dilanjutkan dari sampel yang sudah ada (resume).
    """
    os.makedirs(folder_kelas, exist_ok=True)
    nomor = mulai_dari + 1

    while nomor <= jumlah_target:
        bersihkan_layar()
        print("=" * 50)
        print(f"  DOKTERKU — Perekaman Dataset")
        print("=" * 50)
        print(f"\n  Kata saat ini  : \"{kata.upper()}\"")
        print(f"  Progress       : {tampilkan_progress_bar(nomor - 1, jumlah_target)}")
        print(f"  Sampel ke      : {nomor} dari {jumlah_target}\n")
        print("─" * 50)
        print("  [ENTER]  = mulai rekam")
        print("  [s]      = skip / lewati sampel ini")
        print("  [q]      = keluar & lanjutkan nanti")
        print("─" * 50)

        pilihan = input("\n  Siap? ").strip().lower()

        if pilihan == "q":
            print("\n  Rekaman dihentikan. Progress tersimpan.")
            return "keluar"

        if pilihan == "s":
            print("  Sampel dilewati.")
            time.sleep(0.5)
            nomor += 1
            continue

        for i in range(3, 0, -1):
            print(f"\r  Bersiap... {i}", end="", flush=True)
            time.sleep(0.7)

        print(f"\r  ● REKAM — Ucapkan \"{kata}\"   ", flush=True)
        audio = rekam_audio(DURASI_REKAM, SAMPLE_RATE)
        print(f"  ✓ Rekaman selesai.", flush=True)

        nama_file = f"{kata}_{nomor:03d}.wav"
        path_file = os.path.join(folder_kelas, nama_file)
        simpan_wav(audio, path_file, SAMPLE_RATE)

        print(f"  Disimpan: {path_file}")
        time.sleep(0.8)
        nomor += 1

    return "selesai"


def tampilkan_status_semua():
    """Tampilkan ringkasan progress semua kelas."""
    bersihkan_layar()
    print("=" * 50)
    print("  DOKTERKU — Status Dataset")
    print("=" * 50)
    total_sampel = 0
    for kata in KATA_GEJALA:
        folder = os.path.join(DATASET_FOLDER, kata)
        ada = hitung_sampel_ada(folder)
        total_sampel += ada
        status = "✓ SELESAI" if ada >= JUMLAH_SAMPEL else f"{ada}/{JUMLAH_SAMPEL}"
        bar_mini = "█" * min(ada, JUMLAH_SAMPEL) + "░" * max(0, JUMLAH_SAMPEL - ada)

        bar_mini = bar_mini[:10]
        print(f"  {kata:<15} {bar_mini}  {status}")
    print("─" * 50)
    print(f"  Total sampel   : {total_sampel} / {len(KATA_GEJALA) * JUMLAH_SAMPEL}")
    print("=" * 50)


def menu_utama():
    """Tampilkan menu pilih kata untuk direkam."""
    bersihkan_layar()
    print("=" * 50)
    print("  DOKTERKU — Pilih Kata yang Ingin Direkam")
    print("=" * 50)
    print()

    belum_selesai = []
    for i, kata in enumerate(KATA_GEJALA):
        folder = os.path.join(DATASET_FOLDER, kata)
        ada = hitung_sampel_ada(folder)
        status = "✓" if ada >= JUMLAH_SAMPEL else f"{ada}/{JUMLAH_SAMPEL}"
        print(f"  [{i+1:2d}] {kata:<15} {status}")
        if ada < JUMLAH_SAMPEL:
            belum_selesai.append(i)

    print()
    print("  [a]  = rekam semua kata yang belum selesai (otomatis)")
    print("  [s]  = lihat ringkasan status")
    print("  [q]  = keluar")
    print()

    pilihan = input("  Pilih nomor kata atau perintah: ").strip().lower()
    return pilihan, belum_selesai


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    print("\n  Selamat datang di DOKTERKU Dataset Recorder!")
    print(f"  Setiap kata akan direkam {JUMLAH_SAMPEL} kali selama {DURASI_REKAM} detik.\n")
    input("  Tekan ENTER untuk mulai...")

    while True:
        pilihan, belum_selesai = menu_utama()

        if pilihan == "q":
            print("\n  Sampai jumpa!\n")
            break

        elif pilihan == "s":
            tampilkan_status_semua()
            input("\n  Tekan ENTER untuk kembali ke menu...")

        elif pilihan == "a":
            if not belum_selesai:
                print("\n  Semua kata sudah selesai direkam!")
                input("  Tekan ENTER untuk kembali...")
                continue

            for idx in belum_selesai:
                kata = KATA_GEJALA[idx]
                folder_kelas = os.path.join(DATASET_FOLDER, kata)
                mulai_dari = hitung_sampel_ada(folder_kelas)

                print(f"\n  Mulai merekam: \"{kata}\" (lanjut dari sampel {mulai_dari + 1})")
                time.sleep(1)

                hasil = rekam_satu_kelas(kata, folder_kelas, mulai_dari, JUMLAH_SAMPEL)
                if hasil == "keluar":
                    break

        else:
            try:
                idx = int(pilihan) - 1
                if 0 <= idx < len(KATA_GEJALA):
                    kata = KATA_GEJALA[idx]
                    folder_kelas = os.path.join(DATASET_FOLDER, kata)
                    mulai_dari = hitung_sampel_ada(folder_kelas)

                    if mulai_dari >= JUMLAH_SAMPEL:
                        print(f"\n  Kata \"{kata}\" sudah lengkap ({JUMLAH_SAMPEL} sampel).")
                        input("  Tekan ENTER untuk kembali...")
                        continue

                    rekam_satu_kelas(kata, folder_kelas, mulai_dari, JUMLAH_SAMPEL)
                else:
                    print("  Nomor tidak valid.")
                    time.sleep(1)
            except ValueError:
                print("  Input tidak dikenali.")
                time.sleep(1)


if __name__ == "__main__":
    main()