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


def dapatkan_nomor_yang_ada(folder_kelas, kata):
    """
    Kembalikan set nomor (integer) dari file .wav yang sudah ada.
    Contoh: {1, 2, 3, 11, 12} jika ada demam_001.wav, demam_002.wav, dst.
    """
    if not os.path.exists(folder_kelas):
        return set()
    nomor_ada = set()
    prefix = f"{kata}_"
    for f in os.listdir(folder_kelas):
        if f.endswith(".wav") and f.startswith(prefix):
            bagian = f[len(prefix):-4]   # ambil bagian nomor saja
            if bagian.isdigit():
                nomor_ada.add(int(bagian))
    return nomor_ada


def dapatkan_nomor_kosong(folder_kelas, kata, jumlah_target):
    """
    Kembalikan list nomor yang BELUM ada filenya, urut dari 1 s.d. jumlah_target.
    Inilah nomor-nomor yang perlu direkam.
    """
    nomor_ada = dapatkan_nomor_yang_ada(folder_kelas, kata)
    semua_nomor = set(range(1, jumlah_target + 1))
    nomor_kosong = sorted(semua_nomor - nomor_ada)
    return nomor_kosong


def hitung_sampel_ada(folder_kelas, kata):
    """Hitung file .wav yang sudah ada di folder kelas (hanya file bernama benar)."""
    return len(dapatkan_nomor_yang_ada(folder_kelas, kata))


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


def rekam_satu_kelas(kata, folder_kelas, jumlah_target):
    """
    Loop rekam sampel untuk satu kata.
    Hanya mengisi nomor yang BELUM ada filenya (gap filling).
    Tidak akan pernah menimpa file yang sudah ada.
    """
    os.makedirs(folder_kelas, exist_ok=True)

    nomor_kosong = dapatkan_nomor_kosong(folder_kelas, kata, jumlah_target)

    if not nomor_kosong:
        print(f"\n  Semua {jumlah_target} sampel untuk \"{kata}\" sudah lengkap.")
        input("  Tekan ENTER untuk kembali...")
        return "selesai"

    total_sudah_ada = jumlah_target - len(nomor_kosong)
    indeks_sesi = 0  # pointer ke dalam list nomor_kosong

    while indeks_sesi < len(nomor_kosong):
        nomor = nomor_kosong[indeks_sesi]
        sudah_selesai = total_sudah_ada + indeks_sesi

        bersihkan_layar()
        print("=" * 50)
        print(f"  DOKTERKU — Perekaman Dataset")
        print("=" * 50)
        print(f"\n  Kata saat ini  : \"{kata.upper()}\"")
        print(f"  Progress       : {tampilkan_progress_bar(sudah_selesai, jumlah_target)}")
        print(f"  Merekam nomor  : {nomor:03d}  (tersisa {len(nomor_kosong) - indeks_sesi} slot kosong)\n")
        print("─" * 50)
        print("  [ENTER]  = mulai rekam")
        print("  [s]      = skip / lewati nomor ini")
        print("  [q]      = keluar & lanjutkan nanti")
        print("─" * 50)

        pilihan = input("\n  Siap? ").strip().lower()

        if pilihan == "q":
            print("\n  Rekaman dihentikan. Progress tersimpan.")
            return "keluar"

        if pilihan == "s":
            print("  Sampel dilewati.")
            time.sleep(0.5)
            indeks_sesi += 1
            continue

        # Cek ulang apakah file sudah ada (antisipasi race condition / perubahan di luar program)
        nama_file = f"{kata}_{nomor:03d}.wav"
        path_file = os.path.join(folder_kelas, nama_file)

        if os.path.exists(path_file):
            print(f"\n  ⚠  File {nama_file} sudah ada — dilewati otomatis.")
            time.sleep(1)
            indeks_sesi += 1
            continue

        for i in range(3, 0, -1):
            print(f"\r  Bersiap... {i}", end="", flush=True)
            time.sleep(0.7)

        print(f"\r  ● REKAM — Ucapkan \"{kata}\"   ", flush=True)
        audio = rekam_audio(DURASI_REKAM, SAMPLE_RATE)
        print(f"  ✓ Rekaman selesai.", flush=True)

        simpan_wav(audio, path_file, SAMPLE_RATE)
        print(f"  Disimpan: {path_file}")
        time.sleep(0.8)
        indeks_sesi += 1

    return "selesai"


def tampilkan_status_semua():
    """Tampilkan ringkasan progress semua kelas beserta nomor yang kosong."""
    bersihkan_layar()
    print("=" * 50)
    print("  DOKTERKU — Status Dataset")
    print("=" * 50)
    total_sampel = 0
    for kata in KATA_GEJALA:
        folder = os.path.join(DATASET_FOLDER, kata)
        ada = hitung_sampel_ada(folder, kata)
        total_sampel += ada
        status = "✓ SELESAI" if ada >= JUMLAH_SAMPEL else f"{ada}/{JUMLAH_SAMPEL}"

        # Tampilkan nomor yang kosong jika belum selesai
        if ada < JUMLAH_SAMPEL:
            kosong = dapatkan_nomor_kosong(folder, kata, JUMLAH_SAMPEL)
            # Tampilkan maksimal 5 nomor pertama yang kosong agar tidak terlalu panjang
            preview = ", ".join(f"{n:03d}" for n in kosong[:5])
            if len(kosong) > 5:
                preview += f", ... (+{len(kosong)-5} lagi)"
            info_kosong = f"  ← kosong: {preview}"
        else:
            info_kosong = ""

        bar_mini = ("█" * min(ada, JUMLAH_SAMPEL) + "░" * max(0, JUMLAH_SAMPEL - ada))[:10]
        print(f"  {kata:<15} {bar_mini}  {status}{info_kosong}")

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
        ada = hitung_sampel_ada(folder, kata)
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
    print(f"  Setiap kata akan direkam {JUMLAH_SAMPEL} kali selama {DURASI_REKAM} detik.")
    print(f"  Penomoran otomatis mengisi slot yang belum ada (tidak menimpa file lama).\n")
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
                kosong = dapatkan_nomor_kosong(folder_kelas, kata, JUMLAH_SAMPEL)
                print(f"\n  Mulai merekam: \"{kata}\" ({len(kosong)} slot kosong)")
                time.sleep(1)

                hasil = rekam_satu_kelas(kata, folder_kelas, JUMLAH_SAMPEL)
                if hasil == "keluar":
                    break

        else:
            try:
                idx = int(pilihan) - 1
                if 0 <= idx < len(KATA_GEJALA):
                    kata = KATA_GEJALA[idx]
                    folder_kelas = os.path.join(DATASET_FOLDER, kata)
                    ada = hitung_sampel_ada(folder_kelas, kata)

                    if ada >= JUMLAH_SAMPEL:
                        print(f"\n  Kata \"{kata}\" sudah lengkap ({JUMLAH_SAMPEL} sampel).")
                        input("  Tekan ENTER untuk kembali...")
                        continue

                    rekam_satu_kelas(kata, folder_kelas, JUMLAH_SAMPEL)
                else:
                    print("  Nomor tidak valid.")
                    time.sleep(1)
            except ValueError:
                print("  Input tidak dikenali.")
                time.sleep(1)


if __name__ == "__main__":
    main()