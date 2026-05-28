"""
DOKTERKU — Audit Seluruh File Audio Dataset
=============================================
Mengecek setiap file .wav di folder dataset/ dan mendeteksi:
  1. Audio terlalu pelan (RMS < threshold)
  2. Terlalu banyak silence (>70% dari durasi)
  3. Durasi suara aktif terlalu pendek (<0.2 detik)
  4. File rusak / tidak bisa dibaca
  5. Jumlah sampel per kelas kurang dari target

Output:
  - Ringkasan status per kelas di terminal
  - laporan_audit.txt  → detail semua file bermasalah
  - laporan_audit.csv  → tabel lengkap semua file (bisa dibuka di Excel)

Instalasi:
    pip install librosa numpy scipy tqdm

Cara pakai:
    python cek_dataset.py
"""

import os
import csv
import librosa
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────
# KONFIGURASI — sesuaikan jika perlu
# ─────────────────────────────────────────────
DATASET_FOLDER  = "dataset"
JUMLAH_TARGET   = 25          # sampel yang diharapkan per kelas
SAMPLE_RATE     = 16000       # Hz

# Threshold pengecekan
MIN_RMS          = 0.002      # di bawah ini = terlalu pelan
MIN_DURASI_AKTIF = 0.2        # detik suara nyata setelah trim
MAX_SILENCE_PCT  = 0.70       # di atas ini = terlalu banyak diam
TOP_DB_TRIM      = 25         # agresivitas potong silence

KATA_GEJALA = [
    "demam", "batuk", "pusing", "mual", "sesak",
    "nyeri", "lemas", "bersin", "diare", "muntah",
    "gatal", "bengkak", "panas", "keringat", "berdebar",
    "kebas", "kram", "sakit", "tenggorokan", "hidung"
]

# Kode warna terminal (dinonaktifkan otomatis jika tidak didukung)
try:
    import sys
    USE_COLOR = sys.stdout.isatty()
except Exception:
    USE_COLOR = False

def warna(teks, kode):
    if not USE_COLOR:
        return teks
    return f"\033[{kode}m{teks}\033[0m"

MERAH   = lambda t: warna(t, "91")
HIJAU   = lambda t: warna(t, "92")
KUNING  = lambda t: warna(t, "93")
BIRU    = lambda t: warna(t, "94")
TEBAL   = lambda t: warna(t, "1")


# ─────────────────────────────────────────────
# AUDIT SATU FILE
# ─────────────────────────────────────────────

def audit_file(path):
    """
    Analisis satu file .wav.
    Return dict berisi semua metrik dan list masalah.
    """
    hasil = {
        "path":         path,
        "status":       "OK",          # OK | MASALAH | RUSAK
        "masalah":      [],
        "durasi_asli":  0.0,
        "durasi_aktif": 0.0,
        "silence_pct":  0.0,
        "rms":          0.0,
        "max_amp":      0.0,
        "sr":           0,
    }

    try:
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        durasi_asli = len(audio) / sr

        # Trim silence
        audio_trim, _ = librosa.effects.trim(audio, top_db=TOP_DB_TRIM)
        durasi_aktif  = len(audio_trim) / sr
        silence_pct   = 1.0 - (len(audio_trim) / max(len(audio), 1))

        # Metrik amplitudo
        rms     = float(np.sqrt(np.mean(audio ** 2)))
        max_amp = float(np.abs(audio).max())

        hasil.update({
            "durasi_asli":  round(durasi_asli, 3),
            "durasi_aktif": round(durasi_aktif, 3),
            "silence_pct":  round(silence_pct * 100, 1),
            "rms":          round(rms, 6),
            "max_amp":      round(max_amp, 4),
            "sr":           sr,
        })

        # ── Pengecekan masalah ──
        if rms < MIN_RMS:
            hasil["masalah"].append(
                f"Volume terlalu pelan (RMS={rms:.6f} < {MIN_RMS})"
            )
        if silence_pct > MAX_SILENCE_PCT:
            hasil["masalah"].append(
                f"Terlalu banyak silence ({silence_pct*100:.1f}% > {MAX_SILENCE_PCT*100:.0f}%)"
            )
        if durasi_aktif < MIN_DURASI_AKTIF:
            hasil["masalah"].append(
                f"Durasi suara aktif terlalu pendek ({durasi_aktif:.3f}s < {MIN_DURASI_AKTIF}s)"
            )
        if max_amp < 0.01:
            hasil["masalah"].append(
                f"Amplitudo maksimum sangat kecil ({max_amp:.4f}) — kemungkinan file kosong"
            )

        if hasil["masalah"]:
            hasil["status"] = "MASALAH"

    except Exception as e:
        hasil["status"]  = "RUSAK"
        hasil["masalah"] = [f"Gagal dibaca: {e}"]

    return hasil


# ─────────────────────────────────────────────
# AUDIT SELURUH DATASET
# ─────────────────────────────────────────────

def audit_dataset():
    semua_hasil  = []   # list of dict, satu per file
    ringkasan    = {}   # per kelas: {total, ok, masalah, rusak, kurang}

    print("\n" + "=" * 62)
    print(TEBAL("  DOKTERKU — Audit Dataset Audio"))
    print("=" * 62)
    print(f"  Folder  : {DATASET_FOLDER}/")
    print(f"  Target  : {JUMLAH_TARGET} sampel per kelas")
    print(f"  Kelas   : {len(KATA_GEJALA)}")
    print("=" * 62 + "\n")

    for kata in KATA_GEJALA:
        folder_kelas = os.path.join(DATASET_FOLDER, kata)
        ring = {"kata": kata, "total": 0, "ok": 0,
                "masalah": 0, "rusak": 0, "kurang": False}

        if not os.path.exists(folder_kelas):
            ring["kurang"] = True
            ringkasan[kata] = ring
            print(f"  {KUNING(f'[!] {kata:<15}')} folder tidak ditemukan")
            continue

        wav_files = sorted([f for f in os.listdir(folder_kelas)
                            if f.lower().endswith(".wav")])
        ring["total"] = len(wav_files)

        if len(wav_files) < JUMLAH_TARGET:
            ring["kurang"] = True

        # Proses tiap file dengan progress bar
        for fname in tqdm(wav_files,
                          desc=f"  {kata:<15}",
                          ncols=60, leave=False):
            path   = os.path.join(folder_kelas, fname)
            result = audit_file(path)
            result["kelas"] = kata
            result["file"]  = fname
            semua_hasil.append(result)

            if result["status"] == "OK":
                ring["ok"] += 1
            elif result["status"] == "MASALAH":
                ring["masalah"] += 1
            else:
                ring["rusak"] += 1

        ringkasan[kata] = ring

        # Cetak status per kelas
        n_bermasalah = ring["masalah"] + ring["rusak"]
        kurang_str   = (KUNING(f" ⚠ kurang {JUMLAH_TARGET - ring['total']} sampel")
                        if ring["kurang"] else "")

        if n_bermasalah == 0 and not ring["kurang"]:
            status_str = HIJAU(f"✓ semua {ring['total']} OK")
        elif n_bermasalah > 0:
            status_str = MERAH(f"✗ {n_bermasalah}/{ring['total']} bermasalah")
        else:
            status_str = HIJAU(f"✓ {ring['ok']}/{ring['total']} OK")

        print(f"  {kata:<15} {status_str}{kurang_str}")

    return semua_hasil, ringkasan


# ─────────────────────────────────────────────
# CETAK RINGKASAN AKHIR
# ─────────────────────────────────────────────

def cetak_ringkasan(semua_hasil, ringkasan):
    total_file     = len(semua_hasil)
    total_ok       = sum(1 for r in semua_hasil if r["status"] == "OK")
    total_masalah  = sum(1 for r in semua_hasil if r["status"] == "MASALAH")
    total_rusak    = sum(1 for r in semua_hasil if r["status"] == "RUSAK")
    kelas_kurang   = [k for k, v in ringkasan.items() if v["kurang"]]

    print("\n" + "=" * 62)
    print(TEBAL("  RINGKASAN AUDIT"))
    print("=" * 62)
    print(f"  Total file diperiksa : {total_file}")
    print(f"  {HIJAU('✓ File OK')}              : {total_ok}")
    print(f"  {KUNING('⚠ File bermasalah')}     : {total_masalah}")
    print(f"  {MERAH('✗ File rusak')}           : {total_rusak}")

    if kelas_kurang:
        print(f"\n  {KUNING('Kelas kurang sampel')}  : {', '.join(kelas_kurang)}")

    # Daftar file bermasalah
    bermasalah = [r for r in semua_hasil if r["status"] != "OK"]
    if bermasalah:
        print(f"\n" + "─" * 62)
        print(TEBAL("  DAFTAR FILE BERMASALAH"))
        print("─" * 62)
        for r in bermasalah:
            ikon = MERAH("✗") if r["status"] == "RUSAK" else KUNING("⚠")
            print(f"\n  {ikon} {r['kelas']}/{r['file']}")
            for m in r["masalah"]:
                print(f"      → {m}")
            if r["status"] != "RUSAK":
                print(f"      (durasi_aktif={r['durasi_aktif']}s, "
                      f"silence={r['silence_pct']}%, "
                      f"RMS={r['rms']:.5f})")
    else:
        print(f"\n  {HIJAU('Tidak ada file bermasalah!')} Dataset siap digunakan.")

    # Saran tindakan
    print("\n" + "─" * 62)
    print(TEBAL("  SARAN TINDAKAN"))
    print("─" * 62)
    if total_masalah + total_rusak == 0 and not kelas_kurang:
        print(f"  {HIJAU('✓')} Dataset lengkap dan bersih.")
        print("    Lanjutkan ke: python ekstraksi_mfcc.py")
    else:
        if total_masalah > 0:
            print("  ⚠ Ada file bermasalah:")
            print("    Opsi 1 (cepat) : Naikkan top_db di ekstraksi_mfcc.py")
            print("                     dari top_db=20 menjadi top_db=30")
            print("    Opsi 2 (terbaik): Rekam ulang file yang bermasalah")
            print("                     dengan python rekam_dataset.py")
        if total_rusak > 0:
            print("  ✗ Ada file rusak — hapus dan rekam ulang.")
        if kelas_kurang:
            print("  ⚠ Beberapa kelas kurang sampel — tambah rekaman.")
    print("=" * 62)

    return bermasalah


# ─────────────────────────────────────────────
# SIMPAN LAPORAN
# ─────────────────────────────────────────────

def simpan_laporan_txt(semua_hasil, ringkasan, bermasalah):
    lines = [
        "=" * 62,
        "  LAPORAN AUDIT DATASET DOKTERKU",
        "=" * 62,
        f"  Total file  : {len(semua_hasil)}",
        f"  OK          : {sum(1 for r in semua_hasil if r['status']=='OK')}",
        f"  Bermasalah  : {sum(1 for r in semua_hasil if r['status']=='MASALAH')}",
        f"  Rusak       : {sum(1 for r in semua_hasil if r['status']=='RUSAK')}",
        "",
        "  Ringkasan per kelas:",
        "  " + "-" * 50,
    ]
    for kata, r in ringkasan.items():
        n_ok  = r.get("ok", 0)
        n_err = r.get("masalah", 0) + r.get("rusak", 0)
        total = r.get("total", 0)
        flag  = " ← KURANG SAMPEL" if r.get("kurang") else ""
        lines.append(f"  {kata:<15} OK:{n_ok:2d}  Masalah:{n_err:2d}  Total:{total:2d}{flag}")

    if bermasalah:
        lines += ["", "=" * 62, "  DETAIL FILE BERMASALAH", "=" * 62]
        for r in bermasalah:
            lines.append(f"\n  [{r['status']}] {r['kelas']}/{r['file']}")
            for m in r["masalah"]:
                lines.append(f"    → {m}")
            if r["status"] != "RUSAK":
                lines.append(
                    f"    durasi_aktif={r['durasi_aktif']}s  "
                    f"silence={r['silence_pct']}%  "
                    f"RMS={r['rms']:.6f}  max_amp={r['max_amp']}"
                )

    with open("laporan_audit.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  ✓ Laporan detail  → laporan_audit.txt")


def simpan_laporan_csv(semua_hasil):
    fieldnames = [
        "kelas", "file", "status", "masalah",
        "durasi_asli", "durasi_aktif", "silence_pct",
        "rms", "max_amp", "sr"
    ]
    with open("laporan_audit.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in semua_hasil:
            writer.writerow({
                "kelas":        r.get("kelas", ""),
                "file":         r.get("file",  ""),
                "status":       r["status"],
                "masalah":      " | ".join(r["masalah"]),
                "durasi_asli":  r["durasi_asli"],
                "durasi_aktif": r["durasi_aktif"],
                "silence_pct":  r["silence_pct"],
                "rms":          r["rms"],
                "max_amp":      r["max_amp"],
                "sr":           r["sr"],
            })
    print(f"  ✓ Tabel lengkap   → laporan_audit.csv  (bisa dibuka di Excel)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if not os.path.exists(DATASET_FOLDER):
        print(f"\n  [!] Folder '{DATASET_FOLDER}/' tidak ditemukan.")
        print("      Jalankan rekam_dataset.py terlebih dahulu.\n")
        return

    semua_hasil, ringkasan = audit_dataset()

    if not semua_hasil:
        print("\n  Tidak ada file .wav yang ditemukan di dataset/")
        return

    bermasalah = cetak_ringkasan(semua_hasil, ringkasan)
    simpan_laporan_txt(semua_hasil, ringkasan, bermasalah)
    simpan_laporan_csv(semua_hasil)

    print()


if __name__ == "__main__":
    main()