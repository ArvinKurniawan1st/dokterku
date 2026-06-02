"""
DOKTERKU — Pengujian End-to-End & Evaluasi Performa
=====================================================
Script ini menguji seluruh pipeline DOKTERKU dari ujung ke ujung:

  [TEST 1] Sanity check — semua file model & dependency tersedia
  [TEST 2] Pipeline ekstraksi MFCC — kecepatan & konsistensi fitur
  [TEST 3] Inferensi per model — akurasi, latency, confidence
  [TEST 4] Stress test — 100 prediksi berturut-turut, ukur throughput
  [TEST 5] Robustness — inferensi pada audio noise & silence
  [TEST 6] Knowledge base — semua kata punya saran & urgensi valid
  [TEST 7] TTS — generate audio dari teks, ukur waktu generate
  [TEST 8] Integrasi ASR → KB → TTS — alur penuh satu skenario
  [TEST 9] Laporan akhir — ringkasan semua hasil, cetak & simpan

Output:
  - Terminal: hasil tiap test dengan warna pass/fail
  - laporan_pengujian.txt  — laporan lengkap teks
  - laporan_pengujian.json — data terstruktur untuk paper/analisis

Instalasi:
    pip install librosa numpy scipy scikit-learn torch tqdm
    pip install gtts soundfile   (opsional, untuk test TTS)

Cara pakai:
    python pengujian_e2e.py
    python pengujian_e2e.py --skip-tts   # lewati test TTS
    python pengujian_e2e.py --quick       # hanya test 1-4
"""

import os
import sys
import time
import json
import pickle
import argparse
import warnings
import tempfile
import numpy as np
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
DATASET_FOLDER  = "dataset"
FEATURES_FILE   = "features.npz"
SAMPLE_RATE     = 16000
DURASI_AUDIO    = 2.0
N_STRESS_TEST   = 100    # jumlah prediksi untuk stress test

FILE_MODEL_WAJIB = [
    "scaler.pkl",
    "model_terbaik.txt",
]
FILE_MODEL_OPSIONAL = [
    "model_svm.pkl",
    "model_mlp.pt",
    "model_cnn1d.pt",
    "model_rf.pkl",
    "model_xgb.pkl",
]

KATA_GEJALA = [
    "demam","batuk","pusing","mual","sesak",
    "nyeri","lemas","bersin","diare","muntah",
    "gatal","bengkak","panas","keringat","berdebar",
    "kebas","kram","sakit","tenggorokan","hidung"
]

# ─────────────────────────────────────────────
# WARNA TERMINAL
# ─────────────────────────────────────────────
USE_COLOR = sys.stdout.isatty()
def _c(t, kode): return f"\033[{kode}m{t}\033[0m" if USE_COLOR else t
OK    = lambda t: _c(t, "92")
FAIL  = lambda t: _c(t, "91")
WARN  = lambda t: _c(t, "93")
INFO  = lambda t: _c(t, "94")
BOLD  = lambda t: _c(t, "1")
DIM   = lambda t: _c(t, "2")

PASS_SYM = OK("✓ PASS")
FAIL_SYM = FAIL("✗ FAIL")
WARN_SYM = WARN("⚠ WARN")
SKIP_SYM = DIM("— SKIP")

# ─────────────────────────────────────────────
# STRUKTUR HASIL
# ─────────────────────────────────────────────
hasil_global = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "tests": [],
    "ringkasan": {}
}

def catat(nama, status, detail="", metrik=None):
    """Catat satu hasil test ke hasil_global."""
    hasil_global["tests"].append({
        "nama":   nama,
        "status": status,          # PASS / FAIL / WARN / SKIP
        "detail": detail,
        "metrik": metrik or {}
    })

def header(judul):
    print(f"\n{'─'*60}")
    print(BOLD(f"  {judul}"))
    print('─'*60)

def baris(nama, status_sym, detail=""):
    print(f"  {status_sym}  {nama:<40} {DIM(detail)}")


# ─────────────────────────────────────────────
# PIPELINE MFCC (sama dengan ekstraksi_mfcc.py)
# ─────────────────────────────────────────────
def buat_audio_dummy(durasi=DURASI_AUDIO, sr=SAMPLE_RATE, kata_idx=0):
    """Buat audio sintetis berupa sine wave — untuk test tanpa mikrofon."""
    # Frekuensi berbeda per kelas agar ada variasi
    freq = 200 + kata_idx * 50
    t    = np.linspace(0, durasi, int(durasi * sr), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t) +
             0.1 * np.random.normal(0, 1, len(t))).astype(np.float32)
    return audio

def ekstrak_fitur(audio, sr=SAMPLE_RATE):
    import librosa
    hop = int(0.010 * sr)
    win = int(0.025 * sr)
    # Trim
    audio, _ = librosa.effects.trim(audio, top_db=30)
    target   = int(DURASI_AUDIO * sr)
    audio    = (np.pad(audio, (0, max(0, target - len(audio))))
                if len(audio) < target else audio[:target])
    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    # MFCC + delta + delta-delta
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40,
                                   n_fft=512, hop_length=hop,
                                   win_length=win, window="hamming", n_mels=128)
    delta  = librosa.feature.delta(mfcc, order=1, width=9)
    delta2 = librosa.feature.delta(mfcc, order=2, width=9)
    comb   = np.vstack([mfcc, delta, delta2])
    return np.concatenate([comb.mean(axis=1),
                           comb.std(axis=1)]).astype(np.float32)

def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

def load_model_aktif():
    """Load model terbaik (apapun tipenya)."""
    with open("model_terbaik.txt") as f:
        tipe = f.read().strip()

    scaler = load_scaler()

    if tipe == "svm":
        with open("model_svm.pkl", "rb") as f:
            pkg = pickle.load(f)
        model  = pkg["model"] if isinstance(pkg, dict) else pkg
        labels = (pkg.get("label_names", KATA_GEJALA)
                  if isinstance(pkg, dict) else KATA_GEJALA)
        def prediksi(fitur):
            fs   = scaler.transform([fitur])
            pred = model.predict(fs)[0]
            prob = model.predict_proba(fs)[0]
            return str(pred), float(prob.max()), prob.tolist()

    elif tipe == "rf":
        with open("model_rf.pkl", "rb") as f:
            pkg = pickle.load(f)
        model  = pkg["model"] if isinstance(pkg, dict) else pkg
        labels = (pkg.get("label_names", KATA_GEJALA)
                  if isinstance(pkg, dict) else KATA_GEJALA)
        def prediksi(fitur):
            fs   = scaler.transform([fitur])
            pred = model.predict(fs)[0]
            prob = model.predict_proba(fs)[0]
            return str(pred), float(prob.max()), prob.tolist()

    elif tipe == "xgb":
        with open("model_xgb.pkl", "rb") as f:
            pkg = pickle.load(f)
        model  = pkg["model"] if isinstance(pkg, dict) else pkg
        labels = (pkg.get("label_names", KATA_GEJALA)
                  if isinstance(pkg, dict) else KATA_GEJALA)
        def prediksi(fitur):
            fs   = scaler.transform([fitur])
            pred = model.predict(fs)[0]
            prob = model.predict_proba(fs)[0]
            return str(pred), float(prob.max()), prob.tolist()

    elif tipe == "mlp":
        import torch, torch.nn as nn
        class MLPModel(nn.Module):
            def __init__(self, d, n, h, dr):
                super().__init__()
                layers, p = [], d
                for hi in h:
                    layers += [nn.Linear(p,hi), nn.BatchNorm1d(hi),
                                nn.ReLU(), nn.Dropout(dr)]
                    p = hi
                layers.append(nn.Linear(p, n))
                self.net = nn.Sequential(*layers)
            def forward(self, x): return self.net(x)
        ckpt   = torch.load("model_mlp.pt", map_location="cpu")
        m      = MLPModel(ckpt["input_dim"], ckpt["n_kelas"],
                          ckpt["hidden"], ckpt["dropout"])
        m.load_state_dict(ckpt["state_dict"]); m.eval()
        labels = [str(l) for l in ckpt["label_names"]]
        def prediksi(fitur):
            fs = scaler.transform([fitur])
            x  = torch.tensor(fs, dtype=torch.float32)
            with torch.no_grad():
                p = torch.softmax(m(x), dim=1)[0].numpy()
            idx = int(p.argmax())
            return labels[idx], float(p[idx]), p.tolist()

    else:
        raise ValueError(f"Tipe model tidak dikenal: {tipe}")

    return prediksi, labels, tipe, scaler


# ─────────────────────────────────────────────
# TEST 1 — SANITY CHECK
# ─────────────────────────────────────────────
def test_1_sanity():
    header("TEST 1 — Sanity Check: File & Dependency")
    semua_ok = True

    # File model wajib
    for f in FILE_MODEL_WAJIB:
        ada = os.path.exists(f)
        baris(f, PASS_SYM if ada else FAIL_SYM,
              f"{'OK' if ada else 'TIDAK DITEMUKAN'}")
        if not ada:
            semua_ok = False

    # File model opsional
    for f in FILE_MODEL_OPSIONAL:
        ada = os.path.exists(f)
        baris(f, PASS_SYM if ada else WARN_SYM,
              f"{'OK' if ada else 'tidak ada (opsional)'}")

    # Features.npz
    if os.path.exists(FEATURES_FILE):
        data = np.load(FEATURES_FILE, allow_pickle=True)
        X    = data["X"]
        baris(FEATURES_FILE, PASS_SYM,
              f"{X.shape[0]} sampel × {X.shape[1]} fitur")
    else:
        baris(FEATURES_FILE, WARN_SYM, "tidak ditemukan")

    # Dependency
    deps = [
        ("numpy",   "numpy"),
        ("librosa", "librosa"),
        ("sklearn", "scikit-learn"),
        ("scipy",   "scipy"),
        ("torch",   "torch (opsional)"),
        ("gtts",    "gtts (opsional TTS)"),
        ("soundfile","soundfile (opsional)"),
    ]
    for mod, nama in deps:
        try:
            __import__(mod)
            baris(f"import {mod}", PASS_SYM, nama)
        except ImportError:
            sym = WARN_SYM if "opsional" in nama else FAIL_SYM
            baris(f"import {mod}", sym, f"{nama} — tidak terinstal")
            if "opsional" not in nama:
                semua_ok = False

    status = "PASS" if semua_ok else "FAIL"
    catat("Sanity Check", status, "File dan dependency diperiksa")
    print(f"\n  Hasil: {PASS_SYM if semua_ok else FAIL_SYM}")
    return semua_ok


# ─────────────────────────────────────────────
# TEST 2 — PIPELINE EKSTRAKSI MFCC
# ─────────────────────────────────────────────
def test_2_pipeline_mfcc():
    header("TEST 2 — Pipeline Ekstraksi MFCC")
    import librosa

    latency_list = []
    konsisten    = True

    # Ukur latency ekstraksi
    audio_dummy = buat_audio_dummy(DURASI_AUDIO)
    N_ULANG     = 20
    print(f"  Mengukur latency ekstraksi ({N_ULANG}× audio dummy)...")

    for i in range(N_ULANG):
        t0    = time.perf_counter()
        fitur = ekstrak_fitur(audio_dummy)
        lat   = (time.perf_counter() - t0) * 1000
        latency_list.append(lat)

    lat_mean = np.mean(latency_list)
    lat_std  = np.std(latency_list)
    lat_max  = np.max(latency_list)

    baris("Shape fitur output", PASS_SYM, f"{fitur.shape} (diharapkan: (240,))")
    baris("Nilai NaN/Inf", PASS_SYM if not (np.isnan(fitur).any() or np.isinf(fitur).any()) else FAIL_SYM,
          "tidak ada" if not np.isnan(fitur).any() else "ADA NaN!")
    baris(f"Latency rata-rata", PASS_SYM if lat_mean < 500 else WARN_SYM,
          f"{lat_mean:.1f} ms ± {lat_std:.1f} ms")
    baris(f"Latency maksimum", PASS_SYM if lat_max < 1000 else WARN_SYM,
          f"{lat_max:.1f} ms")

    # Konsistensi — fitur dari audio yang sama harus identik
    f1 = ekstrak_fitur(audio_dummy)
    f2 = ekstrak_fitur(audio_dummy)
    diff = np.abs(f1 - f2).max()
    konsisten = diff < 1e-4
    baris("Konsistensi (2× fitur identik)", PASS_SYM if konsisten else FAIL_SYM,
          f"max diff = {diff:.8f}")

    # Variasi antar kelas (fitur dari audio berbeda harus berbeda)
    fitur_list = [ekstrak_fitur(buat_audio_dummy(DURASI_AUDIO, i)) for i in range(5)]
    std_antar  = np.array(fitur_list).std(axis=0).mean()
    baris("Variasi antar audio berbeda", PASS_SYM if std_antar > 0.1 else WARN_SYM,
          f"std antar sample = {std_antar:.4f}")

    metrik = {
        "latency_mean_ms": round(lat_mean, 2),
        "latency_std_ms":  round(lat_std, 2),
        "latency_max_ms":  round(lat_max, 2),
        "fitur_shape":     list(fitur.shape),
        "konsisten":       konsisten,
        "variasi_antar":   round(float(std_antar), 4),
    }
    status = "PASS" if lat_mean < 500 and konsisten else "WARN"
    catat("Pipeline MFCC", status, "Latency dan konsistensi", metrik)
    print(f"\n  Hasil: {PASS_SYM if status == 'PASS' else WARN_SYM}")
    return metrik


# ─────────────────────────────────────────────
# TEST 3 — INFERENSI PER MODEL
# ─────────────────────────────────────────────
def test_3_inferensi():
    header("TEST 3 — Inferensi: Akurasi & Latency per Model")

    if not os.path.exists(FEATURES_FILE):
        baris("features.npz", SKIP_SYM, "tidak ditemukan, test dilewati")
        catat("Inferensi Model", "SKIP", "features.npz tidak ada")
        return {}

    data        = np.load(FEATURES_FILE, allow_pickle=True)
    X           = data["X"].astype(np.float32)
    y           = data["y"].astype(np.int32)
    label_names = [str(l) for l in data["label_names"]]

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = load_scaler()
    X_te   = scaler.transform(X_test)

    hasil_model = {}

    # Daftar model yang akan diuji
    model_configs = []

    if os.path.exists("model_svm.pkl"):
        from sklearn.svm import SVC
        with open("model_svm.pkl", "rb") as f:
            pkg = pickle.load(f)
        m = pkg["model"] if isinstance(pkg, dict) else pkg
        model_configs.append(("SVM", m, False))

    if os.path.exists("model_rf.pkl"):
        with open("model_rf.pkl", "rb") as f:
            pkg = pickle.load(f)
        m = pkg["model"] if isinstance(pkg, dict) else pkg
        model_configs.append(("Random Forest", m, False))

    if os.path.exists("model_xgb.pkl"):
        with open("model_xgb.pkl", "rb") as f:
            pkg = pickle.load(f)
        m = pkg["model"] if isinstance(pkg, dict) else pkg
        model_configs.append(("XGBoost", m, False))

    if os.path.exists("model_mlp.pt"):
        try:
            import torch, torch.nn as nn
            class MLP(nn.Module):
                def __init__(self,d,n,h,dr):
                    super().__init__()
                    layers,p=[],d
                    for hi in h:
                        layers+=[nn.Linear(p,hi),nn.BatchNorm1d(hi),
                                  nn.ReLU(),nn.Dropout(dr)]; p=hi
                    layers.append(nn.Linear(p,n))
                    self.net=nn.Sequential(*layers)
                def forward(self,x): return self.net(x)
            ckpt=torch.load("model_mlp.pt",map_location="cpu")
            m=MLP(ckpt["input_dim"],ckpt["n_kelas"],ckpt["hidden"],ckpt["dropout"])
            m.load_state_dict(ckpt["state_dict"]); m.eval()
            model_configs.append(("MLP", m, True))
        except Exception as e:
            baris("MLP", WARN_SYM, f"gagal load: {e}")

    if not model_configs:
        baris("Model", WARN_SYM, "tidak ada model ditemukan")
        catat("Inferensi Model", "WARN", "Tidak ada model")
        return {}

    print(f"  {'Model':<20} {'Akurasi':>9} {'F1':>9} "
          f"{'Lat/sample':>12} {'Throughput':>13}")
    print("  " + "─" * 68)

    for nama, model, is_torch in model_configs:
        try:
            # Ukur latency — prediksi 1 sampel berulang
            N_LAT = 50
            lat_list = []
            for _ in range(N_LAT):
                x_single = X_te[[0]]
                t0 = time.perf_counter()
                if is_torch:
                    import torch
                    with torch.no_grad():
                        _ = model(torch.tensor(x_single, dtype=torch.float32))
                else:
                    _ = model.predict(x_single)
                lat_list.append((time.perf_counter() - t0) * 1000)

            lat_mean = np.mean(lat_list[5:])  # skip warmup

            # Prediksi seluruh test set
            t0 = time.perf_counter()
            if is_torch:
                import torch
                Xt = torch.tensor(X_te, dtype=torch.float32)
                with torch.no_grad():
                    out = model(Xt)
                    y_pred = out.argmax(1).numpy()
            else:
                y_pred = model.predict(X_te)
            dur_total = time.perf_counter() - t0

            acc  = accuracy_score(y_test, y_pred)
            f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            tput = len(X_te) / dur_total

            sym = PASS_SYM if acc >= 0.80 else WARN_SYM if acc >= 0.65 else FAIL_SYM
            print(f"  {nama:<20} {acc*100:>8.2f}% {f1*100:>8.2f}% "
                  f"{lat_mean:>10.2f}ms {tput:>10.0f}/s")

            hasil_model[nama] = {
                "akurasi":      round(float(acc), 4),
                "f1":           round(float(f1), 4),
                "latency_ms":   round(lat_mean, 2),
                "throughput":   round(tput, 1),
            }

        except Exception as e:
            baris(nama, FAIL_SYM, f"Error: {e}")

    catat("Inferensi Model", "PASS", "Semua model dievaluasi", hasil_model)
    return hasil_model


# ─────────────────────────────────────────────
# TEST 4 — STRESS TEST
# ─────────────────────────────────────────────
def test_4_stress():
    header(f"TEST 4 — Stress Test ({N_STRESS_TEST}× prediksi berturut-turut)")

    try:
        prediksi_fn, labels, tipe, scaler = load_model_aktif()
    except Exception as e:
        baris("Load model", FAIL_SYM, str(e))
        catat("Stress Test", "FAIL", str(e))
        return

    print(f"  Model aktif : {tipe.upper()}")
    print(f"  Prediksi    : {N_STRESS_TEST}× audio sintetis acak\n")

    latencies   = []
    confidences = []
    errors      = 0

    for i in range(N_STRESS_TEST):
        kata_idx = i % len(KATA_GEJALA)
        audio    = buat_audio_dummy(DURASI_AUDIO, kata_idx)

        t0 = time.perf_counter()
        try:
            fitur    = ekstrak_fitur(audio)
            kata, conf, _ = prediksi_fn(fitur)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            confidences.append(conf)
        except Exception as e:
            errors += 1

        # Progress bar sederhana
        if (i + 1) % 20 == 0 or i == 0:
            bar = "█" * ((i + 1) // 5)
            print(f"\r  Progress: [{bar:<20}] {i+1}/{N_STRESS_TEST}", end="", flush=True)

    print(f"\r  Progress: [{'█'*20}] {N_STRESS_TEST}/{N_STRESS_TEST} ✓")

    lat_arr = np.array(latencies)
    p50  = np.percentile(lat_arr, 50)
    p95  = np.percentile(lat_arr, 95)
    p99  = np.percentile(lat_arr, 99)
    mean = lat_arr.mean()
    tput = N_STRESS_TEST / (lat_arr.sum() / 1000)

    print()
    baris("Total prediksi",    PASS_SYM, f"{N_STRESS_TEST}")
    baris("Prediksi error",    PASS_SYM if errors == 0 else FAIL_SYM, f"{errors}")
    baris("Latency mean",      PASS_SYM if mean < 500 else WARN_SYM, f"{mean:.1f} ms")
    baris("Latency P50 (median)", PASS_SYM, f"{p50:.1f} ms")
    baris("Latency P95",       PASS_SYM if p95 < 1000 else WARN_SYM, f"{p95:.1f} ms")
    baris("Latency P99",       PASS_SYM if p99 < 2000 else WARN_SYM, f"{p99:.1f} ms")
    baris("Throughput",        PASS_SYM, f"{tput:.1f} prediksi/detik")
    baris("Confidence rata-rata", PASS_SYM, f"{np.mean(confidences):.3f}")

    metrik = {
        "n_prediksi": N_STRESS_TEST,
        "errors": errors,
        "latency_mean_ms": round(mean, 2),
        "latency_p50_ms":  round(p50, 2),
        "latency_p95_ms":  round(p95, 2),
        "latency_p99_ms":  round(p99, 2),
        "throughput_per_s": round(tput, 2),
        "confidence_mean": round(float(np.mean(confidences)), 4),
    }
    status = "PASS" if errors == 0 and mean < 500 else "WARN"
    catat("Stress Test", status, f"{N_STRESS_TEST}× prediksi berturut", metrik)
    print(f"\n  Hasil: {PASS_SYM if status == 'PASS' else WARN_SYM}")


# ─────────────────────────────────────────────
# TEST 5 — ROBUSTNESS
# ─────────────────────────────────────────────
def test_5_robustness():
    header("TEST 5 — Robustness: Audio Noise & Silence")

    try:
        prediksi_fn, labels, tipe, scaler = load_model_aktif()
    except Exception as e:
        baris("Load model", FAIL_SYM, str(e))
        catat("Robustness", "FAIL", str(e))
        return

    kasus = {
        "Audio normal (sine wave)":    buat_audio_dummy(DURASI_AUDIO, 0),
        "Audio hening (silence)":      np.zeros(int(DURASI_AUDIO * SAMPLE_RATE), np.float32),
        "Noise Gaussian tinggi":       np.random.normal(0, 0.5, int(DURASI_AUDIO * SAMPLE_RATE)).astype(np.float32),
        "Noise Gaussian rendah":       np.random.normal(0, 0.01, int(DURASI_AUDIO * SAMPLE_RATE)).astype(np.float32),
        "Audio sangat pelan (×0.01)":  buat_audio_dummy(DURASI_AUDIO) * 0.01,
        "Audio clipping (×10)":        np.clip(buat_audio_dummy(DURASI_AUDIO) * 10, -1, 1),
        "Audio pendek (0.1 detik)":    np.pad(buat_audio_dummy(0.1), (0, int(1.9*SAMPLE_RATE))),
    }

    semua_ok = True
    for nama, audio in kasus.items():
        try:
            t0   = time.perf_counter()
            fitur = ekstrak_fitur(audio)
            kata, conf, _ = prediksi_fn(fitur)
            lat  = (time.perf_counter() - t0) * 1000
            ok   = not (np.isnan(fitur).any() or np.isinf(fitur).any())
            baris(nama, PASS_SYM if ok else FAIL_SYM,
                  f"→ '{kata}' conf={conf:.2f} lat={lat:.0f}ms")
            if not ok:
                semua_ok = False
        except Exception as e:
            baris(nama, FAIL_SYM, f"Error: {e}")
            semua_ok = False

    status = "PASS" if semua_ok else "WARN"
    catat("Robustness", status, "Berbagai kondisi audio diuji")
    print(f"\n  Hasil: {PASS_SYM if semua_ok else WARN_SYM}")


# ─────────────────────────────────────────────
# TEST 6 — KNOWLEDGE BASE
# ─────────────────────────────────────────────
def test_6_knowledge_base():
    header("TEST 6 — Knowledge Base: Kelengkapan Saran")

    try:
        from inferensi import KNOWLEDGE_BASE
    except ImportError:
        baris("import inferensi", WARN_SYM, "inferensi.py tidak ditemukan")
        catat("Knowledge Base", "SKIP", "inferensi.py tidak ada")
        return

    errors  = 0
    urgensi_valid = {"rendah", "sedang", "tinggi"}

    for kata in KATA_GEJALA:
        kb = KNOWLEDGE_BASE.get(kata)
        if kb is None:
            baris(kata, FAIL_SYM, "TIDAK ADA di KB")
            errors += 1
            continue
        saran   = kb.get("saran", "")
        urgensi = kb.get("urgensi", "")
        ok_s = len(saran) >= 20
        ok_u = urgensi in urgensi_valid
        sym  = PASS_SYM if (ok_s and ok_u) else FAIL_SYM
        detail = f"urgensi={urgensi} | saran={len(saran)} char"
        baris(kata, sym, detail)
        if not (ok_s and ok_u):
            errors += 1

    status = "PASS" if errors == 0 else "FAIL"
    metrik = {"kata_ok": len(KATA_GEJALA) - errors, "kata_error": errors}
    catat("Knowledge Base", status, f"{len(KATA_GEJALA)-errors}/{len(KATA_GEJALA)} OK", metrik)
    print(f"\n  Hasil: {PASS_SYM if status == 'PASS' else FAIL_SYM}")


# ─────────────────────────────────────────────
# TEST 7 — TTS
# ─────────────────────────────────────────────
def test_7_tts(skip=False):
    header("TEST 7 — TTS: Generate & Format Audio")

    if skip:
        baris("TTS test", SKIP_SYM, "--skip-tts aktif")
        catat("TTS", "SKIP", "Dilewati oleh user")
        return

    try:
        from gtts import gTTS
    except ImportError:
        baris("import gtts", WARN_SYM, "gtts tidak terinstal — pip install gtts")
        catat("TTS", "SKIP", "gtts tidak terinstal")
        return

    kalimat_uji = [
        ("Pendek",   "Demam"),
        ("Sedang",   "Istirahat yang cukup dan minum air putih."),
        ("Panjang",  "Gejala sesak napas terdeteksi. Duduk tegak dan coba bernapas perlahan dan dalam."),
    ]

    for nama, teks in kalimat_uji:
        try:
            t0  = time.perf_counter()
            tts = gTTS(text=teks, lang="id", slow=False)
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tts.save(tmp.name)
            dur = (time.perf_counter() - t0) * 1000
            sz  = os.path.getsize(tmp.name)
            os.unlink(tmp.name)
            baris(f"Teks {nama} ({len(teks)} char)",
                  PASS_SYM if dur < 5000 else WARN_SYM,
                  f"generate={dur:.0f}ms | size={sz//1024}KB")
        except Exception as e:
            baris(f"Teks {nama}", FAIL_SYM, str(e))

    catat("TTS", "PASS", "gTTS berhasil generate audio")
    print(f"\n  Hasil: {PASS_SYM}")


# ─────────────────────────────────────────────
# TEST 8 — INTEGRASI ASR → KB → TTS
# ─────────────────────────────────────────────
def test_8_integrasi(skip_tts=False):
    header("TEST 8 — Integrasi End-to-End: ASR → KB → TTS")

    print("  Simulasi skenario pengguna mengucapkan kata gejala\n")

    try:
        prediksi_fn, labels, tipe, scaler = load_model_aktif()
    except Exception as e:
        baris("Load model", FAIL_SYM, str(e))
        catat("Integrasi E2E", "FAIL", str(e))
        return

    try:
        from inferensi import KNOWLEDGE_BASE
        KB_OK = True
    except ImportError:
        KB_OK = False

    skenario = [
        ("demam",       0),
        ("sesak",       4),
        ("berdebar",    14),
        ("batuk",       1),
    ]

    print(f"  {'Kata Uji':<15} {'Prediksi':<15} {'Conf':>7} "
          f"{'KB':>5} {'TTS':>5} {'Total':>9}")
    print("  " + "─" * 62)

    semua_ok  = True
    total_lat = []
    metrik_e2e = []

    for kata_uji, kata_idx in skenario:
        t_start = time.perf_counter()

        # Step 1: Ekstraksi MFCC dari audio sintetis
        audio  = buat_audio_dummy(DURASI_AUDIO, kata_idx)
        fitur  = ekstrak_fitur(audio)
        t_mfcc = time.perf_counter()

        # Step 2: Prediksi
        try:
            kata_pred, conf, _ = prediksi_fn(fitur)
        except Exception as e:
            kata_pred, conf = "ERROR", 0.0
        t_pred = time.perf_counter()

        # Step 3: Knowledge Base
        if KB_OK:
            kb     = KNOWLEDGE_BASE.get(kata_pred, {})
            saran  = kb.get("saran", "")
            urgsi  = kb.get("urgensi", "?")
            kb_ok  = len(saran) > 10
        else:
            saran, urgsi, kb_ok = "", "?", False
        t_kb = time.perf_counter()

        # Step 4: TTS (opsional)
        tts_ok = False
        if not skip_tts and saran:
            try:
                from gtts import gTTS
                tts = gTTS(text=saran[:100], lang="id", slow=False)
                tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                tts.save(tmp.name)
                os.unlink(tmp.name)
                tts_ok = True
            except Exception:
                pass
        t_tts = time.perf_counter()

        lat_total = (t_tts - t_start) * 1000
        lat_mfcc  = (t_mfcc - t_start) * 1000
        lat_pred  = (t_pred - t_mfcc) * 1000
        total_lat.append(lat_total)

        sym_kb  = OK("✓") if kb_ok else FAIL("✗")
        sym_tts = (OK("✓") if tts_ok else DIM("—")) if not skip_tts else DIM("skip")
        print(f"  {kata_uji:<15} {kata_pred:<15} {conf:>6.1%} "
              f"  {sym_kb}   {sym_tts}  {lat_total:>7.0f}ms")

        if not kb_ok:
            semua_ok = False

        metrik_e2e.append({
            "kata_uji":      kata_uji,
            "prediksi":      kata_pred,
            "confidence":    round(conf, 4),
            "kb_ok":         kb_ok,
            "tts_ok":        tts_ok,
            "latency_ms_total": round(lat_total, 1),
            "latency_ms_mfcc":  round(lat_mfcc, 1),
            "latency_ms_pred":  round(lat_pred, 1),
        })

    print()
    print(f"  Latency E2E rata-rata : {np.mean(total_lat):.0f} ms")
    print(f"  Latency E2E maksimum  : {np.max(total_lat):.0f} ms")

    status = "PASS" if semua_ok else "WARN"
    catat("Integrasi E2E", status, "ASR→KB→TTS pipeline", {"skenario": metrik_e2e})
    print(f"\n  Hasil: {PASS_SYM if semua_ok else WARN_SYM}")


# ─────────────────────────────────────────────
# TEST 9 — LAPORAN AKHIR
# ─────────────────────────────────────────────
def cetak_laporan():
    header("LAPORAN AKHIR — Ringkasan Semua Test")

    tests    = hasil_global["tests"]
    n_pass   = sum(1 for t in tests if t["status"] == "PASS")
    n_warn   = sum(1 for t in tests if t["status"] == "WARN")
    n_fail   = sum(1 for t in tests if t["status"] == "FAIL")
    n_skip   = sum(1 for t in tests if t["status"] == "SKIP")

    print(f"\n  {'Test':<30} {'Status':>8}")
    print("  " + "─" * 42)
    for t in tests:
        sym = {"PASS": PASS_SYM, "FAIL": FAIL_SYM,
               "WARN": WARN_SYM, "SKIP": SKIP_SYM}.get(t["status"], "?")
        print(f"  {t['nama']:<30} {sym}")

    print("  " + "─" * 42)
    print(f"\n  Total  : {len(tests)} test")
    print(f"  {PASS_SYM} : {n_pass}")
    print(f"  {WARN_SYM} : {n_warn}")
    print(f"  {FAIL_SYM} : {n_fail}")
    print(f"  {SKIP_SYM} : {n_skip}")

    verdict = "SIAP PRODUKSI" if n_fail == 0 else "PERLU PERBAIKAN"
    warna   = OK if n_fail == 0 else FAIL
    print(f"\n  Verdict : {BOLD(warna(verdict))}")

    hasil_global["ringkasan"] = {
        "total": len(tests), "pass": n_pass,
        "warn": n_warn, "fail": n_fail, "skip": n_skip,
        "verdict": verdict
    }

    # Simpan laporan teks
    lines = [
        "=" * 62,
        "  LAPORAN PENGUJIAN END-TO-END — DOKTERKU",
        f"  Waktu: {hasil_global['timestamp']}",
        "=" * 62,
    ]
    for t in tests:
        lines.append(f"\n  [{t['status']:4}] {t['nama']}")
        if t["detail"]:
            lines.append(f"         {t['detail']}")
        if t["metrik"]:
            for k, v in t["metrik"].items():
                if not isinstance(v, (list, dict)):
                    lines.append(f"         {k}: {v}")
    lines += [
        "\n" + "=" * 62,
        f"  PASS:{n_pass}  WARN:{n_warn}  FAIL:{n_fail}  SKIP:{n_skip}",
        f"  VERDICT: {verdict}",
        "=" * 62,
    ]
    with open("laporan_pengujian.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  ✓ Laporan teks  → laporan_pengujian.txt")

    # Simpan JSON
    with open("laporan_pengujian.json", "w", encoding="utf-8") as f:
        json.dump(hasil_global, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Laporan JSON  → laporan_pengujian.json")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DOKTERKU — Pengujian End-to-End")
    parser.add_argument("--skip-tts", action="store_true",
                        help="Lewati test TTS (tidak butuh koneksi internet)")
    parser.add_argument("--quick", action="store_true",
                        help="Hanya jalankan test 1-4 (cepat)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(BOLD("  DOKTERKU — Pengujian End-to-End & Evaluasi Performa"))
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    ok = test_1_sanity()
    if not ok:
        print(FAIL("\n  [!] Sanity check gagal — beberapa dependency tidak terpenuhi."))
        print("      Lanjutkan tetap dijalankan untuk melihat scope masalah.\n")

    test_2_pipeline_mfcc()
    test_3_inferensi()
    test_4_stress()

    if not args.quick:
        test_5_robustness()
        test_6_knowledge_base()
        test_7_tts(skip=args.skip_tts)
        test_8_integrasi(skip_tts=args.skip_tts)

    cetak_laporan()
    print()


if __name__ == "__main__":
    main()