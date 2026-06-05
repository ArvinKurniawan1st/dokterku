# 🏥 DokterKu — Asisten Kesehatan Berbasis Suara

> Sistem pengenalan suara (ASR) dan sintesis suara (TTS) berbahasa Indonesia untuk identifikasi gejala kesehatan secara real-time.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Daftar Isi

- [Tentang Proyek](#tentang-proyek)
- [Fitur Utama](#fitur-utama)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Struktur Proyek](#struktur-proyek)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Pipeline ML](#pipeline-ml)
- [Kata Gejala yang Didukung](#kata-gejala-yang-didukung)
- [Model yang Digunakan](#model-yang-digunakan)
- [Teknologi](#teknologi)
- [Kontribusi](#kontribusi)

---

## Tentang Proyek

**DokterKu** adalah aplikasi desktop berbasis Python yang memungkinkan pengguna mengucapkan kata-kata gejala kesehatan dalam Bahasa Indonesia, lalu sistem akan mengenali, mengklasifikasikan, dan memberikan informasi serta saran terkait gejala tersebut melalui suara (TTS).

Proyek ini dibangun sebagai sistem end-to-end dari perekaman audio, augmentasi data, ekstraksi fitur MFCC, hingga training dan inferensi model klasifikasi, dengan antarmuka grafis (GUI) yang intuitif.

---

## Fitur Utama

- 🎙 **ASR (Automatic Speech Recognition)** — Pengenalan kata gejala dari input suara secara real-time
- 🔊 **TTS (Text-to-Speech)** — Pembacaan saran kesehatan menggunakan `edge-tts` dengan suara Neural Indonesia (laki-laki / perempuan)
- 📊 **Visualisasi MFCC** — Tampilan spektrogram MFCC dari audio yang direkam
- 🧠 **Multi-model klasifikasi** — SVM, Random Forest, XGBoost, MLP, dan CNN 1D; sistem otomatis memilih model terbaik
- 🔁 **Augmentasi Data** — 7 teknik augmentasi audio untuk memperbanyak dataset hingga 8× lipat
- 💊 **Knowledge Base** — Informasi gejala, tingkat urgensi, dan saran pertolongan pertama
- 🎨 **GUI Dark Theme** — Antarmuka modern berbasis Tkinter dengan tema gelap

---

## Arsitektur Sistem

```
Input Suara (Mikrofon)
        │
        ▼
  [Perekaman Audio]
  sounddevice / soundfile
        │
        ▼
  [Pra-pemrosesan]
  Load → Trim Silence → Pre-emphasis → Pad/Trim (2 detik)
        │
        ▼
  [Ekstraksi Fitur MFCC]
  MFCC (40) + Δ + ΔΔ → Agregasi Statistik → Vektor 240 dimensi
        │
        ▼
  [Klasifikasi]
  SVM / Random Forest / XGBoost / MLP / CNN 1D
        │
        ▼
  [Inferensi + Knowledge Base]
  Prediksi Gejala + Tingkat Urgensi + Saran
        │
        ▼
  [TTS Output]
  edge-tts (Neural Indonesian Voice)
```

---

## Struktur Proyek

```
dokterku/
├── dataset/                    # Folder dataset audio per kelas
│   ├── demam/
│   ├── batuk/
│   └── ... (20 kelas gejala)
│
├── augmentasi.py               # Augmentasi data audio (7 teknik)
├── mfcc.py                     # Ekstraksi fitur MFCC
├── train_model.py              # Training & evaluasi 5 model ML
├── inferensi.py                # Modul inferensi + knowledge base
├── gui_app.py                  # Aplikasi GUI utama (Tkinter)
│
├── features.npz                # Fitur hasil ekstraksi MFCC (generated)
├── scaler.pkl                  # StandardScaler (generated)
├── model_svm.pkl               # Model SVM (generated)
├── model_rf.pkl                # Model Random Forest (generated)
├── model_xgb.pkl               # Model XGBoost (generated)
├── model_mlp.pt                # Model MLP PyTorch (generated)
├── model_cnn1d.pt              # Model CNN 1D PyTorch (generated)
├── model_terbaik.txt           # Nama model dengan akurasi terbaik (generated)
├── hasil_training.txt          # Laporan perbandingan model (generated)
└── training_history.png        # Grafik training loss & accuracy (generated)
```

---

## Instalasi

### Prasyarat

- Python 3.9 atau lebih baru
- Mikrofon aktif
- Koneksi internet (untuk fitur TTS `edge-tts`)

### Clone Repository

```bash
git clone https://github.com/ArvinKurniawan1st/dokterku.git
cd dokterku
```

### Install Dependensi

```bash
pip install -r requirements.txt
```

Atau install manual:

```bash
pip install numpy librosa soundfile sounddevice torch scikit-learn xgboost tqdm matplotlib pygame edge-tts
```

---

## Cara Penggunaan

### 1. Siapkan Dataset

Rekam atau kumpulkan file audio `.wav` untuk setiap kata gejala ke dalam folder:

```
dataset/<nama_gejala>/<file>.wav
```

Contoh: `dataset/demam/demam_001.wav`

### 2. Augmentasi Data (Opsional, Direkomendasikan)

```bash
python augmentasi.py
```

Pilih opsi `1` untuk augmentasi semua kelas. Setiap file asli akan menghasilkan 7 file augmentasi (noise, pitch shift ±2 semitone, time stretch ×0.9/×1.1, reverb, gain).

### 3. Ekstraksi Fitur MFCC

```bash
python mfcc.py
```

Pilih opsi `2` untuk memproses seluruh dataset. Hasil disimpan ke `features.npz`.

### 4. Training Model

```bash
python train_model.py
```

Script akan melatih 5 model secara otomatis (SVM, Random Forest, XGBoost, MLP, CNN 1D), membandingkan performanya, dan menyimpan model terbaik.

### 5. Jalankan Aplikasi GUI

```bash
python gui_app.py
```

---

## Pipeline ML

### Augmentasi Data (`augmentasi.py`)

| Teknik            | Parameter                  | Tujuan                           |
| ----------------- | -------------------------- | -------------------------------- |
| Gaussian Noise    | factor=0.004               | Simulasi noise mikrofon          |
| Pitch Shift Up    | +2 semitone                | Variasi suara tinggi (perempuan) |
| Pitch Shift Down  | -2 semitone                | Variasi suara rendah (laki-laki) |
| Time Stretch Slow | rate=0.9×                  | Variasi kecepatan bicara lambat  |
| Time Stretch Fast | rate=1.1×                  | Variasi kecepatan bicara cepat   |
| Room Reverb       | delays=[0.02, 0.05, 0.08]s | Simulasi ruangan                 |
| Random Gain       | 0.7×–1.3×                  | Variasi volume mikrofon          |

### Ekstraksi Fitur (`mfcc.py`)

```
Audio (16kHz, 2 detik)
  → Trim silence (top_db=30)
  → Pre-emphasis (coef=0.97)
  → MFCC 40 koefisien (FFT=512, Mel=128, frame=25ms, step=10ms)
  → Delta + Delta-Delta
  → Agregasi mean + std  →  Vektor 240 dimensi
```

### Training Model (`train_model.py`)

- **Split data**: 70% train / 10% validasi / 20% test (stratified)
- **Normalisasi**: StandardScaler (fit pada train, transform pada val & test)
- **Evaluasi**: Accuracy, F1-Score weighted, Classification Report, Confusion Matrix

---

## Kata Gejala yang Didukung

|       |         |        |             |          |
| ----- | ------- | ------ | ----------- | -------- |
| demam | batuk   | pusing | mual        | sesak    |
| nyeri | lemas   | bersin | diare       | muntah   |
| gatal | bengkak | panas  | keringat    | berdebar |
| kebas | kram    | sakit  | tenggorokan | hidung   |

**Total: 20 kelas gejala**

---

## Model yang Digunakan

| Model             | Library      | Konfigurasi                                  |
| ----------------- | ------------ | -------------------------------------------- |
| **SVM**           | scikit-learn | RBF kernel, C=10, gamma=scale                |
| **Random Forest** | scikit-learn | 500 estimators, bootstrap=True               |
| **XGBoost**       | xgboost      | —                                            |
| **MLP**           | PyTorch      | Layer [256, 128, 64], Dropout=0.3, 100 epoch |
| **CNN 1D**        | PyTorch      | Filter [32, 64, 128], 100 epoch              |

Sistem secara otomatis membandingkan akurasi semua model dan menggunakan yang terbaik pada saat inferensi.

---

## Teknologi

| Kategori         | Library                               |
| ---------------- | ------------------------------------- |
| Audio Processing | `librosa`, `soundfile`, `sounddevice` |
| Machine Learning | `scikit-learn`, `xgboost`             |
| Deep Learning    | `PyTorch`                             |
| TTS              | `edge-tts` (Microsoft Neural TTS)     |
| GUI              | `tkinter`, `matplotlib`               |
| Audio Playback   | `pygame`                              |
| Utilities        | `numpy`, `tqdm`                       |

---

## Kontribusi

Pull request dan issue sangat terbuka. Untuk perubahan besar, silakan buka issue terlebih dahulu untuk mendiskusikan apa yang ingin diubah.

1. Fork repository ini
2. Buat branch fitur baru (`git checkout -b fitur/nama-fitur`)
3. Commit perubahan (`git commit -m 'Tambah fitur X'`)
4. Push ke branch (`git push origin fitur/nama-fitur`)
5. Buka Pull Request

---

<p align="center">
  Dibuat dengan ❤️ untuk membantu masyarakat Indonesia mengenali gejala kesehatan lebih awal.
</p>
