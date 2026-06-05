import os
import pickle
import threading
import numpy as np
import librosa
import sounddevice as sd
from scipy.io import wavfile

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
SAMPLE_RATE   = 16000
DURASI_REKAM  = 2.0
PRE_EMPH_COEF = 0.97
FRAME_LENGTH  = 0.025
FRAME_STEP    = 0.010
N_MFCC        = 40
N_FFT         = 512
N_MELS        = 128

# ─────────────────────────────────────────────
# KNOWLEDGE BASE — 20 KATA GEJALA
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "demam": {
        "saran": "Gejala demam terdeteksi. Istirahat yang cukup dan minum air putih minimal 8 gelas per hari. Konsumsi paracetamol jika suhu di atas 38°C. Segera ke dokter jika demam tidak turun dalam 3 hari atau disertai kejang.",
        "urgensi": "sedang",
        "emoji": "🌡️"
    },
    "batuk": {
        "saran": "Gejala batuk terdeteksi. Hindari minuman dingin dan makanan berminyak. Madu hangat dengan lemon dapat membantu meredakan batuk. Jika batuk berlangsung lebih dari 2 minggu atau disertai darah, segera konsultasikan ke dokter.",
        "urgensi": "rendah",
        "emoji": "😮‍💨"
    },
    "pusing": {
        "saran": "Gejala pusing terdeteksi. Duduk atau berbaring dan hindari gerakan mendadak. Pastikan tubuh terhidrasi dengan baik. Jika pusing disertai mual hebat, penglihatan ganda, atau bicara pelo, segera ke IGD.",
        "urgensi": "sedang",
        "emoji": "💫"
    },
    "mual": {
        "saran": "Gejala mual terdeteksi. Makan dalam porsi kecil tapi sering. Hindari makanan berlemak dan berbau menyengat. Jahe hangat dapat membantu meredakan mual. Jika disertai muntah hebat lebih dari 24 jam, konsultasi ke dokter.",
        "urgensi": "rendah",
        "emoji": "🤢"
    },
    "sesak": {
        "saran": "⚠️ PERHATIAN: Gejala sesak napas terdeteksi. Duduk tegak dan coba bernapas perlahan dan dalam. Jika sesak terjadi tiba-tiba, memburuk cepat, atau disertai nyeri dada, SEGERA ke IGD atau hubungi 119.",
        "urgensi": "tinggi",
        "emoji": "🫁"
    },
    "nyeri": {
        "saran": "Gejala nyeri terdeteksi. Lokasi nyeri penting untuk menentukan penanganan. Kompres hangat dapat membantu nyeri otot. Hindari aktivitas berat. Jika nyeri dada mendadak atau sangat hebat, segera ke dokter.",
        "urgensi": "sedang",
        "emoji": "😣"
    },
    "lemas": {
        "saran": "Gejala lemas terdeteksi. Pastikan asupan nutrisi dan cairan cukup. Istirahat yang memadai sangat diperlukan. Jika lemas berkepanjangan lebih dari seminggu tanpa sebab jelas, periksakan diri ke dokter untuk cek darah.",
        "urgensi": "rendah",
        "emoji": "😴"
    },
    "bersin": {
        "saran": "Gejala bersin terdeteksi. Kemungkinan alergi atau gejala awal flu. Hindari pemicu alergi seperti debu dan bulu hewan. Cuci tangan rutin dan gunakan masker di tempat umum. Antihistamin OTC dapat membantu jika karena alergi.",
        "urgensi": "rendah",
        "emoji": "🤧"
    },
    "diare": {
        "saran": "Gejala diare terdeteksi. Perbanyak cairan untuk mencegah dehidrasi — minum oralit atau air putih yang banyak. Hindari makanan berminyak dan susu. Jika diare lebih dari 3 hari, ada darah, atau disertai demam tinggi, segera ke dokter.",
        "urgensi": "sedang",
        "emoji": "🚽"
    },
    "muntah": {
        "saran": "Gejala muntah terdeteksi. Puasakan makan sebentar lalu mulai dengan makanan ringan seperti biskuit. Minum cairan sedikit demi sedikit. Jika muntah terus-menerus lebih dari 12 jam atau ada darah, segera ke dokter.",
        "urgensi": "sedang",
        "emoji": "🤮"
    },
    "gatal": {
        "saran": "Gejala gatal terdeteksi. Hindari menggaruk area yang gatal. Kompres dingin dapat membantu. Perhatikan apakah ada ruam atau kemerahan. Jika gatal menyebar luas atau disertai bengkak di wajah/tenggorokan, segera ke dokter (kemungkinan alergi serius).",
        "urgensi": "rendah",
        "emoji": "🔴"
    },
    "bengkak": {
        "saran": "Gejala bengkak terdeteksi. Elevasi bagian yang bengkak jika memungkinkan. Kompres es selama 15-20 menit. Jika bengkak terjadi di wajah, lidah, atau tenggorokan disertai sesak napas, SEGERA ke IGD — kemungkinan reaksi alergi berat.",
        "urgensi": "sedang",
        "emoji": "🫧"
    },
    "panas": {
        "saran": "Gejala panas badan terdeteksi. Kenakan pakaian tipis dan berada di ruangan sejuk. Kompres hangat di dahi. Minum banyak cairan. Paracetamol dapat diberikan sesuai dosis anjuran. Pantau suhu secara berkala.",
        "urgensi": "sedang",
        "emoji": "🔥"
    },
    "keringat": {
        "saran": "Gejala keringat berlebihan terdeteksi. Keringat malam bisa menjadi tanda infeksi atau kondisi medis tertentu. Pastikan ruangan berventilasi baik. Jika keringat berlebihan terjadi terus-menerus tanpa aktivitas fisik, periksakan ke dokter.",
        "urgensi": "rendah",
        "emoji": "💧"
    },
    "berdebar": {
        "saran": "Gejala jantung berdebar terdeteksi. Duduk dan cobalah bernapas perlahan dan dalam. Hindari kafein dan nikotin. Jika berdebar disertai nyeri dada, pusing berat, atau pingsan, SEGERA ke IGD — ini bisa menjadi tanda gangguan jantung.",
        "urgensi": "tinggi",
        "emoji": "💓"
    },
    "kebas": {
        "saran": "Gejala kebas atau mati rasa terdeteksi. Kebas sementara sering disebabkan posisi tubuh yang salah. Gerakkan bagian yang kebas perlahan. Jika kebas terjadi mendadak di satu sisi tubuh disertai bicara pelo atau wajah mencong, SEGERA ke IGD — kemungkinan stroke.",
        "urgensi": "tinggi",
        "emoji": "🦵"
    },
    "kram": {
        "saran": "Gejala kram otot terdeteksi. Regangkan otot yang kram secara perlahan dan lembut. Pijat area yang kram. Pastikan asupan cairan dan elektrolit (kalium, magnesium) cukup. Kram kaki malam sering terjadi akibat dehidrasi.",
        "urgensi": "rendah",
        "emoji": "💪"
    },
    "sakit": {
        "saran": "Gejala sakit terdeteksi. Keterangan lebih spesifik tentang lokasi dan jenis sakit akan membantu diagnosis. Istirahat yang cukup dan hindari aktivitas yang memperburuk kondisi. Konsultasikan ke dokter untuk evaluasi lebih lanjut.",
        "urgensi": "sedang",
        "emoji": "🏥"
    },
    "tenggorokan": {
        "saran": "Gejala sakit tenggorokan terdeteksi. Berkumur dengan air garam hangat beberapa kali sehari. Minum air hangat dengan madu dan lemon. Hindari makanan pedas dan minuman dingin. Jika disertai demam tinggi atau sulit menelan, konsultasi ke dokter.",
        "urgensi": "rendah",
        "emoji": "🗣️"
    },
    "hidung": {
        "saran": "Gejala masalah hidung terdeteksi (tersumbat/berair). Hirup uap air panas untuk melonggarkan saluran. Gunakan saline nasal spray jika tersedia. Tidur dengan kepala sedikit lebih tinggi. Jika disertai nyeri wajah dan ingus berwarna kuning/hijau lebih dari seminggu, kemungkinan sinusitis.",
        "urgensi": "rendah",
        "emoji": "👃"
    },
}

URGENSI_WARNA = {
    "rendah": "#2D9E75",
    "sedang": "#E89B2A",
    "tinggi": "#D84F3A",
}

# ─────────────────────────────────────────────
# PIPELINE EKSTRAKSI
# ─────────────────────────────────────────────

def _load_audio_array(audio_array, sr=SAMPLE_RATE, durasi_max=DURASI_MAX if 'DURASI_MAX' in dir() else DURASI_REKAM):
    """Proses numpy array audio (bukan file) — untuk inferensi real-time."""
    audio = audio_array.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Trim silence
    try:
        audio, _ = librosa.effects.trim(audio, top_db=30)
    except Exception:
        pass
    target = int(DURASI_REKAM * sr)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)), mode="constant")
    else:
        audio = audio[:target]
    return audio


def _pre_emphasis(audio):
    return np.append(audio[0], audio[1:] - PRE_EMPH_COEF * audio[:-1])


def _ekstrak_mfcc_delta(audio, sr=SAMPLE_RATE):
    hop = int(FRAME_STEP * sr)
    win = int(FRAME_LENGTH * sr)
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                   n_fft=N_FFT, hop_length=hop,
                                   win_length=win, window="hamming",
                                   n_mels=N_MELS)
    delta  = librosa.feature.delta(mfcc, order=1, width=9)
    delta2 = librosa.feature.delta(mfcc, order=2, width=9)
    return np.vstack([mfcc, delta, delta2])  # (120, T)


def _agregasi(mfcc_combined):
    return np.concatenate([
        mfcc_combined.mean(axis=1),
        mfcc_combined.std(axis=1)
    ]).astype(np.float32)  # (240,)


def ekstrak_fitur_dari_array(audio_array):
    """Pipeline lengkap: numpy array → vektor fitur (240,)"""
    audio = _load_audio_array(audio_array)
    audio = _pre_emphasis(audio)
    mfcc  = _ekstrak_mfcc_delta(audio)
    return _agregasi(mfcc)


# ─────────────────────────────────────────────
# KELAS UTAMA — InferensiASR
# ─────────────────────────────────────────────

class InferensiASR:
    """
    Kelas utama untuk inferensi ASR real-time.

    Contoh pakai:
        asr = InferensiASR()
        kata, conf, saran, urgensi = asr.rekam_dan_prediksi()
        print(f"Kata: {kata} | Confidence: {conf:.1%}")
        print(f"Saran: {saran}")
    """

    def __init__(self,
                 model_terbaik_file="model_terbaik.txt",
                 scaler_file="scaler.pkl",
                 model_svm_file="model_svm.pkl",
                 model_rf_file="model_rf.pkl",
                 model_xgb_file="model_xgb.pkl",
                 model_mlp_file="model_mlp.pt",
                 model_cnn_file="model_cnn1d.pt"):

        self.siap      = False
        self.error_msg = ""
        self._model    = None
        self._scaler   = None
        self._tipe     = None
        self._labels   = None

        self._muat_model(model_terbaik_file, scaler_file,
                         model_svm_file, model_rf_file,
                         model_xgb_file, model_mlp_file, model_cnn_file)

    def _muat_model(self, terbaik_file, scaler_file,
                    svm_file, rf_file, xgb_file, mlp_file, cnn_file):
        try:
            # Scaler
            if not os.path.exists(scaler_file):
                raise FileNotFoundError("scaler.pkl tidak ditemukan.")
            with open(scaler_file, "rb") as f:
                self._scaler = pickle.load(f)

            # Baca tipe model terbaik
            if not os.path.exists(terbaik_file):
                raise FileNotFoundError("model_terbaik.txt tidak ditemukan.")
            with open(terbaik_file) as f:
                self._tipe = f.read().strip()

            # ── Helper: load model sklearn dari .pkl ──
            def _load_sklearn_pkl(path, fallback_labels):
                with open(path, "rb") as f:
                    pkg = pickle.load(f)

                if isinstance(pkg, dict):
                    model  = pkg["model"]
                    labels_raw = pkg.get("label_names", fallback_labels)
                else:
                    # Format lama: pkl langsung berisi model
                    model      = pkg
                    labels_raw = fallback_labels

                # Konversi label ke string bersih
                labels = [str(l) for l in labels_raw]

                if labels and labels[0].isdigit():
                    print(f"  ⚠ Label numerik terdeteksi ({labels[:3]}...)")
                    print(f"    Mapping ke nama kata dari KNOWLEDGE_BASE")
                    kb_keys = list(KNOWLEDGE_BASE.keys())
                    labels  = [kb_keys[int(l)] if int(l) < len(kb_keys)
                               else l for l in labels]
                    print(f"    Label setelah mapping: {labels[:5]}...")

                return model, labels

            default_labels = list(KNOWLEDGE_BASE.keys())

            # ── SVM ──
            if self._tipe == "svm":
                self._model, self._labels = _load_sklearn_pkl(
                    svm_file, default_labels)

            # ── Random Forest ──
            elif self._tipe == "rf":
                if not os.path.exists(rf_file):
                    raise FileNotFoundError(
                        f"{rf_file} tidak ditemukan. "
                        "Pastikan training_model.py sudah dijalankan "
                        "dan model RF tersimpan.")
                self._model, self._labels = _load_sklearn_pkl(
                    rf_file, default_labels)

            # ── XGBoost ──
            elif self._tipe == "xgb":
                if not os.path.exists(xgb_file):
                    raise FileNotFoundError(
                        f"{xgb_file} tidak ditemukan.")
                self._model, self._labels = _load_sklearn_pkl(
                    xgb_file, default_labels)

            # ── MLP (PyTorch) ──
            elif self._tipe == "mlp":
                import torch
                import torch.nn as nn

                class MLPModel(nn.Module):
                    def __init__(self, input_dim, n_kelas, hidden, dropout):
                        super().__init__()
                        layers, prev = [], input_dim
                        for h in hidden:
                            layers += [nn.Linear(prev, h),
                                       nn.BatchNorm1d(h),
                                       nn.ReLU(),
                                       nn.Dropout(dropout)]
                            prev = h
                        layers.append(nn.Linear(prev, n_kelas))
                        self.net = nn.Sequential(*layers)
                    def forward(self, x): return self.net(x)

                ckpt = torch.load(mlp_file, map_location="cpu")
                m    = MLPModel(ckpt["input_dim"], ckpt["n_kelas"],
                                ckpt["hidden"], ckpt["dropout"])
                m.load_state_dict(ckpt["state_dict"])
                m.eval()
                self._model  = m
                self._labels = [str(l) for l in ckpt["label_names"]]
                self._torch  = torch

            # ── CNN1D (PyTorch) ──
            elif self._tipe == "cnn1d":
                import torch
                import torch.nn as nn

                class CNN1DModel(nn.Module):
                    def __init__(self, n_kelas, dropout=0.4):
                        super().__init__()
                        self.conv = nn.Sequential(
                            nn.Conv1d(1,32,7,padding=3), nn.BatchNorm1d(32),
                            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout/2),
                            nn.Conv1d(32,64,5,padding=2), nn.BatchNorm1d(64),
                            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout/2),
                            nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128),
                            nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                        )
                        self.fc = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(128,64), nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(64, n_kelas)
                        )
                    def forward(self, x): return self.fc(self.conv(x.unsqueeze(1)))

                ckpt = torch.load(cnn_file, map_location="cpu")
                m    = CNN1DModel(ckpt["n_kelas"])
                m.load_state_dict(ckpt["state_dict"])
                m.eval()
                self._model  = m
                self._labels = [str(l) for l in ckpt["label_names"]]
                self._torch  = torch

            else:
                raise ValueError(
                    f"Tipe model tidak dikenal: '{self._tipe}'. "
                    "Nilai valid: svm, rf, xgb, mlp, cnn1d")

            self.siap = True
            print(f"  ✓ Model [{self._tipe.upper()}] berhasil dimuat")
            print(f"  ✓ {len(self._labels)} kelas")

        except Exception as e:
            self.error_msg = str(e)
            print(f"  ✗ Gagal muat model: {e}")

    # ── Rekam audio dari mikrofon ──
    def rekam_audio(self, durasi=DURASI_REKAM, sr=SAMPLE_RATE,
                    callback_countdown=None):
        """
        Rekam audio dari mikrofon selama `durasi` detik.
        callback_countdown(detik_tersisa) dipanggil tiap detik jika ada.
        Return: numpy array float32
        """
        frames = []
        event  = threading.Event()

        def _callback(indata, frame_count, time_info, status):
            frames.append(indata.copy())

        with sd.InputStream(samplerate=sr, channels=1,
                            dtype="float32", callback=_callback):
            if callback_countdown:
                import time
                for i in range(int(durasi), 0, -1):
                    callback_countdown(i)
                    time.sleep(1)
            else:
                sd.sleep(int(durasi * 1000))

        audio = np.concatenate(frames, axis=0).flatten()
        return audio

    # ── Prediksi dari array audio ──
    def prediksi(self, audio_array):
        """
        Input : numpy array audio (float32, 16kHz)
        Output: (kata:str, confidence:float, top3:list[(kata,conf)])
        """
        if not self.siap:
            raise RuntimeError(f"Model belum siap: {self.error_msg}")

        fitur   = ekstrak_fitur_dari_array(audio_array)
        fitur_s = self._scaler.transform([fitur])

        # ── Sklearn-based: SVM, Random Forest, XGBoost ──
        if self._tipe in ("svm", "rf", "xgb"):
            proba   = self._model.predict_proba(fitur_s)[0]
            classes = self._model.classes_  

            if hasattr(classes[0], 'item'):
                c0 = classes[0].item()   # numpy scalar → python
            else:
                c0 = classes[0]

            if isinstance(c0, (int, np.integer)) or (isinstance(c0, str) and c0.isdigit()):
                # Model dilatih dengan y integer → map via index
                idx  = int(proba.argmax())
                kata = self._labels[idx]
            else:
                # Model dilatih dengan y string langsung
                idx  = int(proba.argmax())
                kata = str(classes[idx])

            conf = float(proba[idx])
            top3 = sorted(zip(self._labels, proba.tolist()),
                          key=lambda x: -x[1])[:3]

        # ── PyTorch-based: MLP, CNN1D ──
        elif self._tipe in ("mlp", "cnn1d"):
            x = self._torch.tensor(fitur_s, dtype=self._torch.float32)
            with self._torch.no_grad():
                out  = self._model(x)
                prob = self._torch.softmax(out, dim=1)[0].numpy()
            idx  = int(prob.argmax())
            kata = self._labels[idx]
            conf = float(prob[idx])
            top3 = sorted(zip(self._labels, prob.tolist()),
                          key=lambda x: -x[1])[:3]

        else:
            raise RuntimeError(f"Tipe model tidak dikenal: {self._tipe}")

        return str(kata), conf, top3

    # ── Rekam + prediksi sekaligus ──
    def rekam_dan_prediksi(self, callback_countdown=None):
        """
        Pipeline lengkap: rekam mikrofon → prediksi → ambil saran KB.
        Return: (kata, confidence, saran, urgensi, top3, audio_array)
        """
        audio = self.rekam_audio(callback_countdown=callback_countdown)
        kata, conf, top3 = self.prediksi(audio)

        kb     = KNOWLEDGE_BASE.get(kata, {})
        saran  = kb.get("saran",  "Tidak ada saran tersedia.")
        urgensi= kb.get("urgensi","sedang")

        return kata, conf, saran, urgensi, top3, audio

    # ── Ambil saran dari KB ──
    @staticmethod
    def ambil_saran(kata):
        kb = KNOWLEDGE_BASE.get(str(kata), {})
        return (kb.get("saran", "Tidak ada saran."),
                kb.get("urgensi", "sedang"),
                kb.get("emoji", "🏥"))


# ─────────────────────────────────────────────
# UJI MANDIRI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("\n" + "=" * 50)
    print("  DOKTERKU — Uji Inferensi Real-Time")
    print("=" * 50)

    asr = InferensiASR()
    if not asr.siap:
        print(f"\n  ✗ Model tidak siap: {asr.error_msg}")
        print("    Pastikan sudah menjalankan training_model.py")
        exit(1)

    while True:
        print("\n  [ENTER] Mulai rekam  |  [q] Keluar")
        inp = input("  > ").strip().lower()
        if inp == "q":
            break

        print("  Bersiap rekam 2 detik...")
        for i in range(3, 0, -1):
            print(f"\r  Mulai dalam {i}...", end="", flush=True)
            time.sleep(0.7)
        print(f"\r  ● REKAM — ucapkan kata gejala!   ")

        kata, conf, saran, urgensi, top3, _ = asr.rekam_dan_prediksi()

        print(f"\n  Hasil prediksi  : {kata.upper()}")
        print(f"  Confidence      : {conf:.1%}")
        print(f"  Urgensi         : {urgensi}")
        print(f"  Top 3 prediksi  :")
        for k, c in top3:
            bar = "█" * int(c * 20)
            print(f"    {k:<15} {bar:<20} {c:.1%}")
        print(f"\n  Saran:\n  {saran}")
        print("─" * 50)