import tkinter as tk
from tkinter import ttk
import threading
import pickle
import numpy as np
import sounddevice as sd
import librosa

# ==========================
# KONFIGURASI
# ==========================

SAMPLE_RATE = 16000
DURATION = 2.0

N_MFCC = 40
N_FFT = 512
N_MELS = 128

FRAME_LENGTH = 0.025
FRAME_STEP = 0.010

# ==========================
# LOAD MODEL
# ==========================

with open("model_rf.pkl", "rb") as f:
    pkg = pickle.load(f)

rf = pkg["model"]
label_names = pkg["label_names"]

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ==========================
# FEATURE EXTRACTION
# ==========================

def pre_emphasis(audio, coef=0.97):
    return np.append(audio[0], audio[1:] - coef * audio[:-1])


def extract_feature(audio):

    audio, _ = librosa.effects.trim(audio, top_db=30)

    target_len = int(SAMPLE_RATE * DURATION)

    if len(audio) < target_len:
        audio = np.pad(
            audio,
            (0, target_len - len(audio))
        )
    else:
        audio = audio[:target_len]

    audio = pre_emphasis(audio)

    hop_length = int(FRAME_STEP * SAMPLE_RATE)
    win_length = int(FRAME_LENGTH * SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=N_MELS
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feat = np.vstack([
        mfcc,
        delta,
        delta2
    ])

    mean_vec = feat.mean(axis=1)
    std_vec = feat.std(axis=1)

    return np.concatenate([
        mean_vec,
        std_vec
    ])


# ==========================
# RECORD + PREDICT
# ==========================

history = []


def predict_audio():

    try:

        status_label.config(
            text="🎤 Recording..."
        )

        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        audio = audio.flatten()

        feature = extract_feature(audio)

        feature_scaled = scaler.transform(
            [feature]
        )

        # ==========================
        # PREDIKSI RF
        # ==========================

        pred_idx = rf.predict(
            feature_scaled
        )[0]

        probs = rf.predict_proba(
            feature_scaled
        )[0]

        confidence = np.max(probs)

        gejala = label_names[pred_idx]

        # ==========================
        # TOP 3 PREDIKSI
        # ==========================

        top_idx = np.argsort(
            probs
        )[::-1][:3]

        txt = "Top 3 Prediksi:\n\n"

        for i in top_idx:

            txt += (
                f"{label_names[i]:15}"
                f"{probs[i]*100:.2f}%\n"
            )

        top3_label.config(
            text=txt
        )

        # ==========================
        # UPDATE GUI
        # ==========================

        result_label.config(
            text=f"Gejala: {gejala}"
        )

        confidence_label.config(
            text=f"Confidence: {confidence*100:.2f}%"
        )

        history.append(
            f"{gejala} ({confidence*100:.2f}%)"
        )

        history_list.delete(
            0,
            tk.END
        )

        for item in reversed(history[-20:]):

            history_list.insert(
                tk.END,
                item
            )

        status_label.config(
            text="✓ Selesai"
        )

    except Exception as e:

        status_label.config(
            text="❌ Error"
        )

        result_label.config(
            text=f"Error:\n{str(e)}"
        )


def start_recording():
    threading.Thread(
        target=predict_audio,
        daemon=True
    ).start()


# ==========================
# GUI
# ==========================

root = tk.Tk()

root.title(
    "DOKTERKU RF REALTIME"
)

root.geometry("700x650")

title = tk.Label(
    root,
    text="DOKTERKU - ASR GEJALA KESEHATAN",
    font=("Arial", 20, "bold")
)
title.pack(pady=20)

record_btn = tk.Button(
    root,
    text="🎤 START RECORDING",
    font=("Arial", 18),
    width=20,
    height=2,
    command=start_recording
)
record_btn.pack()

status_label = tk.Label(
    root,
    text="Siap",
    font=("Arial", 12)
)
status_label.pack(pady=10)

result_label = tk.Label(
    root,
    text="-",
    font=("Arial", 28, "bold")
)
result_label.pack(pady=20)

confidence_label = tk.Label(
    root,
    text="Confidence: -",
    font=("Arial", 18)
)
confidence_label.pack()

# TOP 3
top3_label = tk.Label(
    root,
    text="Top 3 Prediksi",
    font=("Consolas", 14),
    justify="left",
    anchor="w"
)
top3_label.pack(pady=10)

history_title = tk.Label(
    root,
    text="Riwayat Prediksi",
    font=("Arial", 14, "bold")
)
history_title.pack()

history_list = tk.Listbox(
    root,
    width=40,
    height=10
)
history_list.pack(pady=10)

root.mainloop()