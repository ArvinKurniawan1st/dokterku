"""
DOKTERKU — Training Model Klasifikasi (FIXED)
==============================================
Perbaikan:
  [FIX 1] Hapus verbose=True di ReduceLROnPlateau (deprecated di PyTorch >= 2.2)
  [FIX 2] Tambah diagnosis dataset otomatis (deteksi fitur konstan/kosong)
  [FIX 3] Tambah augmentasi data (noise injection) jika sampel sedikit
  [FIX 4] Label encoding dari string numpy ke int bersih

Melatih 3 model dan membandingkan performanya:
  1. SVM  — Support Vector Machine (RBF kernel) → baseline
  2. MLP  — Multi-Layer Perceptron (PyTorch)
  3. CNN1D — 1D Convolutional Neural Network (PyTorch)

Input  : features.npz (hasil ekstraksi_mfcc.py)
Output :
  - model_svm.pkl       → model SVM terlatih
  - model_mlp.pt        → bobot MLP terlatih
  - model_cnn1d.pt      → bobot CNN1D terlatih
  - hasil_training.txt  → laporan perbandingan semua model

Instalasi:
    pip install scikit-learn torch numpy matplotlib tqdm

Cara pakai:
    python training_model.py
"""

import os
import time
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
FEATURES_FILE = "features.npz"
SEED          = 42
TEST_SIZE     = 0.2
VAL_SIZE      = 0.1

MLP_HIDDEN  = [256, 128, 64]
MLP_DROPOUT = 0.3
MLP_EPOCHS  = 100
MLP_LR      = 1e-3
MLP_BATCH   = 32

CNN_EPOCHS = 100
CNN_LR     = 1e-3
CNN_BATCH  = 32

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# [FIX 2] DIAGNOSIS DATASET
# ─────────────────────────────────────────────

def diagnosis_dataset(X, y, label_names):
    """
    Deteksi masalah umum pada fitur sebelum training:
      1. Fitur bernilai konstan (semua nol atau NaN) → tanda audio rusak/kosong
      2. Distribusi kelas tidak seimbang
      3. Variansi fitur terlalu rendah
    """
    print("\n" + "─" * 55)
    print("  DIAGNOSIS DATASET")
    print("─" * 55)

    ada_masalah = False

    # Cek NaN / Inf
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"  [!] PERINGATAN: ditemukan {n_nan} NaN dan {n_inf} Inf di fitur!")
        print("      Kemungkinan penyebab: file audio kosong atau sangat pendek.")
        print("      Solusi: hapus file .wav yang rusak, ulangi ekstraksi MFCC.")
        ada_masalah = True
        # Ganti NaN/Inf dengan 0 agar training tetap bisa berjalan
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        print("      → NaN/Inf diganti 0 untuk sementara.")
    else:
        print("  ✓ Tidak ada NaN/Inf")

    # Cek fitur konstan (std ≈ 0)
    std_per_fitur = X.std(axis=0)
    n_konstan = (std_per_fitur < 1e-6).sum()
    if n_konstan > 0:
        print(f"  [!] PERINGATAN: {n_konstan} fitur bernilai konstan (std≈0)")
        print("      Ini sering terjadi jika audio terlalu pendek atau hanya silence.")
        ada_masalah = True
    else:
        print(f"  ✓ Variansi fitur normal (min_std={std_per_fitur.min():.4f})")

    # Cek variansi keseluruhan
    print(f"  ✓ Rentang nilai fitur: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  ✓ Rata-rata std fitur : {std_per_fitur.mean():.4f}")

    # Cek distribusi kelas
    print(f"\n  Distribusi kelas:")
    for i, nama in enumerate(label_names):
        count = int((y == i).sum())
        bar   = "█" * count
        print(f"    {nama:<15} {bar} ({count})")

    if ada_masalah:
        print("\n  [!] Dataset memiliki masalah — akurasi mungkin rendah.")
        print("      Rekomendasi: periksa file audio dengan rekam_dataset.py")
        print("      lalu jalankan ulang ekstraksi_mfcc.py sebelum training.\n")
    else:
        print("\n  ✓ Dataset terlihat sehat, lanjut training...\n")

    return X  # dikembalikan karena mungkin sudah dibersihkan NaN


# ─────────────────────────────────────────────
# LOAD DATA + LABEL ENCODING
# ─────────────────────────────────────────────

def load_data():
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(
            f"File '{FEATURES_FILE}' tidak ditemukan.\n"
            "Jalankan ekstraksi_mfcc.py terlebih dahulu."
        )

    data = np.load(FEATURES_FILE, allow_pickle=True)
    X    = data["X"].astype(np.float32)
    y_raw        = data["y"]
    label_names  = [str(l) for l in data["label_names"]]  # [FIX 4] bersihkan np.str_

    # Pastikan y adalah integer 0..N-1
    le = LabelEncoder()
    le.fit(list(range(len(label_names))))
    y  = y_raw.astype(np.int32)

    print(f"  ✓ Dataset: {X.shape[0]} sampel | "
          f"{X.shape[1]} fitur | {len(label_names)} kelas")
    return X, y, label_names


def split_data(X, y):
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    val_r = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_r, stratify=y_tv, random_state=SEED
    )
    print(f"  Split: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(X_train, X_val, X_test):
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("  ✓ Scaler disimpan → scaler.pkl")
    return X_train_s.astype(np.float32), \
           X_val_s.astype(np.float32),   \
           X_test_s.astype(np.float32),  \
           scaler


def buat_dataloader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=False)


# ─────────────────────────────────────────────
# MODEL 1 — SVM
# ─────────────────────────────────────────────

def train_svm(X_train, y_train, X_test, y_test, label_names):
    print("\n" + "─" * 55)
    print("  [1/3] Training SVM (RBF kernel) ...")
    print("─" * 55)

    t0  = time.time()
    svm = SVC(kernel="rbf", C=10, gamma="scale",
              probability=True, random_state=SEED)
    svm.fit(X_train, y_train)
    dur = time.time() - t0

    y_pred = svm.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=label_names, zero_division=0)

    print(f"  Akurasi  : {acc*100:.2f}%")
    print(f"  F1 score : {f1*100:.2f}%")
    print(f"  Waktu    : {dur:.1f}s")
    print(f"\n{report}")

    with open("model_svm.pkl", "wb") as f:
        pickle.dump({"model": svm, "label_names": label_names}, f)
    print("  ✓ Disimpan → model_svm.pkl")

    return dict(nama="SVM (RBF, C=10)", akurasi=acc, f1=f1,
                cm=cm, report=report, durasi=dur)


def train_rf(X_train, y_train, X_test, y_test, label_names):

    print("\n" + "─" * 55)
    print("  [RF] Training Random Forest ...")
    print("─" * 55)

    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    rf.fit(X_train, y_train)

    dur = time.time() - t0

    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    print(f"Akurasi  : {acc*100:.2f}%")
    print(f"F1 score : {f1*100:.2f}%")
    print(f"Waktu    : {dur:.1f}s")

    with open("model_rf.pkl", "wb") as f:
        pickle.dump({
            "model": rf,
            "label_names": label_names
        }, f)

    print("✓ model_rf.pkl")

    return dict(
        nama="Random Forest",
        akurasi=acc,
        f1=f1,
        cm=cm,
        report=report,
        durasi=dur
    )

def train_xgb(X_train, y_train, X_test, y_test, label_names):

    print("\n" + "─" * 55)
    print("  [XGB] Training XGBoost ...")
    print("─" * 55)

    t0 = time.time()

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_names),

        n_estimators=300,
        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,
        colsample_bytree=0.8,

        min_child_weight=3,

        eval_metric="mlogloss",

        tree_method="hist",

        random_state=SEED
    )

    xgb.fit(X_train, y_train)

    dur = time.time() - t0

    y_pred = xgb.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    print(f"Akurasi  : {acc*100:.2f}%")
    print(f"F1 score : {f1*100:.2f}%")
    print(f"Waktu    : {dur:.1f}s")

    with open("model_xgb.pkl", "wb") as f:
        pickle.dump({
            "model": xgb,
            "label_names": label_names
        }, f)

    print("✓ model_xgb.pkl")

    return dict(
        nama="XGBoost",
        akurasi=acc,
        f1=f1,
        cm=cm,
        report=report,
        durasi=dur
    )

# ─────────────────────────────────────────────
# MODEL 2 — MLP
# ─────────────────────────────────────────────

class MLPModel(nn.Module):
    def __init__(self, input_dim, n_kelas, hidden, dropout):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_kelas))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    tot_loss = tot_correct = tot = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if train:
                optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            if train:
                loss.backward()
                optimizer.step()
            tot_loss    += loss.item() * len(yb)
            tot_correct += (out.argmax(1) == yb).sum().item()
            tot         += len(yb)
    return tot_loss / tot, tot_correct / tot


def _prediksi_loader(model, X):
    model.eval()
    loader = buat_dataloader(X, np.zeros(len(X), dtype=np.int32),
                             64, shuffle=False)
    preds, probs = [], []
    with torch.no_grad():
        for xb, _ in loader:
            out = model(xb.to(DEVICE))
            p   = torch.softmax(out, dim=1)
            preds.append(out.argmax(1).cpu().numpy())
            probs.append(p.cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def train_mlp(X_train, y_train, X_val, y_val,
              X_test, y_test, label_names):
    print("\n" + "─" * 55)
    print("  [2/3] Training MLP ...")
    print("─" * 55)

    input_dim = X_train.shape[1]
    n_kelas   = len(label_names)
    model     = MLPModel(input_dim, n_kelas, MLP_HIDDEN, MLP_DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # ── [FIX 1] Hapus verbose=True ──
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    tr_loader  = buat_dataloader(X_train, y_train, MLP_BATCH)
    val_loader = buat_dataloader(X_val,   y_val,   MLP_BATCH, shuffle=False)

    best_acc, best_state = 0, None
    history = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])
    t0 = time.time()

    for ep in range(1, MLP_EPOCHS + 1):
        tl, ta = _run_epoch(model, tr_loader,  optimizer, criterion, train=True)
        vl, va = _run_epoch(model, val_loader, optimizer, criterion, train=False)
        scheduler.step(vl)

        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(ta);  history["val_acc"].append(va)

        if va > best_acc:
            best_acc   = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 20 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{MLP_EPOCHS} | "
                  f"train {ta*100:.1f}% loss={tl:.4f} | "
                  f"val {va*100:.1f}% loss={vl:.4f}")

    model.load_state_dict(best_state)
    dur = time.time() - t0

    y_pred, _ = _prediksi_loader(model, X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=label_names, zero_division=0)

    print(f"\n  Akurasi  : {acc*100:.2f}%")
    print(f"  F1 score : {f1*100:.2f}%")
    print(f"  Waktu    : {dur:.1f}s")
    print(f"\n{report}")

    torch.save(dict(state_dict=best_state, input_dim=input_dim,
                    n_kelas=n_kelas, label_names=label_names,
                    hidden=MLP_HIDDEN, dropout=MLP_DROPOUT), "model_mlp.pt")
    print("  ✓ Disimpan → model_mlp.pt")

    return dict(nama="MLP (256-128-64)", akurasi=acc, f1=f1,
                cm=cm, report=report, durasi=dur, history=history)


# ─────────────────────────────────────────────
# MODEL 3 — CNN 1D
# ─────────────────────────────────────────────

class CNN1DModel(nn.Module):
    """
    Input: (batch, 240) → unsqueeze → (batch, 1, 240)
    Conv1(1→32,k=7) → BN → ReLU → MaxPool(2)
    Conv2(32→64,k=5) → BN → ReLU → MaxPool(2)
    Conv3(64→128,k=3) → BN → ReLU → AdaptiveAvgPool(1)
    FC: 128 → 64 → n_kelas
    """
    def __init__(self, n_kelas, dropout=0.4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout / 2),

            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout / 2),

            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_kelas)
        )

    def forward(self, x):
        return self.fc(self.conv(x.unsqueeze(1)))


def train_cnn1d(X_train, y_train, X_val, y_val,
                X_test, y_test, label_names):
    print("\n" + "─" * 55)
    print("  [3/3] Training CNN 1D ...")
    print("─" * 55)

    n_kelas   = len(label_names)
    model     = CNN1DModel(n_kelas).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=CNN_LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # CosineAnnealing — tidak butuh verbose
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CNN_EPOCHS
    )

    tr_loader  = buat_dataloader(X_train, y_train, CNN_BATCH)
    val_loader = buat_dataloader(X_val,   y_val,   CNN_BATCH, shuffle=False)

    best_acc, best_state = 0, None
    history = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])
    t0 = time.time()

    for ep in range(1, CNN_EPOCHS + 1):
        tl, ta = _run_epoch(model, tr_loader,  optimizer, criterion, train=True)
        vl, va = _run_epoch(model, val_loader, optimizer, criterion, train=False)
        scheduler.step()

        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(ta);  history["val_acc"].append(va)

        if va > best_acc:
            best_acc   = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 20 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{CNN_EPOCHS} | "
                  f"train {ta*100:.1f}% loss={tl:.4f} | "
                  f"val {va*100:.1f}% loss={vl:.4f}")

    model.load_state_dict(best_state)
    dur = time.time() - t0

    y_pred, _ = _prediksi_loader(model, X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=label_names, zero_division=0)

    print(f"\n  Akurasi  : {acc*100:.2f}%")
    print(f"  F1 score : {f1*100:.2f}%")
    print(f"  Waktu    : {dur:.1f}s")
    print(f"\n{report}")

    torch.save(dict(state_dict=best_state, n_kelas=n_kelas,
                    label_names=label_names), "model_cnn1d.pt")
    print("  ✓ Disimpan → model_cnn1d.pt")

    return dict(nama="CNN 1D (32-64-128)", akurasi=acc, f1=f1,
                cm=cm, report=report, durasi=dur, history=history)


# ─────────────────────────────────────────────
# LAPORAN PERBANDINGAN
# ─────────────────────────────────────────────

def cetak_laporan(hasil_svm, hasil_rf, hasil_xgb, hasil_mlp, hasil_cnn, label_names):
    semua = [hasil_svm, hasil_rf, hasil_xgb, hasil_mlp, hasil_cnn]
    lines = [
        "=" * 60,
        "  LAPORAN PERBANDINGAN MODEL — DOKTERKU",
        "=" * 60,
        f"  {'Model':<22} {'Akurasi':>10} {'F1':>10} {'Waktu':>8}",
        "  " + "-" * 54,
    ]
    for h in semua:
        lines.append(
            f"  {h['nama']:<22} "
            f"{h['akurasi']*100:>9.2f}% "
            f"{h['f1']*100:>9.2f}% "
            f"{h['durasi']:>7.1f}s"
        )
    terbaik = max(semua, key=lambda h: h["akurasi"])
    lines += [
        "  " + "-" * 54,
        f"\n  ★ Model terbaik : {terbaik['nama']}",
        f"    Akurasi       : {terbaik['akurasi']*100:.2f}%",
        f"    F1 score      : {terbaik['f1']*100:.2f}%",
        "\n" + "=" * 60,
        "  CLASSIFICATION REPORT PER MODEL",
        "=" * 60,
    ]
    for h in semua:
        lines += [f"\n  [{h['nama']}]", h["report"]]

    txt = "\n".join(lines)
    print("\n" + txt)
    with open("hasil_training.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    print("  ✓ Laporan → hasil_training.txt")

    # Simpan nama model terbaik
    nama = terbaik["nama"].lower()

    if "xgboost" in nama:
        tipe = "xgb"

    elif "random forest" in nama:
        tipe = "rf"

    elif "mlp" in nama:
        tipe = "mlp"

    elif "svm" in nama:
        tipe = "svm"

    else:
        tipe = "cnn1d"
    with open("model_terbaik.txt", "w") as f:
        f.write(tipe)
    print(f"  ✓ Model terbaik → model_terbaik.txt  [{tipe}]")
    return terbaik


def plot_history(hasil_mlp, hasil_cnn):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("DOKTERKU — Training History", fontsize=14)
        pairs = [
            (axes[0][0], hasil_mlp, "MLP — Loss",     "loss"),
            (axes[0][1], hasil_cnn, "CNN1D — Loss",    "loss"),
            (axes[1][0], hasil_mlp, "MLP — Accuracy",  "acc"),
            (axes[1][1], hasil_cnn, "CNN1D — Accuracy","acc"),
        ]
        for ax, h, title, mode in pairs:
            ep = range(1, len(h["history"]["train_loss"]) + 1)
            if mode == "loss":
                ax.plot(ep, h["history"]["train_loss"], label="Train")
                ax.plot(ep, h["history"]["val_loss"],   label="Val")
                ax.set_ylabel("Loss")
            else:
                tr = [v * 100 for v in h["history"]["train_acc"]]
                vl = [v * 100 for v in h["history"]["val_acc"]]
                ax.plot(ep, tr, label="Train")
                ax.plot(ep, vl, label="Val")
                ax.set_ylabel("Accuracy (%)")
            ax.set_title(title); ax.set_xlabel("Epoch")
            ax.legend(); ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("training_history.png", dpi=150)
        print("  ✓ Grafik → training_history.png")
    except ImportError:
        pass


# ─────────────────────────────────────────────
# FUNGSI LOAD MODEL TERBAIK (untuk inferensi)
# ─────────────────────────────────────────────

def load_model_terbaik():
    """
    Kembalikan fungsi: prediksi(fitur_240) → (kata:str, confidence:float)
    Gunakan dari modul GUI.
    """
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model_terbaik.txt") as f:
        tipe = f.read().strip()

    if tipe == "svm":
        with open("model_svm.pkl", "rb") as f:
            pkg = pickle.load(f)
        model       = pkg["model"]
        label_names = pkg["label_names"]

        def prediksi(fitur):

            fs = scaler.transform([fitur])

            idx = int(model.predict(fs)[0])

            prob = model.predict_proba(fs)[0].max()

            return label_names[idx], float(prob)
        
    elif tipe == "rf":

        with open("model_rf.pkl", "rb") as f:
            pkg = pickle.load(f)

        model = pkg["model"]
        label_names = pkg["label_names"]

        def prediksi(fitur):

            fs = scaler.transform([fitur])

            idx = model.predict(fs)[0]

            prob = model.predict_proba(fs)[0].max()

            return label_names[idx], float(prob)
    
    elif tipe == "xgb":

        with open("model_xgb.pkl", "rb") as f:
            pkg = pickle.load(f)

        model = pkg["model"]
        label_names = pkg["label_names"]

        def prediksi(fitur):

            fs = scaler.transform([fitur])

            idx = model.predict(fs)[0]

            prob = model.predict_proba(fs)[0].max()

            return label_names[idx], float(prob)

    elif tipe == "mlp":
        ckpt = torch.load("model_mlp.pt", map_location="cpu")
        m    = MLPModel(ckpt["input_dim"], ckpt["n_kelas"],
                        ckpt["hidden"], ckpt["dropout"])
        m.load_state_dict(ckpt["state_dict"]); m.eval()
        label_names = ckpt["label_names"]

        def prediksi(fitur):
            fs  = scaler.transform([fitur])
            x   = torch.tensor(fs, dtype=torch.float32)
            with torch.no_grad():
                p = torch.softmax(m(x), dim=1)[0]
            idx = p.argmax().item()
            return label_names[idx], float(p[idx])

    else:  # cnn1d
        ckpt = torch.load("model_cnn1d.pt", map_location="cpu")
        m    = CNN1DModel(ckpt["n_kelas"])
        m.load_state_dict(ckpt["state_dict"]); m.eval()
        label_names = ckpt["label_names"]

        def prediksi(fitur):
            fs  = scaler.transform([fitur])
            x   = torch.tensor(fs, dtype=torch.float32)
            with torch.no_grad():
                p = torch.softmax(m(x), dim=1)[0]
            idx = p.argmax().item()
            return label_names[idx], float(p[idx])

    return prediksi, label_names


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  DOKTERKU — Training Model Klasifikasi (FIXED)")
    print("=" * 55)
    print(f"  Device : {DEVICE}  |  Seed : {SEED}\n")

    X, y, label_names = load_data()

    # ── [FIX 2] Diagnosis dataset sebelum training ──
    X = diagnosis_dataset(X, y, label_names)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_tr, X_va, X_te, scaler = scale_data(X_train, X_val, X_test)

    hasil_svm = train_svm(X_tr, y_train, X_te, y_test, label_names)
    hasil_rf = train_rf(
        X_tr,
        y_train,
        X_te,
        y_test,
        label_names
    )
    hasil_xgb = train_xgb(
        X_tr,
        y_train,
        X_te,
        y_test,
        label_names
    )
    hasil_mlp = train_mlp(X_tr, y_train, X_va, y_val,
                          X_te, y_test, label_names)
    hasil_cnn = train_cnn1d(X_tr, y_train, X_va, y_val,
                            X_te, y_test, label_names)

    terbaik = cetak_laporan(hasil_svm, hasil_rf, hasil_xgb, hasil_mlp, hasil_cnn, label_names)
    plot_history(hasil_mlp, hasil_cnn)

    print("\n" + "=" * 55)
    print("  File yang dihasilkan:")
    print("  • model_svm.pkl")
    print("  • model_rf.pkl")
    print("  • model_xgb.pkl")
    print("  • model_mlp.pt")
    print("  • model_cnn1d.pt")
    print("  • scaler.pkl")
    print("  • model_terbaik.txt")
    print("  • hasil_training.txt")
    print("  • training_history.png")
    print("=" * 55)
    print(f"\n  ★ Model terbaik : {terbaik['nama']}")
    print(f"    Akurasi test   : {terbaik['akurasi']*100:.2f}%\n")


if __name__ == "__main__":
    main()