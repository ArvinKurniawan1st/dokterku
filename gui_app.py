import os
import io
import time
import threading
import tempfile
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import librosa

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

try:
    import edge_tts
    import asyncio
    EDGE_TTS_OK = True
except ImportError:
    EDGE_TTS_OK = False



TTS_OK = EDGE_TTS_OK

try:
    import pygame
    pygame.mixer.init()
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False

try:
    import soundfile as sf
    SF_OK = True
except ImportError:
    SF_OK = False

# ── Konfigurasi suara edge-tts ──
# Suara Microsoft Neural untuk Bahasa Indonesia
EDGE_VOICES = {
    "perempuan": "id-ID-GadisNeural",   # Suara perempuan Indonesia
    "laki-laki": "id-ID-ArdiNeural",    # Suara laki-laki Indonesia
}
# Rate bicara edge-tts (persentase dari normal)
EDGE_RATE = {
    "lambat": "-20%",
    "normal": "+0%",
    "cepat":  "+20%",
}

# Import modul inferensi lokal
try:
    from inferensi import InferensiASR, KNOWLEDGE_BASE, URGENSI_WARNA
    INFERENSI_OK = True
except ImportError as e:
    INFERENSI_OK = False
    _INFERENSI_ERR = str(e)

# ─────────────────────────────────────────────
# TEMA & WARNA
# ─────────────────────────────────────────────
TEMA = {
    "bg":           "#0F1117",
    "bg_panel":     "#1A1D27",
    "bg_card":      "#22263A",
    "bg_input":     "#2A2F45",
    "accent":       "#4F8EF7",
    "accent2":      "#34C98A",
    "accent_warn":  "#F5A623",
    "accent_danger":"#E05252",
    "text_primary": "#F0F2FF",
    "text_secondary":"#8B90A8",
    "text_muted":   "#555A70",
    "border":       "#2E3350",
    "urgensi_rendah":"#34C98A",
    "urgensi_sedang":"#F5A623",
    "urgensi_tinggi":"#E05252",
}

FONT_JUDUL  = ("Segoe UI", 13, "bold")
FONT_NORMAL = ("Segoe UI", 10)
FONT_KECIL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)
FONT_BESAR  = ("Segoe UI", 22, "bold")
FONT_MEDIUM = ("Segoe UI", 14, "bold")

# ─────────────────────────────────────────────
# HELPER WIDGETS
# ─────────────────────────────────────────────

def label(parent, text, font=FONT_NORMAL, fg=None, bg=None, **kw):
    return tk.Label(parent, text=text, font=font,
                    fg=fg or TEMA["text_primary"],
                    bg=bg or TEMA["bg_panel"], **kw)


def card_frame(parent, **kw):
    return tk.Frame(parent, bg=TEMA["bg_card"],
                    highlightthickness=1,
                    highlightbackground=TEMA["border"], **kw)


def section_label(parent, text):
    f = tk.Frame(parent, bg=TEMA["bg_panel"])
    tk.Label(f, text=text, font=("Segoe UI", 9, "bold"),
             fg=TEMA["text_secondary"], bg=TEMA["bg_panel"],
             padx=0).pack(side="left")
    tk.Frame(f, bg=TEMA["border"], height=1).pack(
        side="left", fill="x", expand=True, padx=(8, 0), pady=6)
    return f


def tombol(parent, text, command, color=None, width=None, **kw):
    c = color or TEMA["accent"]
    b = tk.Button(parent, text=text, command=command,
                  bg=c, fg=TEMA["text_primary"],
                  font=("Segoe UI", 9, "bold"),
                  relief="flat", bd=0, cursor="hand2",
                  padx=14, pady=7,
                  activebackground=c,
                  activeforeground=TEMA["text_primary"], **kw)
    if width:
        b.config(width=width)
    return b


# ─────────────────────────────────────────────
# KELAS UTAMA GUI
# ─────────────────────────────────────────────

class DokterKuApp:

    def __init__(self, root):
        self.root = root
        self.root.title("DOKTERKU — Asisten Kesehatan Suara")
        self.root.configure(bg=TEMA["bg"])
        self.root.geometry("1360x780")
        self.root.minsize(1100, 650)

        # State
        self.sedang_rekam   = False
        self.audio_terakhir = None
        self.mfcc_terakhir  = None
        self.tts_file_tmp   = None

        # Variabel tkinter
        self.var_kecepatan = tk.StringVar(value="normal")
        self.var_gender    = tk.StringVar(value="perempuan")
        self.var_status    = tk.StringVar(value="Siap")
        self.var_conf      = tk.DoubleVar(value=0)

        # Inisialisasi ASR
        self.asr = None
        if INFERENSI_OK:
            try:
                self.asr = InferensiASR()
            except Exception as e:
                self.asr = None

        self._build_ui()
        self._update_status("Siap — ucapkan kata gejala atau ketik teks untuk TTS")

    # ──────────────────────────────────────────
    # BUILD UI
    # ──────────────────────────────────────────

    def _build_ui(self):
        # Header
        self._build_header()

        # Body: 3 kolom
        body = tk.Frame(self.root, bg=TEMA["bg"])
        body.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        body.columnconfigure(0, weight=4)
        body.columnconfigure(1, weight=4)
        body.columnconfigure(2, weight=3)
        body.rowconfigure(0, weight=1)

        # Panel kiri — ASR
        self._build_panel_asr(body)

        # Panel tengah — TTS
        self._build_panel_tts(body)

        # Panel kanan — Visualisasi + Top3
        self._build_panel_visual(body)

        # Status bar
        self._build_statusbar()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=TEMA["bg_panel"],
                       highlightthickness=1,
                       highlightbackground=TEMA["border"])
        hdr.pack(fill="x", padx=16, pady=(12, 8))

        tk.Label(hdr, text="🏥", font=("Segoe UI", 18),
                 bg=TEMA["bg_panel"], fg=TEMA["accent"]).pack(side="left", padx=(14, 6))
        tk.Label(hdr, text="DOKTERKU", font=("Segoe UI", 15, "bold"),
                 bg=TEMA["bg_panel"], fg=TEMA["text_primary"]).pack(side="left")
        tk.Label(hdr, text="Asisten Kesehatan Berbasis Suara",
                 font=("Segoe UI", 9),
                 bg=TEMA["bg_panel"], fg=TEMA["text_secondary"]).pack(side="left", padx=10)


    # ── Panel ASR ──
    def _build_panel_asr(self, parent):
        panel = tk.Frame(parent, bg=TEMA["bg_panel"],
                         highlightthickness=1,
                         highlightbackground=TEMA["border"])
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        tk.Label(panel, text="🎙  ASR — Pengenalan Suara",
                 font=FONT_JUDUL, bg=TEMA["bg_panel"],
                 fg=TEMA["accent"]).pack(anchor="w", padx=14, pady=(10, 4))

        # Tombol rekam
        btn_frame = tk.Frame(panel, bg=TEMA["bg_panel"])
        btn_frame.pack(fill="x", padx=14, pady=(0, 8))

        self.btn_rekam = tombol(btn_frame, "⏺  REKAM SEKARANG",
                                self._mulai_rekam,
                                color=TEMA["accent"], width=22)
        self.btn_rekam.pack(side="left")

        self.lbl_countdown = tk.Label(btn_frame, text="",
                                      font=("Segoe UI", 14, "bold"),
                                      bg=TEMA["bg_panel"],
                                      fg=TEMA["accent_warn"])
        self.lbl_countdown.pack(side="left", padx=12)

        # Hasil prediksi utama
        section_label(panel, "HASIL PREDIKSI").pack(fill="x", padx=14, pady=(2, 0))

        hasil_card = card_frame(panel)
        hasil_card.pack(fill="x", padx=14, pady=(4, 6))

        self.lbl_kata = tk.Label(hasil_card, text="—",
                                 font=("Segoe UI", 26, "bold"),
                                 bg=TEMA["bg_card"], fg=TEMA["text_primary"])
        self.lbl_kata.pack(pady=(8, 2))

        self.lbl_emoji = tk.Label(hasil_card, text="",
                                  font=("Segoe UI", 16),
                                  bg=TEMA["bg_card"], fg=TEMA["text_primary"])
        self.lbl_emoji.pack(pady=(0, 8))

        # Confidence bar
        section_label(panel, "CONFIDENCE SCORE").pack(fill="x", padx=14, pady=(2, 0))
        conf_card = card_frame(panel)
        conf_card.pack(fill="x", padx=14, pady=(4, 6))

        conf_inner = tk.Frame(conf_card, bg=TEMA["bg_card"])
        conf_inner.pack(fill="x", padx=12, pady=10)

        self.lbl_conf_pct = tk.Label(conf_inner, text="0%",
                                     font=("Segoe UI", 13, "bold"),
                                     bg=TEMA["bg_card"], fg=TEMA["accent"])
        self.lbl_conf_pct.pack(side="right")

        bar_bg = tk.Frame(conf_inner, bg=TEMA["bg_input"], height=12)
        bar_bg.pack(fill="x", side="left", expand=True, pady=4)
        bar_bg.pack_propagate(False)
        self.bar_conf = tk.Frame(bar_bg, bg=TEMA["accent"], height=12, width=0)
        self.bar_conf.place(x=0, y=0, relheight=1, width=0)
        self._bar_bg_ref = bar_bg

        # Urgensi badge
        self.lbl_urgensi = tk.Label(conf_card, text="",
                                    font=("Segoe UI", 9, "bold"),
                                    bg=TEMA["bg_card"],
                                    fg=TEMA["text_secondary"])
        self.lbl_urgensi.pack(pady=(0, 8))

        # Saran
        section_label(panel, "SARAN KESEHATAN").pack(fill="x", padx=14, pady=(2, 0))
        saran_card = card_frame(panel)
        saran_card.pack(fill="both", expand=True, padx=14, pady=(4, 8))

        self.txt_saran = tk.Text(saran_card, wrap="word",
                                 bg=TEMA["bg_card"], fg=TEMA["text_primary"],
                                 font=("Segoe UI", 10), relief="flat",
                                 padx=12, pady=10,
                                 state="disabled", cursor="arrow",
                                 highlightthickness=0)
        self.txt_saran.pack(fill="both", expand=True)

        # Tombol kirim ke TTS
        self.btn_ke_tts = tombol(panel, "➡  Kirim Saran ke TTS",
                                 self._kirim_ke_tts,
                                 color=TEMA["accent2"])
        self.btn_ke_tts.pack(pady=(4, 10), padx=14, anchor="w")

    # ── Panel TTS ──
    def _build_panel_tts(self, parent):
        panel = tk.Frame(parent, bg=TEMA["bg_panel"],
                         highlightthickness=1,
                         highlightbackground=TEMA["border"])
        panel.grid(row=0, column=1, sticky="nsew", padx=(0, 6))

        tk.Label(panel, text="🔊  TTS — Teks ke Suara",
                 font=FONT_JUDUL, bg=TEMA["bg_panel"],
                 fg=TEMA["accent2"]).pack(anchor="w", padx=14, pady=(10, 4))

        # Input teks
        section_label(panel, "TEKS INPUT").pack(fill="x", padx=14, pady=(2, 0))
        input_card = card_frame(panel)
        input_card.pack(fill="x", padx=14, pady=(4, 6))

        self.txt_input = tk.Text(input_card, wrap="word", height=6,
                                 bg=TEMA["bg_input"], fg=TEMA["text_primary"],
                                 font=("Segoe UI", 10), relief="flat",
                                 padx=12, pady=10, insertbackground=TEMA["text_primary"],
                                 highlightthickness=0)
        self.txt_input.pack(fill="both", padx=2, pady=2)
        self.txt_input.insert("1.0", "Ketik teks bahasa Indonesia di sini...")
        self.txt_input.bind("<FocusIn>",  self._clear_placeholder)
        self.txt_input.bind("<FocusOut>", self._restore_placeholder)

        # Opsi TTS
        section_label(panel, "PENGATURAN SUARA").pack(fill="x", padx=14, pady=(2, 0))
        opt_card = card_frame(panel)
        opt_card.pack(fill="x", padx=14, pady=(4, 6))

        opt_inner = tk.Frame(opt_card, bg=TEMA["bg_card"])
        opt_inner.pack(fill="x", padx=12, pady=8)

        # Gender
        tk.Label(opt_inner, text="Gender :", font=FONT_KECIL,
                 bg=TEMA["bg_card"], fg=TEMA["text_secondary"]).grid(
                 row=0, column=0, sticky="w", pady=4)

        for i, (val, txt) in enumerate([("perempuan", "👩 Perempuan"),
                                         ("laki-laki",  "👨 Laki-laki")]):
            tk.Radiobutton(opt_inner, text=txt, variable=self.var_gender,
                           value=val, font=FONT_KECIL,
                           bg=TEMA["bg_card"], fg=TEMA["text_primary"],
                           selectcolor=TEMA["bg_input"],
                           activebackground=TEMA["bg_card"],
                           relief="flat").grid(row=0, column=i+1, padx=6)

        # Kecepatan
        tk.Label(opt_inner, text="Kecepatan:", font=FONT_KECIL,
                 bg=TEMA["bg_card"], fg=TEMA["text_secondary"]).grid(
                 row=1, column=0, sticky="w", pady=4)

        for i, (val, txt) in enumerate([("lambat", "🐢 Lambat"),
                                         ("normal", "🚶 Normal"),
                                         ("cepat",  "🏃 Cepat")]):
            tk.Radiobutton(opt_inner, text=txt, variable=self.var_kecepatan,
                           value=val, font=FONT_KECIL,
                           bg=TEMA["bg_card"], fg=TEMA["text_primary"],
                           selectcolor=TEMA["bg_input"],
                           activebackground=TEMA["bg_card"],
                           relief="flat").grid(row=1, column=i+1, padx=4)

        # Tombol TTS
        btn_tts_row = tk.Frame(panel, bg=TEMA["bg_panel"])
        btn_tts_row.pack(fill="x", padx=14, pady=(0, 6))

        tombol(btn_tts_row, "▶  Putar Suara",
               self._putar_tts, color=TEMA["accent2"]).pack(side="left", padx=(0, 8))
        tombol(btn_tts_row, "💾  Simpan Audio",
               self._simpan_audio, color=TEMA["bg_input"]).pack(side="left")

        # Preview audio
        section_label(panel, "STATUS TTS").pack(fill="x", padx=14, pady=(2, 0))
        status_card = card_frame(panel)
        status_card.pack(fill="x", padx=14, pady=(4, 6))

        self.lbl_tts_status = tk.Label(status_card,
                                       text="Belum ada audio dihasilkan.",
                                       font=FONT_KECIL, wraplength=320,
                                       justify="left",
                                       bg=TEMA["bg_card"],
                                       fg=TEMA["text_secondary"])
        self.lbl_tts_status.pack(padx=12, pady=8, anchor="w")

        # Progress bar TTS
        self.tts_progress = ttk.Progressbar(panel, mode="indeterminate",
                                            length=200)
        self.tts_progress.pack(padx=14, pady=(0, 6), fill="x")
        self.tts_progress.stop()

        # Riwayat TTS
        section_label(panel, "RIWAYAT").pack(fill="x", padx=14, pady=(2, 0))
        riwayat_card = card_frame(panel)
        riwayat_card.pack(fill="both", expand=True, padx=14, pady=(4, 10))

        self.txt_riwayat = tk.Text(riwayat_card, wrap="word",
                                   bg=TEMA["bg_card"], fg=TEMA["text_secondary"],
                                   font=FONT_MONO, relief="flat",
                                   padx=10, pady=8, state="disabled",
                                   highlightthickness=0)
        self.txt_riwayat.pack(fill="both", expand=True)

    # ── Panel visualisasi ──
    def _build_panel_visual(self, parent):
        panel = tk.Frame(parent, bg=TEMA["bg_panel"],
                         highlightthickness=1,
                         highlightbackground=TEMA["border"])
        panel.grid(row=0, column=2, sticky="nsew")

        tk.Label(panel, text="📊  Analisis",
                 font=FONT_JUDUL, bg=TEMA["bg_panel"],
                 fg=TEMA["accent_warn"]).pack(anchor="w", padx=14, pady=(10, 4))

        # Top 3 prediksi
        section_label(panel, "TOP 3 PREDIKSI").pack(fill="x", padx=14, pady=(2, 0))
        top3_card = card_frame(panel)
        top3_card.pack(fill="x", padx=14, pady=(4, 6))

        self.top3_frames = []
        for i in range(3):
            row = tk.Frame(top3_card, bg=TEMA["bg_card"])
            row.pack(fill="x", padx=10, pady=(5 if i == 0 else 2, 0))

            lbl_rank = tk.Label(row, text=f"#{i+1}",
                                font=("Segoe UI", 9, "bold"),
                                bg=TEMA["bg_card"],
                                fg=[TEMA["accent"], TEMA["text_secondary"],
                                    TEMA["text_muted"]][i],
                                width=3)
            lbl_rank.pack(side="left")

            lbl_nm = tk.Label(row, text="—", font=FONT_KECIL,
                              bg=TEMA["bg_card"], fg=TEMA["text_primary"],
                              width=12, anchor="w")
            lbl_nm.pack(side="left")

            bar_bg = tk.Frame(row, bg=TEMA["bg_input"], height=8)
            bar_bg.pack(side="left", fill="x", expand=True, padx=4)
            bar_fill = tk.Frame(bar_bg, bg=[TEMA["accent"], TEMA["accent2"],
                                            TEMA["accent_warn"]][i], height=8)
            bar_fill.place(x=0, y=0, relheight=1, width=0)

            lbl_pct = tk.Label(row, text="0%", font=FONT_MONO,
                               bg=TEMA["bg_card"], fg=TEMA["text_secondary"],
                               width=5)
            lbl_pct.pack(side="left")

            self.top3_frames.append((lbl_nm, bar_fill, bar_bg, lbl_pct))

        # spacer
        tk.Frame(top3_card, bg=TEMA["bg_card"], height=6).pack()

        # MFCC plot — di bawah Top 3
        section_label(panel, "VISUALISASI MFCC").pack(fill="x", padx=14, pady=(6, 0))

        mfcc_card = card_frame(panel)
        mfcc_card.pack(fill="both", expand=True, padx=14, pady=(4, 10))

        if MATPLOTLIB_OK:
            self.fig, self.ax = plt.subplots(figsize=(3.6, 2.4), dpi=80)
            self.fig.patch.set_facecolor(TEMA["bg_card"])
            self.ax.set_facecolor(TEMA["bg_input"])
            self.ax.set_title("MFCC (40 koef × frame)", fontsize=7, color=TEMA["text_secondary"])
            self.ax.set_ylabel("Koefisien", fontsize=6, color=TEMA["text_muted"])
            self.ax.set_xlabel("Frame", fontsize=6, color=TEMA["text_muted"])
            self.ax.tick_params(colors=TEMA["text_muted"], labelsize=5)
            self.fig.tight_layout(pad=0.5)
            self.canvas_mfcc = FigureCanvasTkAgg(self.fig, master=mfcc_card)
            self.canvas_mfcc.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        else:
            tk.Label(mfcc_card, text="matplotlib tidak tersedia\npip install matplotlib",
                     font=FONT_KECIL, bg=TEMA["bg_card"],
                     fg=TEMA["text_muted"]).pack(pady=20)


    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=TEMA["bg_panel"],
                       highlightthickness=1,
                       highlightbackground=TEMA["border"], height=28)
        bar.pack(fill="x", padx=16, pady=(0, 8))
        bar.pack_propagate(False)
        tk.Label(bar, textvariable=self.var_status,
                 font=FONT_KECIL, bg=TEMA["bg_panel"],
                 fg=TEMA["text_secondary"]).pack(side="left", padx=12)

    # ──────────────────────────────────────────
    # AKSI — ASR
    # ──────────────────────────────────────────

    def _mulai_rekam(self):
        if self.sedang_rekam:
            return
        if not self.asr or not self.asr.siap:
            err = self.asr.error_msg if self.asr else "InferensiASR gagal diinisialisasi"
            messagebox.showerror("Model Tidak Siap",
                f"Model ASR tidak dapat dimuat.\n\n"
                f"Detail: {err}\n\n"
                "Pastikan file berikut ada di folder yang sama:\n"
                "  • model_terbaik.txt  (berisi: rf / svm / xgb / mlp)\n"
                "  • model_rf.pkl       (jika model terbaik = RF)\n"
                "  • model_svm.pkl      (jika model terbaik = SVM)\n"
                "  • model_xgb.pkl      (jika model terbaik = XGB)\n"
                "  • scaler.pkl\n\n"
                "Jalankan training_model.py terlebih dahulu.")
            return
        self.sedang_rekam = True
        self.btn_rekam.config(state="disabled", text="⏺  Merekam...")
        self._update_status("Merekam audio dari mikrofon...")
        threading.Thread(target=self._thread_rekam, daemon=True).start()

    def _countdown(self, sisa):
        if sisa > 0:
            self.lbl_countdown.config(text=f"{sisa}s")
        else:
            self.lbl_countdown.config(text="")

    def _thread_rekam(self):
        try:
            kata, conf, saran, urgensi, top3, audio = \
                self.asr.rekam_dan_prediksi(
                    callback_countdown=lambda s: self.root.after(0, self._countdown, s)
                )
            self.audio_terakhir = audio
            self.root.after(0, self._tampil_hasil_asr,
                            kata, conf, saran, urgensi, top3, audio)
        except Exception as e:
            self.root.after(0, self._update_status, f"Error: {e}")
        finally:
            self.sedang_rekam = False
            self.root.after(0, lambda: self.btn_rekam.config(
                state="normal", text="⏺  REKAM SEKARANG"))
            self.root.after(0, lambda: self.lbl_countdown.config(text=""))

    def _tampil_hasil_asr(self, kata, conf, saran, urgensi, top3, audio):
        # Kata utama
        kb_item = {}
        try:
            from inferensi import KNOWLEDGE_BASE
            kb_item = KNOWLEDGE_BASE.get(kata, {})
        except Exception:
            pass

        self.lbl_kata.config(text=kata.upper())
        self.lbl_emoji.config(text=kb_item.get("emoji", "🏥"))

        # Confidence bar
        conf_pct = int(conf * 100)
        self.lbl_conf_pct.config(text=f"{conf_pct}%")
        warna_conf = (TEMA["accent2"] if conf >= 0.7
                      else TEMA["accent_warn"] if conf >= 0.4
                      else TEMA["accent_danger"])
        self.bar_conf.config(bg=warna_conf)

        def _update_bar():
            w = self._bar_bg_ref.winfo_width()
            self.bar_conf.place(width=int(w * conf))
        self.root.after(50, _update_bar)

        # Urgensi
        warna_urg = {
            "rendah": TEMA["urgensi_rendah"],
            "sedang": TEMA["urgensi_sedang"],
            "tinggi": TEMA["urgensi_tinggi"],
        }.get(urgensi, TEMA["text_secondary"])
        self.lbl_urgensi.config(
            text=f"⚑ Urgensi: {urgensi.upper()}",
            fg=warna_urg
        )

        # Saran
        self.txt_saran.config(state="normal")
        self.txt_saran.delete("1.0", "end")
        self.txt_saran.insert("1.0", saran)
        self.txt_saran.config(state="disabled")

        # Top 3
        for i, (lbl_nm, bar_fill, bar_bg, lbl_pct) in enumerate(self.top3_frames):
            if i < len(top3):
                k, c = top3[i]
                lbl_nm.config(text=str(k))
                lbl_pct.config(text=f"{c:.0%}")
                def _set_bar(bf=bar_fill, bg_=bar_bg, c_=c):
                    w = bg_.winfo_width()
                    bf.place(width=int(w * c_))
                self.root.after(60, _set_bar)
            else:
                lbl_nm.config(text="—")
                lbl_pct.config(text="")

        # MFCC visualisasi
        if MATPLOTLIB_OK and audio is not None:
            self._plot_mfcc(audio)

        # Riwayat
        self._tambah_riwayat(f"[ASR] {kata.upper()} ({conf:.1%}) — {urgensi}")

        self._update_status(
            f"Prediksi: {kata.upper()} | Confidence: {conf:.1%} | Urgensi: {urgensi}")

    def _plot_mfcc(self, audio):
        try:
            hop = int(0.010 * 16000)
            win = int(0.025 * 16000)
            mfcc = librosa.feature.mfcc(y=audio.astype(np.float32),
                                        sr=16000, n_mfcc=40,
                                        hop_length=hop, win_length=win)
            self.ax.clear()
            self.ax.imshow(mfcc, aspect="auto", origin="lower",
                           cmap="magma", interpolation="nearest")
            self.ax.set_title("MFCC (40 koef × frame)", fontsize=7,
                              color=TEMA["text_secondary"])
            self.ax.set_ylabel("Koefisien", fontsize=6,
                               color=TEMA["text_muted"])
            self.ax.set_xlabel("Frame", fontsize=6,
                               color=TEMA["text_muted"])
            self.ax.tick_params(colors=TEMA["text_muted"], labelsize=5)
            self.fig.tight_layout(pad=0.4)
            self.canvas_mfcc.draw()
        except Exception as e:
            pass

    # ──────────────────────────────────────────
    # AKSI — TTS
    # ──────────────────────────────────────────

    def _get_teks(self):
        teks = self.txt_input.get("1.0", "end").strip()
        if teks in ("", "Ketik teks bahasa Indonesia di sini..."):
            return None
        return teks

    def _putar_tts(self):
        teks = self._get_teks()
        if not teks:
            messagebox.showwarning("TTS", "Masukkan teks terlebih dahulu.")
            return
        threading.Thread(target=self._thread_tts,
                         args=(teks, False), daemon=True).start()

    def _simpan_audio(self):
        teks = self._get_teks()
        if not teks:
            messagebox.showwarning("TTS", "Masukkan teks terlebih dahulu.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".mp3",
            filetypes=[("MP3", "*.mp3"), ("WAV", "*.wav"), ("Semua", "*.*")],
            title="Simpan Audio TTS"
        )
        if path:
            threading.Thread(target=self._thread_tts,
                             args=(teks, True, path), daemon=True).start()

    def _thread_tts(self, teks, simpan=False, save_path=None):
        """
        Generate audio TTS menggunakan edge-tts (prioritas) atau gTTS (fallback).
        edge-tts: support gender laki-laki/perempuan via suara Neural Indonesia.
        """
        self.root.after(0, self.tts_progress.start, 8)
        self.root.after(0, self._update_status, "Menghasilkan audio TTS...")

        try:
            kecepatan = self.var_kecepatan.get()   # lambat / normal / cepat
            gender    = self.var_gender.get()       # perempuan / laki-laki

            tmp_dir  = tempfile.gettempdir()
            tmp_path = os.path.join(
                tmp_dir, f"dokterku_tts_{int(time.time()*1000)}.mp3"
            )
            out_path = save_path if (simpan and save_path) else tmp_path

            if self.tts_file_tmp and os.path.exists(self.tts_file_tmp):
                try:
                    if PYGAME_OK:
                        pygame.mixer.music.stop()
                        pygame.mixer.music.unload()
                    os.remove(self.tts_file_tmp)
                except Exception:
                    pass

            if EDGE_TTS_OK:
                voice = EDGE_VOICES.get(gender, "id-ID-GadisNeural")
                rate  = EDGE_RATE.get(kecepatan, "+0%")

                async def _generate():
                    communicate = edge_tts.Communicate(
                        text=teks,
                        voice=voice,
                        rate=rate
                    )
                    await communicate.save(out_path)

                # Jalankan async di event loop baru
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_generate())
                loop.close()

                engine_used = f"edge-tts [{voice}]"


            # ── Verifikasi file berhasil dibuat ──
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                raise RuntimeError(
                    "File audio gagal dibuat. Periksa koneksi internet."
                )

            # ── Mode simpan ──
            if simpan and save_path:
                nama = os.path.basename(save_path)
                self.root.after(0, self.lbl_tts_status.config,
                                {"text": f"✓ Disimpan: {nama}"})
                self.root.after(0, self._tambah_riwayat,
                                f"[TTS] Simpan: {nama} [{engine_used}]")
                self.root.after(0, self._update_status,
                                f"Audio disimpan: {save_path}")

            # ── Mode putar ──
            else:
                self.tts_file_tmp = tmp_path

                if PYGAME_OK:
                    pygame.mixer.music.load(tmp_path)
                    pygame.mixer.music.play()
                    preview = teks[:50] + "..." if len(teks) > 50 else teks
                    self.root.after(0, self.lbl_tts_status.config,
                                    {"text": f"▶ Memutar [{gender}]: {preview}"})
                    riwayat = teks[:40] + "..." if len(teks) > 40 else teks
                    self.root.after(0, self._tambah_riwayat,
                                    f"[TTS] {gender} | {riwayat}")
                    self.root.after(0, self._update_status,
                                    f"TTS sedang diputar — {engine_used}")
                else:
                    self.root.after(0, self.lbl_tts_status.config,
                                    {"text": "⚠ pygame tidak tersedia — pip install pygame"})

        except Exception as e:
            err = str(e)
            if "getaddrinfo" in err or "Connection" in err or "Network" in err:
                err = "Tidak ada koneksi internet. edge-tts membutuhkan internet."
            elif "No such file" in err:
                err = f"File audio gagal dibuat. Detail: {err}"
            self.root.after(0, self.lbl_tts_status.config,
                            {"text": f"⚠ {err[:120]}"})
            self.root.after(0, self._update_status, f"Error TTS: {str(e)[:60]}")
        finally:
            self.root.after(0, self.tts_progress.stop)

    def _kirim_ke_tts(self):
        """Ambil saran dari panel ASR → kirim ke input TTS."""
        saran = self.txt_saran.get("1.0", "end").strip()
        if not saran or saran == "—":
            messagebox.showinfo("Info", "Belum ada saran. Rekam suara dulu.")
            return
        self.txt_input.config(state="normal")
        self.txt_input.delete("1.0", "end")
        self.txt_input.insert("1.0", saran)
        self._update_status("Saran dikirim ke panel TTS — klik Putar Suara")

    # ──────────────────────────────────────────
    # HELPER
    # ──────────────────────────────────────────

    def _update_status(self, msg):
        self.var_status.set(f"  ●  {msg}")

    def _tambah_riwayat(self, teks):
        waktu = time.strftime("%H:%M:%S")
        self.txt_riwayat.config(state="normal")
        self.txt_riwayat.insert("end", f"[{waktu}] {teks}\n")
        self.txt_riwayat.see("end")
        self.txt_riwayat.config(state="disabled")

    def _tulis_info(self, teks):
        self.txt_info.config(state="normal")
        self.txt_info.delete("1.0", "end")
        self.txt_info.insert("1.0", teks)
        self.txt_info.config(state="disabled")

    def _clear_placeholder(self, event):
        if self.txt_input.get("1.0", "end").strip() == \
                "Ketik teks bahasa Indonesia di sini...":
            self.txt_input.delete("1.0", "end")
            self.txt_input.config(fg=TEMA["text_primary"])

    def _restore_placeholder(self, event):
        if not self.txt_input.get("1.0", "end").strip():
            self.txt_input.insert("1.0", "Ketik teks bahasa Indonesia di sini...")
            self.txt_input.config(fg=TEMA["text_muted"])

    def on_close(self):
        if self.tts_file_tmp and os.path.exists(self.tts_file_tmp):
            try:
                os.unlink(self.tts_file_tmp)
            except Exception:
                pass
        self.root.destroy()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app  = DokterKuApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    # Style ttk progressbar
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TProgressbar",
                    troughcolor=TEMA["bg_input"],
                    background=TEMA["accent2"],
                    thickness=6)

    root.mainloop()