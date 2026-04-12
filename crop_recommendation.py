import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# ── Colors ────────────────────────────────────
BG       = "#F4F7F0"
CARD     = "#FFFFFF"
GREEN    = "#2E7D32"
GREEN_L  = "#66BB6A"
LIGHT_G  = "#E8F5E9"
TEXT     = "#1C2B1D"
TEXT_S   = "#607D63"
BORDER   = "#C8DEC9"

# ── Load & Train on startup ───────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")

def load_and_train():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found!\nPlace 'Crop_recommendation.csv' in:\n{BASE_DIR}"
        )
    df = pd.read_csv(CSV_PATH)
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[features].values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, scaler, df, features, round(acc * 100, 2)


# ── Main App ──────────────────────────────────
class CropApp(tk.Tk):
    CROP_EMOJI = {
        "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
        "pigeonpeas": "🫘", "mothbeans": "🫘", "mungbean": "🫘", "blackgram": "🫘",
        "lentil": "🌿", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
        "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
        "orange": "🍊", "papaya": "🍐", "coconut": "🥥", "cotton": "🌸",
        "jute": "🌿", "coffee": "☕",
    }

    def __init__(self, model, scaler, df, features, accuracy):
        super().__init__()
        self.model    = model
        self.scaler   = scaler
        self.df       = df
        self.features = features
        self.accuracy = accuracy

        self.title("Crop Recommendation System")
        self.geometry("1024x768")
        self.resizable(False, False)
        self.configure(bg=BG)
        self._center()
        self._build()

    def _center(self):
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - 1024) // 2
        y = (self.winfo_screenheight() - 768) // 2
        self.geometry(f"1024x768+{x}+{y}")

    # ── Build UI ──────────────────────────────
    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=GREEN, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🌾  Crop Recommendation System",
                 font=("Segoe UI", 17, "bold"), bg=GREEN, fg="white").pack()
        tk.Label(hdr, text=f"Random Forest  ·  Model Accuracy: {self.accuracy}%",
                 font=("Segoe UI", 9), bg=GREEN, fg=GREEN_L).pack()

        # Body — left inputs, right result
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        self._build_inputs(body)
        self._build_result(body)

    def _build_inputs(self, parent):
        left = tk.Frame(parent, bg=CARD,
                        highlightbackground=BORDER, highlightthickness=1)
        left.pack(side="left", fill="both", padx=(0, 8), pady=0, ipadx=12, ipady=10)

        tk.Label(left, text="Set Soil & Climate Parameters",
                 font=("Segoe UI", 11, "bold"), bg=CARD, fg=TEXT).pack(anchor="w", pady=(4, 10))

        self.sliders = {}
        params = [
            ("N",           "Nitrogen (N)",    0,   140,  60,  "mg/kg"),
            ("P",           "Phosphorus (P)",  0,   145,  40,  "mg/kg"),
            ("K",           "Potassium (K)",   0,   205,  50,  "mg/kg"),
            ("temperature", "Temperature",     5,   50,   28,  "°C"),
            ("humidity",    "Humidity",        10,  100,  65,  "%"),
            ("ph",          "Soil pH",         3.0, 10.0, 6.5, ""),
            ("rainfall",    "Rainfall",        20,  300,  100, "mm/yr"),
        ]
        for key, label, lo, hi, init, unit in params:
            self._make_slider(left, key, label, lo, hi, init, unit)

        # Predict button
        tk.Button(left, text="  Get Recommendation  ",
                  font=("Segoe UI", 11, "bold"),
                  bg=GREEN, fg="white", activebackground=GREEN_L,
                  relief="flat", cursor="hand2", pady=8,
                  command=self._predict).pack(fill="x", pady=(14, 4))

    def _make_slider(self, parent, key, label, lo, hi, init, unit):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=3)

        tk.Label(row, text=label, font=("Segoe UI", 9), bg=CARD,
                 fg=TEXT_S, width=14, anchor="w").pack(side="left")

        is_float = isinstance(lo, float)
        var = tk.DoubleVar(value=init)
        self.sliders[key] = var

        val_lbl = tk.Label(row, font=("Segoe UI", 9, "bold"), bg=CARD,
                           fg=GREEN, width=7, anchor="e")
        val_lbl.pack(side="right")
        tk.Label(row, text=unit, font=("Segoe UI", 8), bg=CARD,
                 fg=TEXT_S, width=5).pack(side="right")

        def update(*_):
            v = var.get()
            val_lbl.config(text=f"{v:.1f}" if is_float else f"{int(v)}")

        tk.Scale(row, from_=lo, to=hi, orient="horizontal", variable=var,
                 resolution=0.1 if is_float else 1,
                 bg=CARD, troughcolor=LIGHT_G, activebackground=GREEN,
                 highlightthickness=0, showvalue=False, sliderlength=12,
                 command=update).pack(side="left", fill="x", expand=True, padx=4)
        update()

    def _build_result(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.pack(side="right", fill="both", expand=True)

        # Result card
        res = tk.Frame(right, bg=CARD,
                       highlightbackground=BORDER, highlightthickness=1)
        res.pack(fill="x", ipadx=12, ipady=10)

        tk.Label(res, text="Recommendation",
                 font=("Segoe UI", 11, "bold"), bg=CARD, fg=TEXT).pack(anchor="w", padx=12, pady=(8, 0))

        self.emoji_lbl = tk.Label(res, text="🌱", font=("Segoe UI", 42), bg=CARD)
        self.emoji_lbl.pack(pady=(4, 0))

        self.crop_lbl = tk.Label(res, text="—",
                                  font=("Segoe UI", 18, "bold"), bg=CARD, fg=GREEN)
        self.crop_lbl.pack()

        self.conf_lbl = tk.Label(res, text="Set parameters and click\n'Get Recommendation'",
                                  font=("Segoe UI", 9), bg=CARD, fg=TEXT_S)
        self.conf_lbl.pack(pady=(2, 4))

        # Confidence bar bg
        bar_bg = tk.Frame(res, bg=LIGHT_G, height=8)
        bar_bg.pack(fill="x", padx=16, pady=(0, 6))
        bar_bg.pack_propagate(False)
        self.conf_bar = tk.Frame(bar_bg, bg=GREEN, height=8)
        self.conf_bar.place(relwidth=0, height=8)

        # Alternatives
        self.alt_lbl = tk.Label(res, text="", font=("Segoe UI", 9),
                                 bg=CARD, fg=TEXT_S)
        self.alt_lbl.pack(pady=(0, 8))

        # Charts
        chart_card = tk.Frame(right, bg=CARD,
                               highlightbackground=BORDER, highlightthickness=1)
        chart_card.pack(fill="both", expand=True, pady=(8, 0))

        tk.Label(chart_card, text="Your Values vs Ideal  &  Top 5 Predictions",
                 font=("Segoe UI", 10, "bold"), bg=CARD, fg=TEXT).pack(anchor="w", padx=12, pady=(8, 4))

        self.chart_parent = chart_card
        self.chart_canvas = None
        self._draw_default_charts(chart_card)

    def _draw_default_charts(self, parent):
        """Draw placeholder charts before first prediction."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.8))
        fig.patch.set_facecolor(CARD)
        for ax in (ax1, ax2):
            ax.set_facecolor("#F6FAF6")
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            ax.set_xticks([])
            ax.set_yticks([])

        ax1.text(0.5, 0.5, "Bar comparison appears\nafter prediction",
                 ha="center", va="center", fontsize=8, color=TEXT_S,
                 transform=ax1.transAxes)
        ax1.set_title("Your Values vs Ideal", fontsize=8, color=TEXT, pad=4)

        ax2.text(0.5, 0.5, "Confidence chart appears\nafter prediction",
                 ha="center", va="center", fontsize=8, color=TEXT_S,
                 transform=ax2.transAxes)
        ax2.set_title("Top 5 Crop Predictions", fontsize=8, color=TEXT, pad=4)

        plt.tight_layout(pad=1.2)
        self._render_canvas(fig, parent)

    def _draw_prediction_charts(self, crop, user_vals, top_crops, top_probs):
        """Draw radar + confidence charts after prediction."""
        # Clear old canvas
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.8))
        fig.patch.set_facecolor(CARD)
        for ax in (ax1, ax2):
            ax.set_facecolor("#F6FAF6")
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)

        # ── Chart 1: Grouped Bar — Your Values vs Ideal ──
        ideal      = self.df[self.df["label"] == crop][self.features].mean().values
        short_lbls = ["N", "P", "K", "Temp", "Hum", "pH", "Rain"]
        x          = np.arange(len(short_lbls))
        w          = 0.35

        ax1.bar(x - w/2, user_vals, width=w, color="#F57F17",
                label="Your values", zorder=3)
        ax1.bar(x + w/2, ideal,     width=w, color=GREEN,
                label=f"Ideal ({crop.title()})", zorder=3)

        ax1.set_xticks(x)
        ax1.set_xticklabels(short_lbls, fontsize=7)
        ax1.set_title(f"Your Values vs Ideal ({crop.title()})",
                      fontsize=7.5, color=TEXT, pad=4)
        ax1.set_ylabel("Value", fontsize=7)
        ax1.tick_params(labelsize=7)
        ax1.legend(fontsize=6.5, loc="upper right")
        ax1.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax1.set_axisbelow(True)

        # ── Chart 2: Confidence bars ─────────────
        colors = [GREEN if i == 0 else GREEN_L if i == 1 else "#A5D6A7"
                  for i in range(len(top_crops))]
        bars = ax2.barh(top_crops[::-1], top_probs[::-1], color=colors[::-1], height=0.5)
        ax2.set_xlim(0, 1)
        ax2.set_title("Top 5 Crop Predictions", fontsize=8, color=TEXT, pad=4)
        ax2.set_xlabel("Confidence", fontsize=7)
        ax2.tick_params(labelsize=7)
        for bar, prob in zip(bars, top_probs[::-1]):
            ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{prob*100:.1f}%", va="center", fontsize=6.5, color=TEXT)

        plt.tight_layout(pad=1.2)
        self._render_canvas(fig, self.chart_parent)

    def _render_canvas(self, fig, parent):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.chart_canvas = canvas

    # ── Predict ───────────────────────────────
    def _predict(self):
        user_vals = np.array([self.sliders[k].get() for k in self.features])
        X         = self.scaler.transform([user_vals])
        proba     = self.model.predict_proba(X)[0]
        classes   = self.model.classes_

        top_idx   = np.argsort(proba)[::-1]
        crop      = classes[top_idx[0]]
        conf      = proba[top_idx[0]]

        # Update result card
        self.emoji_lbl.config(text=self.CROP_EMOJI.get(crop.lower(), "🌱"))
        self.crop_lbl.config(text=crop.title())
        self.conf_lbl.config(
            text=f"Confidence: {conf*100:.1f}%",
            fg=GREEN if conf > 0.7 else "#F57F17"
        )
        self.conf_bar.place(relwidth=conf, height=8)

        # Alternatives
        alts = [classes[i] for i in top_idx[1:4]]
        self.alt_lbl.config(
            text="Alternatives: " + "  |  ".join(
                f"{self.CROP_EMOJI.get(a.lower(),'🌱')} {a.title()}" for a in alts
            )
        )

        # Update charts
        top5_crops = [classes[i] for i in top_idx[:5]]
        top5_probs = [proba[i]   for i in top_idx[:5]]
        self._draw_prediction_charts(crop, user_vals, top5_crops, top5_probs)


# ── Run ───────────────────────────────────────
if __name__ == "__main__":
    try:
        model, scaler, df, features, accuracy = load_and_train()
        app = CropApp(model, scaler, df, features, accuracy)
        app.mainloop()
    except FileNotFoundError as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Dataset Not Found", str(e))
        root.destroy()
