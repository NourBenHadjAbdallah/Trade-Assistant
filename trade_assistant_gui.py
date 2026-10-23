#!/usr/bin/env python3
"""
Trade Assistant Pro (GUI): Volume Profile + Anchored VWAP + Order Flow + Crypto Logos + Modern Chat

Requirements:
    pip install customtkinter pandas numpy requests matplotlib pillow
"""
import contextlib
import importlib
import io
import os
import queue
import subprocess
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def ensure_deps():
    needed = {
        "numpy": "numpy",
        "pandas": "pandas",
        "requests": "requests",
        "matplotlib": "matplotlib",
        "customtkinter": "customtkinter",
        "PIL": "pillow"
    }
    missing = []
    for mod, pkg in needed.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        except Exception as e:
            print(f"Failed to install packages: {e}")
            sys.exit(1)

ensure_deps()

import customtkinter as ctk
import numpy as np
import pandas as pd
import requests
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

try:
    import trade_assistant as ta
except ImportError:
    print("Error: trade_assistant.py must be in the same folder as this file.")
    sys.exit(1)

try:
    import trade_chat as tc
    COACH_ERR = None
except Exception as _e:
    tc = None
    COACH_ERR = str(_e)

# Enable dark mode
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Palette
BG = "#0B0E14"
PANEL = "#151922"
PANEL_BORDER = "#232834"
FG = "#E1E6F0"
MUTED = "#788296"
GREEN = "#26A69A"
RED = "#EF5350"
BLUE = "#2962FF"
AMBER = "#FFB300"
USER_BUBBLE_BG = "#1E4BD8"
COACH_BUBBLE_BG = "#1E2330"

LINE_COLORS = ["#FFB300", "#C77DFF", "#4DD0E1", "#FF8A65", "#AED581", "#F06292"]
FALLBACK_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
                  "AVAXUSDT", "LINKUSDT", "TRXUSDT", "TONUSDT", "DOTUSDT", "LTCUSDT", "SUIUSDT"]

INTERVALS = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1M"]
KLINE_URLS = ["https://api.binance.com/api/v3/klines", "https://data-api.binance.vision/api/v3/klines"]
TICKER_URLS = ["https://api.binance.com/api/v3/ticker/24hr", "https://data-api.binance.vision/api/v3/ticker/24hr"]

ICON_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".icon_cache")
os.makedirs(ICON_CACHE_DIR, exist_ok=True)


def load_coins():
    for url in TICKER_URLS:
        try:
            data = requests.get(url, timeout=10).json()
            rows = [(d["symbol"], float(d["quoteVolume"])) for d in data if d["symbol"].endswith("USDT")]
            rows.sort(key=lambda r: -r[1])
            if rows:
                return [s for s, _ in rows[:250]]
        except Exception:
            continue
    return FALLBACK_COINS


def get_coin_icon(symbol):
    base = symbol.upper().replace("USDT", "").replace("USDC", "").replace("BUSD", "").lower()
    local_path = os.path.join(ICON_CACHE_DIR, f"{base}.png")

    if not os.path.exists(local_path):
        urls = [
            f"https://cryptoicons.org/api/icon/{base}/64",
            f"https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/{base}.png"
        ]
        for url in urls:
            try:
                r = requests.get(url, timeout=4)
                if r.status_code == 200 and len(r.content) > 200:
                    with open(local_path, "wb") as f:
                        f.write(r.content)
                    break
            except Exception:
                continue

    if os.path.exists(local_path):
        try:
            pil_img = Image.open(local_path).convert("RGBA")
            return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(24, 24))
        except Exception:
            pass
    return None


def normalize_symbol(s):
    s = s.upper().replace("/", "").replace("-", "").replace(" ", "")
    if not s:
        raise ValueError("Type or choose a coin first.")
    if not s.endswith(("USDT", "USDC", "FDUSD", "BUSD", "BTC", "ETH", "BNB")):
        s += "USDT"
    return s


def get_klines(symbol, interval, limit):
    last = None
    for url in KLINE_URLS:
        try:
            ta.BINANCE_URL = url
            return ta.fetch_binance(symbol, interval, limit)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                raise ValueError(f"Binance does not list '{symbol}'. Try e.g. BTC, ETH, SOL.")
            last = e
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not reach Binance. Check your internet connection.\n({last})")


# Chart Styling
def style_ax(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(PANEL_BORDER)
        sp.set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color="#1E2330", lw=0.6, ls=":")


def draw_chart(fig, df, vp, avwaps, of_df, symbol, interval, price):
    fig.clear()
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1], width_ratios=[7, 1.2], hspace=0.06, wspace=0.02,
                          left=0.06, right=0.96, top=0.94, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0])
    axp = fig.add_subplot(gs[0, 1], sharey=ax)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    for a in (ax, axp, ax2):
        style_ax(a)

    n = len(df)
    x = np.arange(n)
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    col = np.where(c >= o, GREEN, RED)
    ax.vlines(x, l, h, colors=col, lw=0.8)
    body = np.maximum(np.abs(c - o), (h - l) * 0.03)
    ax.bar(x, body, bottom=np.minimum(o, c), color=col, width=0.7)

    for i, (name, (idx, s)) in enumerate(avwaps.items()):
        colr = LINE_COLORS[i % len(LINE_COLORS)]
        ax.plot(x, s.values, color=colr, lw=1.3, label=name.replace("AVWAP ", ""))
        ax.scatter([idx], [s.iloc[idx]], color=colr, s=18, zorder=5)

    for p in vp["hvn"]:
        ax.axhline(p, color=GREEN, ls=":", lw=0.7, alpha=0.5)
    for p in vp["lvn"]:
        ax.axhline(p, color=AMBER, ls=":", lw=0.7, alpha=0.5)
    for name, y, colr in [("POC", vp["poc"], RED), ("VAH", vp["vah"], "#9AA3B5"), ("VAL", vp["val"], "#9AA3B5")]:
        ax.axhline(y, color=colr, ls="--", lw=1)
        axp.axhline(y, color=colr, ls="--", lw=1)
        axp.text(1.02, y, name, transform=axp.get_yaxis_transform(), color=colr, fontsize=8, va="center")
    ax.axhline(price, color="#FFFFFF", lw=0.8, ls="-", alpha=0.6)

    in_va = (vp["centers"] >= vp["val"]) & (vp["centers"] <= vp["vah"])
    axp.barh(vp["centers"], vp["vol"], height=vp["centers"][1] - vp["centers"][0],
             color=np.where(in_va, BLUE, "#2A3142"))
    axp.tick_params(labelleft=False, labelbottom=False)
    axp.set_title("Volume Profile", color=MUTED, fontsize=8, pad=4)
    axp.grid(False)

    ax.margins(x=0.01)
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left", fontsize=7.5, facecolor=PANEL, edgecolor=PANEL_BORDER, labelcolor=FG, framealpha=0.9)
    ax.set_title(f"{symbol}  •  {interval}   Last: {price:.6g}", color=FG, fontsize=11, loc="left", fontweight="bold")

    d = of_df["delta"].values
    ax2.bar(x, d, color=np.where(d >= 0, GREEN, RED), width=0.7)
    ax2.set_ylabel("Delta", color=MUTED, fontsize=8)
    ax3 = ax2.twinx()
    ax3.plot(x, of_df["cvd"].values, color=BLUE, lw=1.2)
    ax3.tick_params(colors=MUTED, labelsize=8)
    for sp in ax3.spines.values():
        sp.set_color(PANEL_BORDER)
    ax3.set_ylabel("CVD", color=BLUE, fontsize=8)

    times = df["time"].reset_index(drop=True)
    fmt = "%Y-%m-%d" if interval in ("1d", "1w", "1M") else "%m-%d %H:%M"
    ax2.xaxis.set_major_locator(MaxNLocator(8))
    ax2.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: times.iloc[int(round(v))].strftime(fmt) if 0 <= int(round(v)) < n else ""))
    ax2.set_xlim(-1, n + 1)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Trade Assistant Pro")
        self.geometry("1480x880")
        self.configure(fg_color=BG)

        self.busy = False
        self.job = None
        self.coins = FALLBACK_COINS[:]
        self.q = queue.Queue()
        self.plan = None
        self.plan_key = None
        self.chat_hist = []
        self.coach_busy = False
        self.active_bot_label = None

        self._build_top_bar()
        self._build_main_layout()

        threading.Thread(target=self._load_coins, daemon=True).start()
        threading.Thread(target=self._load_models, daemon=True).start()
        self._poll()
        self.run()

    def _poll(self):
        try:
            while True:
                fn, args = self.q.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _build_top_bar(self):
        top = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=10, border_width=1, border_color=PANEL_BORDER)
        top.pack(fill="x", padx=12, pady=(10, 6))

        self.logo_label = ctk.CTkLabel(top, text="", width=24)
        self.logo_label.pack(side="left", padx=(12, 4), pady=8)

        ctk.CTkLabel(top, text="Coin:", font=("Segoe UI", 12, "bold")).pack(side="left", padx=(0, 4))
        self.symbol_var = ctk.StringVar(value="BTCUSDT")
        self.symbol_cb = ctk.CTkOptionMenu(top, variable=self.symbol_var, values=self.coins, width=120, command=lambda _: self._update_icon())
        self.symbol_cb.pack(side="left", padx=(0, 12))

        ctk.CTkLabel(top, text="TF:", font=("Segoe UI", 12, "bold")).pack(side="left", padx=(0, 4))
        self.interval_var = ctk.StringVar(value="15m")
        ctk.CTkOptionMenu(top, variable=self.interval_var, values=INTERVALS, width=75).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(top, text="Limit:", font=("Segoe UI", 12, "bold")).pack(side="left", padx=(0, 4))
        self.limit_var = ctk.StringVar(value="500")
        ctk.CTkOptionMenu(top, variable=self.limit_var, values=["100", "200", "300", "500", "1000"], width=85).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(top, text="Anchor:", font=("Segoe UI", 12, "bold")).pack(side="left", padx=(0, 4))
        self.anchor_entry = ctk.CTkEntry(top, placeholder_text="YYYY-MM-DD HH:MM", width=140)
        self.anchor_entry.pack(side="left", padx=(0, 12))

        self.btn_analyze = ctk.CTkButton(top, text="Analyze", font=("Segoe UI", 12, "bold"), fg_color=BLUE, hover_color="#1E4BD8", command=self.run)
        self.btn_analyze.pack(side="left", padx=(0, 12))

        self.auto_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(top, text="Auto-refresh (30s)", variable=self.auto_var, command=self._toggle_auto).pack(side="left", padx=(0, 12))

        self._update_icon()

    def _update_icon(self):
        def _fetch():
            icon = get_coin_icon(self.symbol_var.get())
            self.q.put((lambda: self.logo_label.configure(image=icon, text="" if icon else "🪙"), ()))
        threading.Thread(target=_fetch, daemon=True).start()

    def _build_main_layout(self):
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        chart_frame = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=10, border_width=1, border_color=PANEL_BORDER)
        chart_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        right_panel = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=10, border_width=1, border_color=PANEL_BORDER, width=460)
        right_panel.pack(side="right", fill="y", padx=(6, 0))
        right_panel.pack_propagate(False)

        self.tabview = ctk.CTkTabview(right_panel, fg_color=PANEL)
        self.tabview.pack(fill="both", expand=True, padx=6, pady=6)
        
        tab_a = self.tabview.add("Analysis")
        tab_c = self.tabview.add("Coach (Ollama)")

        self.verdict_label = ctk.CTkLabel(tab_a, text="Ready", font=("Segoe UI", 18, "bold"), anchor="w")
        self.verdict_label.pack(fill="x", padx=12, pady=(8, 2))

        self.sub_label = ctk.CTkLabel(tab_a, text="Run analysis to inspect market regime.", font=("Segoe UI", 11), text_color=MUTED, anchor="w")
        self.sub_label.pack(fill="x", padx=12, pady=(0, 8))

        self.text_analysis = ctk.CTkTextbox(tab_a, font=("Consolas", 11), fg_color=BG, border_width=1, border_color=PANEL_BORDER)
        self.text_analysis.pack(fill="both", expand=True, padx=8, pady=8)

        self._build_coach_tab(tab_c)

        self.status_var = ctk.StringVar(value="System Ready")
        status_bar = ctk.CTkLabel(self, textvariable=self.status_var, font=("Segoe UI", 10), text_color=MUTED, anchor="w")
        status_bar.pack(fill="x", padx=16, pady=(0, 4))

    def _build_coach_tab(self, parent):
        if tc is None:
            ctk.CTkLabel(parent, text=f"Coach unavailable.\n{COACH_ERR}", text_color=RED, justify="left").pack(fill="both", expand=True, padx=12, pady=12)
            return

        controls = ctk.CTkFrame(parent, fg_color="transparent")
        controls.pack(fill="x", padx=4, pady=4)

        ctk.CTkLabel(controls, text="Model:", font=("Segoe UI", 11)).grid(row=0, column=0, sticky="w")
        self.model_var = ctk.StringVar(value=tc.DEFAULT_MODEL)
        self.model_cb = ctk.CTkOptionMenu(controls, variable=self.model_var, values=[tc.DEFAULT_MODEL], width=130)
        self.model_cb.grid(row=0, column=1, padx=4, pady=2)

        ctk.CTkLabel(controls, text="Style:", font=("Segoe UI", 11)).grid(row=0, column=2, sticky="w")
        self.style_var = ctk.StringVar(value="Intraday")
        ctk.CTkOptionMenu(controls, variable=self.style_var, values=list(tc.ai.STYLES), width=95).grid(row=0, column=3, padx=4, pady=2)

        action_bar = ctk.CTkFrame(parent, fg_color="transparent")
        action_bar.pack(fill="x", padx=4, pady=4)
        
        ctk.CTkButton(action_bar, text="Explain Setup", font=("Segoe UI", 11, "bold"), fg_color=BLUE, command=lambda: self._coach_send(tc.BRIEFING, "Explain this setup.")).pack(side="left", padx=(0, 4))
        ctk.CTkButton(action_bar, text="Clear Chat", font=("Segoe UI", 11), fg_color="#333A48", command=self._coach_clear).pack(side="left")

        # Scrollable Chat Container for Speech Bubbles
        self.chat_container = ctk.CTkScrollableFrame(parent, fg_color=BG, border_width=1, border_color=PANEL_BORDER, corner_radius=8)
        self.chat_container.pack(fill="both", expand=True, padx=4, pady=6)

        input_bar = ctk.CTkFrame(parent, fg_color="transparent")
        input_bar.pack(fill="x", padx=4, pady=4)

        self.coach_input = ctk.CTkEntry(input_bar, placeholder_text="Ask about this setup...")
        self.coach_input.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.coach_input.bind("<Return>", lambda e: self._coach_send())

        self.send_btn = ctk.CTkButton(input_bar, text="Send", width=70, fg_color=BLUE, command=self._coach_send)
        self.send_btn.pack(side="right")

    def _add_chat_bubble(self, text, sender="user"):
        """Renders speech bubble containers inside the chat view."""
        row_frame = ctk.CTkFrame(self.chat_container, fg_color="transparent")
        row_frame.pack(fill="x", pady=4, padx=4)

        if sender == "user":
            bubble = ctk.CTkFrame(row_frame, fg_color=USER_BUBBLE_BG, corner_radius=12)
            bubble.pack(side="right", anchor="e", padx=(40, 0))
            label = ctk.CTkLabel(bubble, text=text, font=("Segoe UI", 11), text_color="#FFFFFF", wraplength=280, justify="left")
            label.pack(padx=10, pady=6)
        else:
            bubble = ctk.CTkFrame(row_frame, fg_color=COACH_BUBBLE_BG, border_width=1, border_color=PANEL_BORDER, corner_radius=12)
            bubble.pack(side="left", anchor="w", padx=(0, 40))
            
            sender_title = ctk.CTkLabel(bubble, text=f"AI Coach ({self.model_var.get()})", font=("Segoe UI", 9, "bold"), text_color=AMBER)
            sender_title.pack(anchor="w", padx=10, pady=(6, 0))
            
            label = ctk.CTkLabel(bubble, text=text, font=("Segoe UI", 11), text_color=FG, wraplength=280, justify="left")
            label.pack(padx=10, pady=(2, 6))
            self.active_bot_label = label

        self.chat_container._parent_canvas.yview_moveto(1.0)

    def _update_active_bot_bubble(self, chunk):
        if self.active_bot_label:
            cur_text = self.active_bot_label.cget("text")
            self.active_bot_label.configure(text=cur_text + chunk)
            self.chat_container._parent_canvas.yview_moveto(1.0)

    def _load_coins(self):
        coins = load_coins()
        self.coins = coins
        self.q.put((lambda: self.symbol_cb.configure(values=coins), ()))

    def _load_models(self):
        if tc is None:
            return
        try:
            models = tc.list_models()
        except Exception:
            return
        if models:
            pick = tc.DEFAULT_MODEL if tc.DEFAULT_MODEL in models else models[0]
            self.q.put((lambda: self.model_cb.configure(values=models), ()))
            self.model_var.set(pick)

    def _toggle_auto(self):
        if self.job:
            self.after_cancel(self.job)
            self.job = None
        if self.auto_var.get():
            self.run()

    def run(self):
        if self.busy:
            return
        try:
            sym = normalize_symbol(self.symbol_var.get())
            limit = int(self.limit_var.get())
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            return
        self.symbol_var.set(sym)
        self._update_icon()
        self.busy = True
        self.btn_analyze.configure(state="disabled")
        self.status_var.set(f"Fetching data for {sym}...")
        threading.Thread(target=self._work, args=(sym, self.interval_var.get(), limit, self.anchor_entry.get().strip()), daemon=True).start()

    def _work(self, sym, interval, limit, anchor):
        try:
            df = get_klines(sym, interval, limit)
            vp = ta.volume_profile(df, bins=60)
            avwaps = ta.build_avwaps(df, anchor or None)
            of_df, of = ta.order_flow(df)
            res = ta.analyze(df, vp, avwaps, of, of_df)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ta.report(df, vp, avwaps, of, res, sym)
            self.q.put((self._show, (sym, interval, df, vp, avwaps, of_df, res, buf.getvalue())))
        except Exception as e:
            self.q.put((self._error, (e,)))

    def _show(self, sym, interval, df, vp, avwaps, of_df, res, report_text):
        draw_chart(self.fig, df, vp, avwaps, of_df, sym, interval, res["price"])
        self.canvas.draw_idle()

        lab = ta.label(res["score"])
        colr = GREEN if "BULL" in lab.upper() else RED if "BEAR" in lab.upper() else MUTED
        self.verdict_label.configure(text=lab, text_color=colr)
        self.sub_label.configure(text=f"{sym} • {interval}  |  Score: {res['score']:+.1f}  |  ATR: {res['atr']:.6g}")

        self.text_analysis.delete("1.0", "end")
        self.text_analysis.insert("1.0", report_text)
        self.status_var.set(f"Updated: {pd.Timestamp.now():%H:%M:%S} | Candles: {len(df)}")
        self._finish()

    def _finish(self):
        self.busy = False
        self.btn_analyze.configure(state="normal")
        if self.auto_var.get():
            self.job = self.after(30000, self.run)

    def _error(self, e):
        self.status_var.set(f"Analysis error: {e}")
        self._finish()

    def _coach_send(self, question=None, shown=None):
        if tc is None or self.coach_busy:
            return
        q = question or self.coach_input.get().strip()
        if not q:
            return
        sym = normalize_symbol(self.symbol_var.get())
        if question is None:
            self.coach_input.delete(0, "end")
        
        self.coach_busy = True
        self._add_chat_bubble(shown or q, sender="user")
        self._add_chat_bubble("", sender="bot")

        threading.Thread(target=self._coach_work, args=(q, sym, self.style_var.get(), self.model_var.get().strip()), daemon=True).start()

    def _coach_work(self, q, sym, style, model):
        try:
            plan = tc.ai.build_plan(lambda s, tf: get_klines(s, tf, 500), sym, style, 1000.0, 1.0)
            ans = tc.ask_ollama(q, plan, self.chat_hist, tc.journal_text(), model=model,
                                on_token=lambda t: self.q.put((self._update_active_bot_bubble, (t,))))
            self.chat_hist += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
        except Exception as e:
            self.q.put((self._update_active_bot_bubble, (f"\n[Error: {e}]",)))
        finally:
            self.coach_busy = False

    def _coach_clear(self):
        self.chat_hist = []
        for widget in self.chat_container.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()