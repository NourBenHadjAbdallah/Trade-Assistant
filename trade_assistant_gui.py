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
        "PIL": "pillow",
        "dotenv": "python-dotenv",
        "anthropic": "anthropic",
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

import config  # noqa: F401  (loads .env)
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
    import trade_ai as ai
    AI_ERR = None
except Exception as _e:
    ai = None
    AI_ERR = str(_e)

try:
    import signal_log as sl
    SL_ERR = None
except Exception as _e:
    sl = None
    SL_ERR = str(_e)

try:
    import trade_chat as tc
    COACH_ERR = None
except Exception as _e:
    tc = None
    COACH_ERR = str(_e)

# Backtesting (trade_backtest.py must be in the same folder).
try:
    import trade_backtest as tb
    BT_ERR = None
except Exception as _e:
    tb = None
    BT_ERR = str(_e)

try:
    import trade_replay as trp
except Exception as _e:
    trp = None

# News alerts (news_alerts.py must be in the same folder). The GUI keeps its own state file.
os.environ.setdefault("STATE_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_news_state.json"))
try:
    import news_alerts as na
    NEWS_ERR = None
except Exception as _e:
    na = None
    NEWS_ERR = str(_e)
NEWS_ID = "gui"          # the GUI is one "user" in the news state
NEWS_EVERY_MS = 5 * 60 * 1000

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
CARD = "#181D28"
TONE = {"long": GREEN, "short": RED, "wait": AMBER, "avoid": MUTED}
REGIME_COLOR = {"trend_up": GREEN, "trend_down": RED, "range": "#4DA3FF", "transition": AMBER}
SCAN_EVERY_MS = 15 * 60 * 1000

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


# UI building blocks
def pill(parent, text, color, size=11):
    return ctk.CTkLabel(parent, text=f"  {text}  ", font=("Segoe UI", size, "bold"), text_color=BG,
                        fg_color=color, corner_radius=6, height=size * 2 + 4)


def card(parent, title=None, border=PANEL_BORDER):
    f = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=10, border_width=1, border_color=border)
    f.pack(fill="x", padx=4, pady=5)
    if title:
        ctk.CTkLabel(f, text=title.upper(), font=("Segoe UI", 10, "bold"), text_color=MUTED,
                     anchor="w").pack(fill="x", padx=12, pady=(8, 2))
    return f


def text(parent, s, color=FG, size=12, bold=False, wrap=430, **pack):
    lbl = ctk.CTkLabel(parent, text=s, font=("Segoe UI", size, "bold" if bold else "normal"), text_color=color,
                       anchor="w", justify="left", wraplength=wrap)
    lbl.pack(fill="x", padx=pack.pop("padx", 12), pady=pack.pop("pady", 1), **pack)
    return lbl


def bullet_list(parent, items, mark, color, limit=5):
    for it in items[:limit]:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=1)
        ctk.CTkLabel(row, text=mark, font=("Segoe UI", 12, "bold"), text_color=color, width=18, anchor="n").pack(side="left", anchor="n")
        ctk.CTkLabel(row, text=it, font=("Segoe UI", 12), text_color=FG, anchor="w", justify="left",
                     wraplength=400).pack(side="left", fill="x", expand=True)


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
        self.news_busy = False
        self.bt_busy = False
        self.bt_text = ""

        self._build_top_bar()
        self._build_main_layout()

        threading.Thread(target=self._load_coins, daemon=True).start()
        threading.Thread(target=self._load_models, daemon=True).start()
        self._poll()
        self.after(5000, self._news_tick)
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

        ctk.CTkLabel(top, text="Style:", font=("Segoe UI", 12, "bold")).pack(side="left", padx=(0, 4))
        self.style_var = ctk.StringVar(value="Intraday")
        ctk.CTkOptionMenu(top, variable=self.style_var, values=list(ai.STYLES) if ai else ["Intraday"],
                          width=100).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(top, text="Anchor:", font=("Segoe UI", 12, "bold")).pack(side="left", padx=(0, 4))
        self.anchor_entry = ctk.CTkEntry(top, placeholder_text="YYYY-MM-DD HH:MM", width=140)
        self.anchor_entry.pack(side="left", padx=(0, 12))

        self.btn_analyze = ctk.CTkButton(top, text="Analyze", font=("Segoe UI", 12, "bold"), fg_color=BLUE, hover_color="#1E4BD8", command=self.run)
        self.btn_analyze.pack(side="left", padx=(0, 12))

        self.auto_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(top, text="Auto-refresh (30s)", variable=self.auto_var, command=self._toggle_auto).pack(side="left", padx=(0, 12))

        self._update_icon()

    def _update_icon(self):
        sym = self.symbol_var.get()          # read Tk variables on the UI thread only

        def _fetch():
            icon = get_coin_icon(sym)
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

        right_panel = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=10, border_width=1, border_color=PANEL_BORDER, width=520)
        right_panel.pack(side="right", fill="y", padx=(6, 0))
        right_panel.pack_propagate(False)

        self.tabview = ctk.CTkTabview(right_panel, fg_color=PANEL)
        self.tabview.pack(fill="both", expand=True, padx=6, pady=6)

        tab_s = self.tabview.add("Signals")
        tab_t = self.tabview.add("Track record")
        tab_a = self.tabview.add("Analysis")
        tab_c = self.tabview.add("Coach")
        tab_n = self.tabview.add("News")
        tab_b = self.tabview.add("Backtest")

        self.an_body = ctk.CTkScrollableFrame(tab_a, fg_color=BG, corner_radius=8)
        self.an_body.pack(fill="both", expand=True, padx=4, pady=(4, 2))
        text(self.an_body, "Pick a coin and press Analyze, or open one from the Signals tab.", MUTED, pady=12)
        self.an_details_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(tab_a, text="Show full technical report", variable=self.an_details_var, font=("Segoe UI", 10),
                      command=self._an_toggle_details).pack(anchor="w", padx=10, pady=(2, 4))
        self.text_analysis = ctk.CTkTextbox(tab_a, height=260, font=("Consolas", 10), fg_color=BG, border_width=1,
                                            border_color=PANEL_BORDER, wrap="none")      # shown only when the switch is on

        self._build_signals_tab(tab_s)
        self._build_track_tab(tab_t)
        self._build_coach_tab(tab_c)
        self._build_news_tab(tab_n)
        self._build_backtest_tab(tab_b)

        self.status_var = ctk.StringVar(value="System Ready")
        status_bar = ctk.CTkLabel(self, textvariable=self.status_var, font=("Segoe UI", 10), text_color=MUTED, anchor="w")
        status_bar.pack(fill="x", padx=16, pady=(0, 4))

    def _build_coach_tab(self, parent):
        if tc is None:
            ctk.CTkLabel(parent, text=f"Coach unavailable.\n{COACH_ERR}", text_color=RED, justify="left").pack(fill="both", expand=True, padx=12, pady=12)
            return

        controls = ctk.CTkFrame(parent, fg_color="transparent")
        controls.pack(fill="x", padx=4, pady=4)

        self.backend_var = ctk.StringVar(value="Ollama")
        ctk.CTkSegmentedButton(controls, values=["Ollama", "Claude"], variable=self.backend_var,
                               command=self._coach_backend).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(controls, text="Model:", font=("Segoe UI", 11)).pack(side="left", padx=(0, 4))
        self.ollama_models = [tc.DEFAULT_MODEL]
        self.model_var = ctk.StringVar(value=tc.DEFAULT_MODEL)
        self.model_cb = ctk.CTkOptionMenu(controls, variable=self.model_var, values=self.ollama_models, width=160)
        self.model_cb.pack(side="left")
        self.coach_status = text(parent, "Local and free. Uses the coin and style from the top bar.", MUTED, 10,
                                 wrap=470, padx=8, pady=(0, 2))

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
            
            sender_title = ctk.CTkLabel(bubble, text=f"AI Coach · {self.model_var.get()}", font=("Segoe UI", 9, "bold"), text_color=AMBER)
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
        if na is not None:
            self.q.put((self._news_fill_list, ()))

    def _load_models(self):
        if tc is None:
            return
        try:
            models = tc.list_models()
        except Exception:
            return
        if models:
            self.q.put((self._models_loaded, (models,)))     # widgets are updated on the UI thread

    def _models_loaded(self, models):
        self.ollama_models = models
        if self.backend_var.get() == "Ollama":
            self.model_cb.configure(values=models)
            self.model_var.set(tc.DEFAULT_MODEL if tc.DEFAULT_MODEL in models else models[0])

    def _coach_backend(self, choice):
        """Ollama (default, local) or Claude (subscription via Claude Code, or API key)."""
        if choice == "Claude":
            models, pick = tc.cc.CLAUDE_MODELS, tc.cc.CLAUDE_MODEL
            ready = tc.cc.mode() is not None
            self.coach_status.configure(text=tc.cc.status_text(), text_color=MUTED if ready else AMBER)
        else:
            models = self.ollama_models
            pick = tc.DEFAULT_MODEL if tc.DEFAULT_MODEL in models else models[0]
            self.coach_status.configure(text="Local and free. Uses the coin and style from the top bar.", text_color=MUTED)
        self.model_cb.configure(values=models)
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
        threading.Thread(target=self._work, args=(sym, self.interval_var.get(), limit, self.anchor_entry.get().strip(),
                                                  self.style_var.get()), daemon=True).start()

    def _work(self, sym, interval, limit, anchor, style):
        try:
            df = get_klines(sym, interval, limit)
            vp = ta.volume_profile(df, bins=60)
            avwaps = ta.build_avwaps(df, anchor or None)
            of_df, of = ta.order_flow(df)
            res = ta.analyze(df, vp, avwaps, of, of_df)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ta.report(df, vp, avwaps, of, res, sym)
            plan = rec = plan_err = None
            if ai is None:
                plan_err = f"trade_ai.py unavailable: {AI_ERR}"
            else:
                try:
                    plan = ai.build_plan(lambda s, tf: get_klines(s, tf, 500), sym, style)
                    rec = ai.recommend(plan)
                    buf.write("\n\n" + ai.render_plan_text(plan))
                except Exception as e:
                    plan_err = f"Multi-timeframe plan failed: {e}"
            self.q.put((self._show, (sym, interval, df, vp, avwaps, of_df, res, buf.getvalue(), plan, rec, plan_err)))
        except Exception as e:
            self.q.put((self._error, (e,)))

    def _show(self, sym, interval, df, vp, avwaps, of_df, res, report_text, plan=None, rec=None, plan_err=None):
        draw_chart(self.fig, df, vp, avwaps, of_df, sym, interval, res["price"])
        self.canvas.draw_idle()
        self.plan = plan
        self._render_analysis(sym, interval, res, plan, rec, plan_err)
        self.text_analysis.delete("1.0", "end")
        self.text_analysis.insert("1.0", report_text)
        self.status_var.set(f"Updated: {pd.Timestamp.now():%H:%M:%S} | Candles: {len(df)}")
        self._finish()

    def _an_toggle_details(self):
        if self.an_details_var.get():
            self.text_analysis.pack(fill="x", padx=8, pady=(0, 8))
        else:
            self.text_analysis.pack_forget()

    # ---------------- Analysis cards ----------------
    def _render_analysis(self, sym, interval, res, plan, rec, plan_err):
        body = self.an_body
        for w in body.winfo_children():
            w.destroy()
        price = res["price"]
        if plan is None:
            lab = ta.label(res["score"])
            colr = GREEN if "BULL" in lab.upper() else RED if "BEAR" in lab.upper() else MUTED
            c = card(body, border=colr)
            text(c, f"{sym}  ·  {interval}  ·  {price:.6g}", MUTED, 11, pady=(10, 0))
            text(c, lab, colr, 18, True, pady=(2, 4))
            text(c, plan_err or "", AMBER, 11, pady=(0, 10))
            return

        color = TONE[rec["tone"]]
        hero = card(body, border=color)
        head = ctk.CTkFrame(hero, fg_color="transparent")
        head.pack(fill="x", padx=12, pady=(12, 4))
        pill(head, rec["action"], color, 15).pack(side="left")
        ctk.CTkLabel(head, text=sym, font=("Segoe UI", 16, "bold"), text_color=FG).pack(side="left", padx=10)
        ctk.CTkLabel(head, text=f"{price:.6g}", font=("Segoe UI", 14), text_color=MUTED).pack(side="right")
        text(hero, rec["headline"], FG, 13, True, pady=(4, 0))
        text(hero, f"{plan['style']} · {' > '.join(plan['tfs'][::-1])} · {plan['alignment']}", MUTED, 11)

        conf = ctk.CTkFrame(hero, fg_color="transparent")
        conf.pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(conf, text=f"Confidence {rec['score']}/100", font=("Segoe UI", 11, "bold"),
                     text_color=FG, width=130, anchor="w").pack(side="left")
        bar = ctk.CTkProgressBar(conf, height=10, progress_color=color, fg_color=PANEL_BORDER)
        bar.set(rec["score"] / 100)
        bar.pack(side="left", fill="x", expand=True, padx=(6, 0))

        step = ctk.CTkFrame(hero, fg_color=PANEL, corner_radius=8)
        step.pack(fill="x", padx=12, pady=(8, 4))
        text(step, "WHAT TO DO", MUTED, 10, True, pady=(8, 0))
        text(step, rec["next_step"], FG, 12, wrap=410, pady=(2, 8))
        if rec["max_risk_pct"]:
            text(hero, f"Risk at most {rec['max_risk_pct']:g}% of your account on this trade. "
                       "Backtests show only a thin edge, so keep size small.", AMBER, 11)
        text(hero, "Rule-based signal, not financial advice.", MUTED, 10, pady=(2, 10))

        s = plan["setup"]
        if s:
            c = card(body, "Trade plan")
            grid = ctk.CTkFrame(c, fg_color="transparent")
            grid.pack(fill="x", padx=12, pady=(2, 10))
            rows = [("Side", s["side"], GREEN if s["side"] == "LONG" else RED),
                    ("Entry zone", f"{s['zone_lo']:.6g} – {s['zone_hi']:.6g}", FG),
                    ("Stop loss", f"{s['stop']:.6g}", RED),
                    ("Target 1", f"{s['t1']:.6g}   ({s['rr1']:.1f}R)", GREEN),
                    ("Target 2", f"{s['t2']:.6g}   ({s['rr2']:.1f}R)", GREEN),
                    ("Status", s["state"], FG if s["in_zone"] else AMBER)]
            for i, (k, v, col) in enumerate(rows):
                ctk.CTkLabel(grid, text=k, font=("Segoe UI", 11), text_color=MUTED, anchor="w", width=90).grid(
                    row=i, column=0, sticky="nw", pady=2)
                ctk.CTkLabel(grid, text=v, font=("Segoe UI", 12, "bold"), text_color=col, anchor="w", justify="left",
                             wraplength=320).grid(row=i, column=1, sticky="w", pady=2)

        if rec["pro"] or rec["con"]:
            c = card(body, "Key insights")
            if rec["pro"]:
                text(c, "For the trade", GREEN, 11, True, pady=(4, 2))
                bullet_list(c, rec["pro"], "✓", GREEN)
            if rec["con"]:
                text(c, "Against the trade", RED, 11, True, pady=(8, 2))
                bullet_list(c, rec["con"], "✗", RED)
            ctk.CTkFrame(c, fg_color="transparent", height=6).pack()

        c = card(body, "Timeframes")
        row = ctk.CTkFrame(c, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(2, 10))
        for i, t in enumerate(plan["tf"]):
            row.grid_columnconfigure(i, weight=1, uniform="tf")
            box = ctk.CTkFrame(row, fg_color=PANEL, corner_radius=8)
            box.grid(row=0, column=i, sticky="nsew", padx=4)
            ctk.CTkLabel(box, text=f"{t['tf']} · {t['role']}", font=("Segoe UI", 11, "bold"), text_color=FG).pack(pady=(8, 4))
            pill(box, ai.REGIME_TEXT[t["regime"]][0], REGIME_COLOR[t["regime"]], 9).pack()
            ctk.CTkLabel(box, text=f"Δ {t['delta_bias']:+.0%}   RSI {t['rsi']:.0f}", font=("Segoe UI", 11),
                         text_color=MUTED).pack(pady=(4, 8))

        if s and s.get("confluence"):
            c = card(body, f"Confluence checklist  {s['confluence_met']}/{len(s['confluence'])}")
            for txt, ok in s["confluence"]:
                bullet_list(c, [txt], "✓" if ok else "✗", GREEN if ok else MUTED)
            ctk.CTkFrame(c, fg_color="transparent", height=6).pack()

        mid = plan["tf"][1]
        c = card(body, f"Key levels ({mid['tf']} volume profile)")
        for name, lv in (("VAH  value high", mid["vah"]), ("POC  most traded", mid["poc"]), ("VAL  value low", mid["val"])):
            r = ctk.CTkFrame(c, fg_color="transparent")
            r.pack(fill="x", padx=12, pady=1)
            ctk.CTkLabel(r, text=name, font=("Segoe UI", 11), text_color=MUTED, anchor="w").pack(side="left")
            ctk.CTkLabel(r, text=f"{lv:.6g}   ({(lv - price) / price:+.2%})", font=("Segoe UI", 12, "bold"),
                         text_color=FG).pack(side="right")
        ctk.CTkFrame(c, fg_color="transparent", height=8).pack()
        body._parent_canvas.yview_moveto(0)

    # ---------------- Signals scanner ----------------
    def _build_signals_tab(self, parent):
        if ai is None:
            ctk.CTkLabel(parent, text=f"Signals unavailable.\n{AI_ERR}", text_color=RED, justify="left").pack(
                fill="both", expand=True, padx=12, pady=12)
            return
        text(parent, "Your signals live here. The scanner checks the most traded coins with the multi-timeframe "
                     "engine and ranks them: BUY / SELL = ready to act, WAIT = setup forming. "
                     "Click Open for the full plan.", MUTED, 11, wrap=470, pady=(6, 4))
        r1 = ctk.CTkFrame(parent, fg_color="transparent")
        r1.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(r1, text="Coins:", font=("Segoe UI", 11)).pack(side="left", padx=(4, 4))
        self.sig_n = ctk.StringVar(value="20")
        ctk.CTkOptionMenu(r1, variable=self.sig_n, values=["10", "20", "40"], width=64).pack(side="left", padx=(0, 8))
        self.sig_filter = ctk.StringVar(value="Ready + forming")
        ctk.CTkSegmentedButton(r1, values=["Ready", "Ready + forming", "All"], variable=self.sig_filter,
                               command=lambda _v: self._sig_render()).pack(side="left", padx=(0, 8))
        self.sig_btn = ctk.CTkButton(r1, text="Scan", width=70, font=("Segoe UI", 11, "bold"), fg_color=BLUE,
                                     command=self._sig_scan)
        self.sig_btn.pack(side="right", padx=4)

        r2 = ctk.CTkFrame(parent, fg_color="transparent")
        r2.pack(fill="x", padx=4, pady=(2, 2))
        self.sig_auto = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(r2, text="Auto-scan every 15 min (beeps on a new STRONG signal)", variable=self.sig_auto,
                      font=("Segoe UI", 10)).pack(side="left", padx=4)
        self.sig_prog = ctk.CTkProgressBar(parent, height=6, progress_color=BLUE, fg_color=PANEL_BORDER)
        self.sig_prog.set(0)
        self.sig_prog.pack(fill="x", padx=10, pady=(4, 2))
        self.sig_info = text(parent, "", MUTED, 10, pady=(0, 2))
        self.sig_list = ctk.CTkScrollableFrame(parent, fg_color=BG, corner_radius=8)
        self.sig_list.pack(fill="both", expand=True, padx=4, pady=4)
        self.sig_rows, self.sig_busy, self.sig_strong = [], False, set()
        text(self.sig_list, "Scanning starts automatically in a moment...", MUTED, pady=12)
        self.after(3000, self._sig_tick)

    def _sig_tick(self):
        if self.sig_auto.get():
            self._sig_scan(auto=True)
        self.after(SCAN_EVERY_MS, self._sig_tick)

    def _sig_scan(self, auto=False):
        if ai is None or self.sig_busy:
            return
        coins = ai.scan_symbols(self.coins, int(self.sig_n.get()))
        style = self.style_var.get()
        self.sig_busy, self.sig_done_n, self.sig_total = True, 0, len(coins)
        self.sig_btn.configure(state="disabled", text="...")
        self.sig_prog.set(0)
        self.sig_info.configure(text=f"Scanning {len(coins)} coins ({style})...")

        def work():
            rows = ai.scan(lambda s, tf: get_klines(s, tf, 500), coins, style, 4,
                           on_result=lambda _r: self.q.put((self._sig_progress, ())))
            logged = 0
            if sl is not None:                       # track record: log new signals, grade the open ones
                try:
                    logged = sl.record(rows)
                    sl.update()
                except Exception as e:
                    self.q.put((self.status_var.set, (f"Track record not updated: {e}",)))
            self.q.put((self._sig_done, (rows, style, auto, logged)))
        threading.Thread(target=work, daemon=True).start()

    def _sig_progress(self):
        self.sig_done_n += 1
        self.sig_prog.set(self.sig_done_n / max(self.sig_total, 1))

    def _sig_done(self, rows, style, auto, logged=0):
        self.sig_busy = False
        self.sig_btn.configure(state="normal", text="Scan")
        self.sig_rows = rows
        ok = [r for r in rows if r["rec"]]
        strong = {r["symbol"] for r in ok if r["rec"]["grade"] == "strong"}
        ready = sum(1 for r in ok if r["rec"]["grade"] in ("strong", "actionable"))
        forming = sum(1 for r in ok if r["rec"]["grade"] == "forming")
        self.sig_info.configure(text=f"Last scan {pd.Timestamp.now():%H:%M} · {style} · {len(ok)} coins: "
                                     f"{ready} ready, {forming} forming"
                                     + (f", {len(rows) - len(ok)} failed" if len(ok) < len(rows) else "")
                                     + (f" · {logged} new logged to Track record" if logged else ""))
        new = strong - self.sig_strong
        if auto and new:
            self.bell()
            self.status_var.set("New STRONG signal: " + ", ".join(sorted(new)))
        self.sig_strong = strong
        self._sig_render()
        self._tr_refresh()

    def _sig_render(self):
        if not hasattr(self, "sig_list"):
            return
        for w in self.sig_list.winfo_children():
            w.destroy()
        allowed = {"Ready": ("strong", "actionable"), "Ready + forming": ("strong", "actionable", "forming"),
                   "All": tuple(ai.GRADE_RANK)}[self.sig_filter.get()]
        rows = [r for r in self.sig_rows if r["rec"] and r["rec"]["grade"] in allowed]
        if not rows:
            msg = ("No coins match this filter right now. That is normal: strong setups are rare. "
                   "Switch to 'All' to see every coin, or wait for the next scan.") if self.sig_rows else "No scan yet."
            text(self.sig_list, msg, MUTED, wrap=440, pady=12)
            return
        for r in rows:
            rec, plan = r["rec"], r["plan"]
            color = TONE[rec["tone"]]
            f = ctk.CTkFrame(self.sig_list, fg_color=CARD, corner_radius=10, border_width=1,
                             border_color=color if rec["grade"] in ("strong", "actionable") else PANEL_BORDER)
            f.pack(fill="x", padx=2, pady=4)
            top = ctk.CTkFrame(f, fg_color="transparent")
            top.pack(fill="x", padx=10, pady=(8, 2))
            pill(top, rec["action"], color, 10).pack(side="left")
            if rec["grade"] == "strong":
                pill(top, "STRONG", AMBER, 10).pack(side="left", padx=(4, 0))
            ctk.CTkLabel(top, text=r["symbol"].replace("USDT", ""), font=("Segoe UI", 14, "bold"),
                         text_color=FG).pack(side="left", padx=8)
            ctk.CTkButton(top, text="Open ›", width=64, height=26, fg_color="#2A3142", hover_color=BLUE,
                          command=lambda s=r["symbol"]: self._open_symbol(s)).pack(side="right")
            setup = plan["setup"]
            chk = f" · checklist {setup['confluence_met']}/{len(setup['confluence'])}" if setup and setup.get("confluence") else ""
            ctk.CTkLabel(top, text=f"{rec['score']}/100{chk}", font=("Segoe UI", 11), text_color=MUTED).pack(side="right", padx=8)
            text(f, rec["headline"], FG, 11, wrap=440, padx=10)
            why = rec["pro"][0] if rec["pro"] else (rec["con"][0] if rec["con"] else "")
            if why:
                text(f, ("✓ " if rec["pro"] else "✗ ") + why, GREEN if rec["pro"] else RED, 10, wrap=440, padx=10, pady=(0, 8))

    # ---------------- Track record ----------------
    def _build_track_tab(self, parent):
        if sl is None:
            ctk.CTkLabel(parent, text=f"Track record unavailable.\n{SL_ERR}", text_color=RED, justify="left").pack(
                fill="both", expand=True, padx=12, pady=12)
            return
        text(parent, "Every signal the scanner gives is logged here and graded against what price did next, "
                     "following the app's own plan (fees included). This is how you find out if the signals work.",
             MUTED, 11, wrap=470, pady=(6, 4))
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.pack(fill="x", padx=4, pady=2)
        self.tr_btn = ctk.CTkButton(bar, text="Update results", width=120, font=("Segoe UI", 11, "bold"),
                                    fg_color=BLUE, command=self._tr_update)
        self.tr_btn.pack(side="left", padx=4)
        self.tr_info = ctk.CTkLabel(bar, text="", font=("Segoe UI", 10), text_color=MUTED)
        self.tr_info.pack(side="left", padx=8)
        self.tr_body = ctk.CTkScrollableFrame(parent, fg_color=BG, corner_radius=8)
        self.tr_body.pack(fill="both", expand=True, padx=4, pady=4)
        self.tr_fig = Figure(figsize=(4.6, 1.8), dpi=100)
        self._tr_refresh()

    def _tr_update(self):
        self.tr_btn.configure(state="disabled", text="Updating...")

        def work():
            try:
                msg = f"{sl.update()} signal(s) changed status."
            except Exception as e:
                msg = f"Update failed: {e}"
            self.q.put((self._tr_updated, (msg,)))
        threading.Thread(target=work, daemon=True).start()

    def _tr_updated(self, msg):
        self.tr_btn.configure(state="normal", text="Update results")
        self.status_var.set("Track record: " + msg)
        self._tr_refresh()

    def _tr_refresh(self):
        if sl is None or not hasattr(self, "tr_body"):
            return
        try:
            df = sl.load()
        except Exception as e:
            self.tr_info.configure(text=f"Could not read the log: {e}")
            return
        self.tr_info.configure(text=f"Updated {pd.Timestamp.now():%H:%M}")
        self._tr_render(df, sl.summary(df))

    def _tr_table(self, title, rows):
        c = card(self.tr_body, title)
        g = ctk.CTkFrame(c, fg_color="transparent")
        g.pack(fill="x", padx=12, pady=(2, 10))
        for j, h in enumerate(("", "closed", "win rate", "avg R", "total R")):
            ctk.CTkLabel(g, text=h, font=("Segoe UI", 10), text_color=MUTED, anchor="w").grid(
                row=0, column=j, sticky="w", padx=(0, 14))
        for i, (name, b) in enumerate(rows, start=1):
            done = b["closed"] > 0
            vals = [name, str(b["closed"]),
                    f"{b['win_rate']:.0%}" if done else "–",
                    f"{b['avg_r']:+.2f}" if done else "–",
                    f"{b['total_r']:+.1f}" if done else "–"]
            for j, v in enumerate(vals):
                col = (GREEN if b["avg_r"] > 0 else RED) if done and j >= 3 else FG
                ctk.CTkLabel(g, text=v, font=("Segoe UI", 11, "bold" if j == 0 else "normal"), text_color=col,
                             anchor="w").grid(row=i, column=j, sticky="w", padx=(0, 14), pady=1)

    def _tr_render(self, df, s):
        body = self.tr_body
        for w in body.winfo_children():
            w.destroy()
        if df.empty:
            text(body, "No signals logged yet. They are added automatically after each scan in the Signals tab "
                       "(only BUY / SELL, STRONG and WAIT FOR ENTRY signals).", MUTED, wrap=440, pady=12)
            return
        a = s["all"]
        done = a["closed"] > 0
        tiles = ctk.CTkFrame(body, fg_color="transparent")
        tiles.pack(fill="x", padx=2, pady=(4, 2))
        stats = [("Signals", f"{a['signals']}", FG),
                 ("Closed trades", f"{a['closed']}", FG),
                 ("Win rate", f"{a['win_rate']:.0%}" if done else "–", FG),
                 ("Avg per trade", f"{a['avg_r']:+.2f}R" if done else "–",
                  (GREEN if a["avg_r"] > 0 else RED) if done else MUTED)]
        for i, (k, v, col) in enumerate(stats):
            tiles.grid_columnconfigure(i, weight=1, uniform="kpi")
            box = ctk.CTkFrame(tiles, fg_color=CARD, corner_radius=10, border_width=1, border_color=PANEL_BORDER)
            box.grid(row=0, column=i, sticky="nsew", padx=3)
            ctk.CTkLabel(box, text=v, font=("Segoe UI", 18, "bold"), text_color=col).pack(pady=(8, 0))
            ctk.CTkLabel(box, text=k, font=("Segoe UI", 10), text_color=MUTED).pack(pady=(0, 8))

        tone = AMBER if a["closed"] < 30 else GREEN if a["avg_r"] > 0 else RED
        c = card(body, border=tone)
        text(c, s["verdict"], tone, 12, True, pady=(10, 2))
        text(c, f"{a['open']} open · {a['pending']} waiting for entry · {a['expired']} expired"
                + (f" · {a['fill_rate']:.0%} of decided signals got filled" if a["fill_rate"] is not None else "")
                + f" · total {a['total_r']:+.1f}R", MUTED, 10, pady=(0, 10))

        if s["curve"]:
            c = card(body, "Cumulative result (R, after fees)")
            fig = self.tr_fig
            fig.clear()
            fig.patch.set_facecolor(CARD)
            ax = fig.add_subplot(111)
            style_ax(ax)
            ax.set_facecolor(CARD)
            curve = [0.0] + s["curve"]
            ax.plot(curve, color=GREEN if curve[-1] >= 0 else RED, lw=1.6)
            ax.axhline(0, color="#9AA3B5", lw=0.8)
            ax.set_xlabel("closed trade #", color=MUTED, fontsize=8)
            fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.25)
            FigureCanvasTkAgg(fig, master=c).get_tk_widget().pack(fill="x", padx=6, pady=(0, 8))

        self._tr_table("By signal type", s["by_grade"])
        self._tr_table("By confidence score (higher should win more)", s["by_score"])

        c = card(body, "Recent signals")
        status_pill = {"pending": ("WAITING", MUTED), "open": ("OPEN", BLUE), "expired": ("EXPIRED", "#3A4152")}
        for _, r in df.sort_values("id", ascending=False).head(15).iterrows():
            row = ctk.CTkFrame(c, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=2)
            side_txt, side_col = ("BUY", GREEN) if r["side"] > 0 else ("SELL", RED)
            pill(row, side_txt, side_col, 9).pack(side="left")
            ctk.CTkLabel(row, text=r["symbol"].replace("USDT", ""), font=("Segoe UI", 12, "bold"),
                         text_color=FG, width=60, anchor="w").pack(side="left", padx=6)
            ctk.CTkLabel(row, text=f"{sl.GRADE_LABEL.get(r['grade'], r['grade'])} · {r['score']}/100 · "
                                   f"{r['created']:%m-%d %H:%M} UTC", font=("Segoe UI", 10), text_color=MUTED).pack(side="left")
            if r["status"] == "closed":
                won = r["r"] > 0
                pill(row, f"{'WIN' if won else 'LOSS'} {r['r']:+.2f}R", GREEN if won else RED, 9).pack(side="right")
            else:
                lab, col = status_pill.get(r["status"], (str(r["status"]).upper(), MUTED))
                pill(row, lab, col, 9).pack(side="right")
        ctk.CTkFrame(c, fg_color="transparent", height=6).pack()

    def _open_symbol(self, sym):
        self.symbol_var.set(sym)
        self.tabview.set("Analysis")
        self.run()

    def _finish(self):
        self.busy = False
        self.btn_analyze.configure(state="normal")
        if self.auto_var.get():
            self.job = self.after(30000, self.run)

    def _error(self, e):
        self.status_var.set(f"Analysis error: {e}")
        for w in self.an_body.winfo_children():
            w.destroy()
        text(card(self.an_body, "Analysis failed", RED), str(e), RED, 11, pady=(2, 10))
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

        threading.Thread(target=self._coach_work, args=(q, sym, self.style_var.get(), self.model_var.get().strip(),
                                                        self.backend_var.get().lower()), daemon=True).start()

    def _coach_plan(self, sym, style):
        """Reuse the plan for follow-up questions (same coin + style, under 2 minutes old)."""
        key, now = (sym, style), pd.Timestamp.now()
        cached = getattr(self, "_coach_cache", None)
        if cached and cached[0] == key and now - cached[1] < pd.Timedelta(minutes=2):
            return cached[2]
        plan = tc.ai.build_plan(lambda s, tf: get_klines(s, tf, 500), sym, style, 1000.0, 1.0)
        self._coach_cache = (key, now, plan)
        return plan

    def _coach_work(self, q, sym, style, model, backend):
        try:
            plan = self._coach_plan(sym, style)
            ans = tc.ask(q, plan, self.chat_hist, tc.journal_text(), backend, model=model,
                         on_token=lambda t: self.q.put((self._update_active_bot_bubble, (t,))))
            self.chat_hist += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
        except Exception as e:
            self.q.put((self._update_active_bot_bubble, (f"\n[Error: {e}]",)))
        finally:
            self.coach_busy = False

    # ---------------- News alerts ----------------
    def _build_news_tab(self, parent):
        if na is None:
            ctk.CTkLabel(parent, text=f"News unavailable.\n{NEWS_ERR}", text_color=RED, justify="left").pack(
                fill="both", expand=True, padx=12, pady=12)
            return
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(8, 2))
        self.news_search = ctk.CTkEntry(row, placeholder_text="Search coin...", width=150)
        self.news_search.pack(side="left", padx=(0, 6))
        self.news_search.bind("<KeyRelease>", lambda _e: self._news_fill_list())
        ctk.CTkButton(row, text="+ Current", width=76, fg_color="#3A4152", command=self._news_follow_current).pack(side="left", padx=2)
        ctk.CTkButton(row, text="Clear all", width=70, fg_color="#3A4152", command=self._news_clear).pack(side="left", padx=2)

        self.news_follow_lbl = ctk.CTkLabel(parent, text="", font=("Segoe UI", 11), text_color=AMBER, anchor="w", justify="left", wraplength=400)
        self.news_follow_lbl.pack(fill="x", padx=12, pady=(2, 2))

        self.news_list = ctk.CTkScrollableFrame(parent, height=150, fg_color=BG, border_width=1, border_color=PANEL_BORDER)
        self.news_list.pack(fill="x", padx=8, pady=2)
        self.news_vars = {}

        row2 = ctk.CTkFrame(parent, fg_color="transparent")
        row2.pack(fill="x", padx=8, pady=(6, 2))
        ctk.CTkLabel(row2, text="Sensitivity:", font=("Segoe UI", 11)).pack(side="left", padx=(4, 4))
        self.news_sens = ctk.CTkSlider(row2, from_=1, to=10, number_of_steps=9, width=110, command=self._news_sens)
        self.news_sens.set(na.get_user(NEWS_ID).get("threshold", na.DEFAULT_THRESHOLD))
        self.news_sens.pack(side="left", padx=4)
        self.news_sens_lbl = ctk.CTkLabel(row2, text="", font=("Segoe UI", 11), text_color=MUTED, width=70)
        self.news_sens_lbl.pack(side="left")
        ctk.CTkButton(row2, text="Check now", width=80, command=lambda: self._news_check(True)).pack(side="right", padx=2)
        ctk.CTkButton(row2, text="Today", width=60, fg_color="#3A4152", command=self._news_latest).pack(side="right", padx=2)

        self.news_auto = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(parent, text="Auto-check every 5 min (today's news only)", variable=self.news_auto,
                      font=("Segoe UI", 11)).pack(anchor="w", padx=12, pady=(4, 2))

        self.news_box = ctk.CTkTextbox(parent, font=("Segoe UI", 11), fg_color=BG, border_width=1,
                                       border_color=PANEL_BORDER, wrap="word")
        self.news_box.pack(fill="both", expand=True, padx=8, pady=8)
        self.news_box.insert("end", "Tick coins in the list. High-impact news from today shows up here with a popup.\n")
        self._news_fill_list()
        self._news_refresh_labels()

    def _news_symbols(self):
        syms = [c[:-4] if c.endswith("USDT") else c for c in self.coins]
        mine = na.get_user(NEWS_ID)["coins"]
        return syms + [c for c in mine if c not in syms]

    def _news_fill_list(self):
        if na is None or not hasattr(self, "news_list"):
            return
        for w in self.news_list.winfo_children():
            w.destroy()
        mine = set(na.get_user(NEWS_ID)["coins"])
        flt = self.news_search.get().strip().upper()
        self.news_vars = {}
        shown = [s for s in self._news_symbols() if flt in s][:90]
        for i, sym in enumerate(shown):
            var = ctk.BooleanVar(value=sym in mine)
            self.news_vars[sym] = var
            ctk.CTkCheckBox(self.news_list, text=sym, variable=var, width=90, checkbox_width=18, checkbox_height=18,
                            command=lambda s=sym: self._news_toggle(s)).grid(row=i // 3, column=i % 3, padx=6, pady=3, sticky="w")

    def _news_toggle(self, sym):
        if self.news_vars[sym].get():
            na.follow(NEWS_ID, [sym])
            self._news_check(True)
        else:
            na.unfollow(NEWS_ID, [sym])
        self._news_refresh_labels()

    def _news_refresh_labels(self):
        u = na.get_user(NEWS_ID)
        self.news_follow_lbl.configure(text="Following: " + (", ".join(u["coins"]) or "nothing yet"))
        self.news_sens_lbl.configure(text=f"alert >= {u['threshold']}/10")

    def _news_follow_current(self):
        na.follow(NEWS_ID, [self.symbol_var.get()])
        self._news_fill_list()
        self._news_refresh_labels()
        self._news_check(True)

    def _news_clear(self):
        na.unfollow(NEWS_ID, na.get_user(NEWS_ID)["coins"])
        self._news_fill_list()
        self._news_refresh_labels()

    def _news_sens(self, v):
        na.set_threshold(NEWS_ID, round(v))
        self._news_refresh_labels()

    def _news_tick(self):
        if na is not None and self.news_auto.get():
            self._news_check(False)
        self.after(NEWS_EVERY_MS, self._news_tick)

    def _news_check(self, manual=False):
        if na is None or self.news_busy:
            return
        if not na.get_user(NEWS_ID)["coins"]:
            if manual:
                self.status_var.set("News: follow a coin first.")
            return
        self.news_busy = True

        def work():
            msg = None
            try:
                n = na.run_check(lambda _cid, text: self.q.put((self._news_alert, (text,))))
                if manual or n:
                    msg = f"News check done: {n} new alert(s) today ({na.LAST[0]}/{na.LAST[1]} sources answered)."
            except Exception as e:
                msg = f"News check failed: {e}"
            self.q.put((self._news_done, (msg,)))
        threading.Thread(target=work, daemon=True).start()

    def _news_done(self, msg):
        self.news_busy = False
        if msg:
            self.status_var.set(msg)

    def _news_latest(self):
        coins = na.get_user(NEWS_ID)["coins"]
        if not coins:
            self.status_var.set("News: follow a coin first.")
            return
        self.status_var.set("Fetching latest news...")

        def work():
            try:
                items = na.latest_for(coins)
                text = "\n\n".join(items) if items else "No news about your coins today yet."
            except Exception as e:
                text = f"[Could not fetch news: {e}]"
            self.q.put((self._news_show, (text,)))
        threading.Thread(target=work, daemon=True).start()

    def _news_show(self, text):
        self.news_box.delete("1.0", "end")
        self.news_box.insert("end", text + "\n")
        self.status_var.set("Latest news loaded.")

    def _news_alert(self, text):
        self.news_box.insert("1.0", text + "\n" + "-" * 48 + "\n\n")
        self.status_var.set("News alert: " + text.split("\n")[0])
        self.bell()
        self._news_popup(text)

    def _news_popup(self, text):
        try:
            win = ctk.CTkToplevel(self)
            win.title("News alert")
            win.geometry("420x230+40+40")
            win.attributes("-topmost", True)
            win.configure(fg_color=PANEL)
            ctk.CTkLabel(win, text=text.split("\nSource:")[0], wraplength=390, justify="left",
                         font=("Segoe UI", 12)).pack(fill="both", expand=True, padx=14, pady=(14, 6))
            ctk.CTkButton(win, text="Dismiss", command=win.destroy).pack(pady=(0, 12))
            win.after(15000, lambda: win.winfo_exists() and win.destroy())
        except Exception:
            pass

    # ---------------- Backtest ----------------
    def _build_backtest_tab(self, parent):
        if tb is None:
            ctk.CTkLabel(parent, text=f"Backtest unavailable.\n{BT_ERR}", text_color=RED, justify="left").pack(
                fill="both", expand=True, padx=12, pady=12)
            return
        r1 = ctk.CTkFrame(parent, fg_color="transparent")
        r1.pack(fill="x", padx=4, pady=(6, 2))
        ctk.CTkLabel(r1, text="Strategy:", font=("Segoe UI", 11)).pack(side="left", padx=(4, 4))
        self.bt_strategy = ctk.StringVar(value="all")
        ctk.CTkOptionMenu(r1, variable=self.bt_strategy, values=["all"] + list(tb.STRATEGIES), width=130).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(r1, text="Years:", font=("Segoe UI", 11)).pack(side="left", padx=(0, 4))
        self.bt_years = ctk.StringVar(value="2")
        ctk.CTkOptionMenu(r1, variable=self.bt_years, values=["1", "2", "3"], width=60).pack(side="left")

        r2 = ctk.CTkFrame(parent, fg_color="transparent")
        r2.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(r2, text="Mode:", font=("Segoe UI", 11)).pack(side="left", padx=(4, 4))
        self.bt_mode = ctk.StringVar(value="Default")
        ctk.CTkOptionMenu(r2, variable=self.bt_mode, values=["Default", "Optimize", "Walk-forward"], width=120).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(r2, text="Risk %:", font=("Segoe UI", 11)).pack(side="left", padx=(0, 4))
        self.bt_risk = ctk.CTkEntry(r2, width=44)
        self.bt_risk.insert(0, "1")
        self.bt_risk.pack(side="left")

        r3 = ctk.CTkFrame(parent, fg_color="transparent")
        r3.pack(fill="x", padx=4, pady=(4, 2))
        self.bt_btn = ctk.CTkButton(r3, text="Run backtest", font=("Segoe UI", 11, "bold"), fg_color=BLUE, command=self._bt_run)
        self.bt_btn.pack(side="left", padx=(4, 6))
        ctk.CTkButton(r3, text="Explain with Coach", fg_color="#3A4152", command=self._bt_to_coach).pack(side="left")

        ctk.CTkLabel(parent, text="Uses the coin and TF from the top bar. Default = fixed settings, Optimize = fit on 70% and "
                                  "judge on the unseen 30%, Walk-forward = re-fit before each test window.",
                     font=("Segoe UI", 10), text_color=MUTED, wraplength=420, justify="left", anchor="w").pack(fill="x", padx=10, pady=(2, 2))

        chart = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=8, border_width=1, border_color=PANEL_BORDER)
        chart.pack(fill="x", padx=8, pady=4)
        self.bt_fig = Figure(figsize=(4.4, 2.2), dpi=100)
        self.bt_canvas = FigureCanvasTkAgg(self.bt_fig, master=chart)
        self.bt_canvas.get_tk_widget().pack(fill="x", padx=2, pady=2)
        self._bt_draw({})

        self.bt_cards = ctk.CTkScrollableFrame(parent, fg_color=BG, border_width=1, border_color=PANEL_BORDER, corner_radius=8)
        self.bt_cards.pack(fill="both", expand=True, padx=8, pady=(4, 4))
        self.bt_details_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(parent, text="Show full technical report", variable=self.bt_details_var, font=("Segoe UI", 10),
                      command=self._bt_toggle_details).pack(anchor="w", padx=12, pady=(0, 6))
        self.bt_box = ctk.CTkTextbox(parent, height=220, font=("Consolas", 10), fg_color=BG, border_width=1,
                                     border_color=PANEL_BORDER, wrap="word")      # shown only when the switch is on
        self.bt_res = None
        self.replays = []
        self._bt_message("Pick a strategy and press Run backtest.\nThe first run downloads and caches history "
                         "(a minute or so); later runs are fast.\nThen press 'Watch the trades' to see every entry on the chart.")

    def _bt_draw(self, curves):
        fig = self.bt_fig
        fig.clear()
        fig.patch.set_facecolor(PANEL)
        ax = fig.add_subplot(111)
        style_ax(ax)
        drawn = False
        for i, (name, (r, split)) in enumerate(curves.items()):
            if len(r) == 0:
                continue
            cum = np.r_[0.0, np.cumsum(r)]
            ax.plot(cum, color=LINE_COLORS[i % len(LINE_COLORS)], lw=1.4, label=name)
            if split and 0 < split < len(r):
                ax.axvline(split, color=MUTED, ls=":", lw=0.8)
            drawn = True
        ax.axhline(0, color="#9AA3B5", lw=0.8)
        ax.set_title("Cumulative R after costs" + ("  (dotted line = last 30% starts)" if drawn else ""),
                     color=FG, fontsize=9, loc="left")
        ax.set_xlabel("trade #", color=MUTED, fontsize=8)
        if drawn:
            ax.legend(loc="upper left", fontsize=7, facecolor=PANEL, edgecolor=PANEL_BORDER, labelcolor=FG)
        else:
            ax.text(0.5, 0.5, "no results yet", transform=ax.transAxes, ha="center", color=MUTED, fontsize=9)
        fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.2)
        self.bt_canvas.draw_idle()

    def _bt_run(self):
        if tb is None or self.bt_busy:
            return
        try:
            sym = normalize_symbol(self.symbol_var.get())
            tf = self.interval_var.get()
            if tf not in tb.TF_MIN:
                raise ValueError(f"Backtests do not support the '{tf}' timeframe. Pick 1m-1w in the top bar.")
            risk = float(self.bt_risk.get().replace(",", "."))
            if not 0.05 <= risk <= 10:
                raise ValueError("Risk % must be between 0.05 and 10.")
            years = float(self.bt_years.get())
        except Exception as e:
            self.status_var.set(f"Backtest: {e}")
            return
        mode = {"Default": "default", "Optimize": "optimize", "Walk-forward": "walkforward"}[self.bt_mode.get()]
        strat = self.bt_strategy.get()
        self.symbol_var.set(sym)
        self.bt_busy = True
        self.bt_btn.configure(state="disabled")
        self.status_var.set(f"Backtest started: {sym} {tf}...")

        def work():
            try:
                res = tb.run_backtest(sym, tf, strat, mode, years, risk, False,
                                      progress=lambda m: self.q.put((self.status_var.set, (m,))))
                self.q.put((self._bt_show, (res,)))
            except Exception as e:
                self.q.put((self._bt_error, (e,)))
        threading.Thread(target=work, daemon=True).start()

    def _bt_toggle_details(self):
        if self.bt_details_var.get():
            self.bt_box.pack(fill="x", padx=8, pady=(0, 8))
        else:
            self.bt_box.pack_forget()

    def _bt_message(self, text, color=MUTED):
        for w in self.bt_cards.winfo_children():
            w.destroy()
        ctk.CTkLabel(self.bt_cards, text=text, font=("Segoe UI", 11), text_color=color, wraplength=390,
                     justify="left", anchor="w").pack(fill="x", padx=8, pady=8)

    def _bt_show(self, res):
        self.bt_res = res
        self.bt_text = res["text"]
        self._bt_draw(res["curves"])
        self.bt_box.delete("1.0", "end")
        self.bt_box.insert("end", res["text"] + "\n")
        self._bt_fill_cards(res)
        self.bt_busy = False
        self.bt_btn.configure(state="normal")
        self.status_var.set("Backtest finished.")

    def _bt_fill_cards(self, res):
        for w in self.bt_cards.winfo_children():
            w.destroy()
        ctk.CTkLabel(self.bt_cards, text="R = the money you risk on one trade (distance to your stop). "
                                         "+2R = you made twice what you risked, -1R = you lost it.",
                     font=("Segoe UI", 10), text_color=MUTED, wraplength=390, justify="left", anchor="w").pack(fill="x", padx=8, pady=(2, 6))
        colors = {"good": GREEN, "bad": RED, "maybe": AMBER, "nodata": MUTED}
        for card in res["cards"]:
            colr = colors[card["kind"]]
            f = ctk.CTkFrame(self.bt_cards, fg_color=PANEL, corner_radius=8, border_width=1, border_color=colr)
            f.pack(fill="x", padx=4, pady=5)
            head = ctk.CTkFrame(f, fg_color="transparent")
            head.pack(fill="x", padx=8, pady=(6, 0))
            ctk.CTkLabel(head, text=card["name"], font=("Segoe UI", 13, "bold")).pack(side="left")
            ctk.CTkLabel(head, text=card["badge"], font=("Segoe UI", 10, "bold"), text_color="#0B0E14",
                         fg_color=colr, corner_radius=6, width=110).pack(side="right")
            for line in card["lines"]:
                ctk.CTkLabel(f, text=line, font=("Segoe UI", 11), wraplength=380, justify="left", anchor="w",
                             text_color=FG).pack(fill="x", padx=10, pady=1)
            ctk.CTkButton(f, text="▶ Watch the trades", fg_color=BLUE, height=28,
                          state="normal" if card["has_trades"] and trp is not None else "disabled",
                          command=lambda n=card["name"]: self._bt_open_replay(n)).pack(anchor="w", padx=10, pady=(6, 8))

    def _bt_open_replay(self, name):
        if trp is None or not self.bt_res:
            self.status_var.set("Replay unavailable: trade_replay.py must be in the same folder.")
            return
        res = self.bt_res
        w = trp.ReplayWindow(self, res["df"], res["trades"][name], title=f"{res['symbol']} {res['tf']} · {name}")
        self.replays.append(w)

    def _bt_error(self, e):
        self.bt_busy = False
        self.bt_btn.configure(state="normal")
        self.status_var.set(f"Backtest failed: {e}")
        self._bt_message(f"Backtest failed: {e}", RED)

    def _bt_to_coach(self):
        if tc is None:
            self.status_var.set("Coach is unavailable.")
            return
        if not self.bt_text:
            self.status_var.set("Run a backtest first.")
            return
        self.tabview.set("Coach")
        q = ("Here are my backtest results. Use only these numbers.\n\n" + self.bt_text[:5000] +
             "\n\nIn plain language: is there evidence of an edge, how reliable is it, what could go wrong, "
             "and how much should I risk?")
        self._coach_send(q, "Interpret my backtest results.")

    def _coach_clear(self):
        self.chat_hist = []
        for widget in self.chat_container.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()