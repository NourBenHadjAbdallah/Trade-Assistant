#!/usr/bin/env python3
"""
trade_replay.py - animated replay of backtest trades (used by trade_assistant_gui.py).

Plays the price chart candle by candle. Every entry is marked with an arrow
(green up = BUY / long, red down = SELL / short), the position is drawn as a green
profit zone (up to the target) and a red risk zone (down to the stop), and each trade
shows its result (+2.0R / -1.0R) when it closes. A second chart grows the equity curve.

Keys: Space = play/pause, Right = next trade, Left = previous trade.
"""
import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

BG, PANEL, BORDER = "#0B0E14", "#151922", "#232834"
FG, MUTED = "#E1E6F0", "#788296"
GREEN, RED, BLUE, AMBER = "#26A69A", "#EF5350", "#2962FF", "#FFB300"


def _style(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color="#1E2330", lw=0.6, ls=":")


class ReplayWindow(ctk.CTkToplevel):
    WINDOW = 110            # candles visible at once

    def __init__(self, master, df, trades, title="Trade replay"):
        super().__init__(master)
        self.title(title)
        self.geometry("1180x780")
        self.configure(fg_color=BG)
        self.df = df.reset_index(drop=True)
        self.o, self.h, self.l, self.c = (self.df[k].values for k in ("open", "high", "low", "close"))
        self.times = self.df["time"].reset_index(drop=True)
        t = trades.sort_values("entry_idx").reset_index(drop=True)
        self.N = len(t)
        self.ei = t["entry_idx"].values.astype(int)
        self.xi = t["exit_idx"].values.astype(int)
        self.side = t["side"].values.astype(int)
        self.ent, self.stp, self.tgt = (t[k].values.astype(float) for k in ("entry", "stop", "target"))
        self.r = t["r"].values.astype(float)
        self.reason = t["reason"].values
        self.exitp = np.array([self.tgt[k] if self.reason[k] == "target" else self.stp[k] if self.reason[k] == "stop"
                               else self.c[self.xi[k]] for k in range(self.N)])
        self.cum = np.r_[0.0, np.cumsum(self.r)]
        self.last = min(len(self.df) - 1, (int(self.xi.max()) + 25) if self.N else len(self.df) - 1)
        self.pos = max(int(self.ei[0]) - 25, 5) if self.N else min(self.WINDOW, len(self.df) - 1)
        self.playing, self.job = False, None

        ctk.CTkLabel(self, text=title, font=("Segoe UI", 15, "bold"), anchor="w").pack(fill="x", padx=14, pady=(10, 0))
        ctk.CTkLabel(self, text="▲ green arrow = BUY entry    ▼ red arrow = SELL entry    green zone = profit target    "
                                "red zone = stop (where you are wrong)    ● = trade closed, with its result in R",
                     font=("Segoe UI", 10), text_color=MUTED, anchor="w").pack(fill="x", padx=14, pady=(0, 4))

        body = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=10, border_width=1, border_color=BORDER)
        body.pack(fill="both", expand=True, padx=12, pady=4)
        self.fig = Figure(figsize=(11, 6.2), dpi=100)
        self.fig.patch.set_facecolor(BG)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3.4, 1], hspace=0.22, left=0.06, right=0.95, top=0.95, bottom=0.07)
        self.ax, self.ax2 = self.fig.add_subplot(gs[0]), self.fig.add_subplot(gs[1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        self.info = ctk.CTkLabel(self, text="", font=("Consolas", 12), anchor="w", justify="left")
        self.info.pack(fill="x", padx=14, pady=(2, 2))

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=12, pady=(2, 10))
        self.btn_prev = ctk.CTkButton(bar, text="⏮ Prev trade", width=100, fg_color="#3A4152", command=self.prev_trade)
        self.btn_play = ctk.CTkButton(bar, text="▶ Play", width=90, fg_color=BLUE, command=self.toggle)
        self.btn_next = ctk.CTkButton(bar, text="Next trade ⏭", width=100, fg_color="#3A4152", command=self.next_trade)
        self.btn_restart = ctk.CTkButton(bar, text="↺ Restart", width=80, fg_color="#3A4152", command=self.restart)
        for b in (self.btn_prev, self.btn_play, self.btn_next, self.btn_restart):
            b.pack(side="left", padx=3)
        ctk.CTkLabel(bar, text="Speed:", font=("Segoe UI", 11)).pack(side="left", padx=(14, 4))
        self.speed = ctk.CTkSlider(bar, from_=1, to=8, number_of_steps=7, width=130)
        self.speed.set(2)
        self.speed.pack(side="left")
        self.slow_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(bar, text="Slow down at entries and exits", variable=self.slow_var,
                      font=("Segoe UI", 11)).pack(side="left", padx=16)

        self.bind("<space>", lambda _e: self.toggle())
        self.bind("<Right>", lambda _e: self.next_trade())
        self.bind("<Left>", lambda _e: self.prev_trade())
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(150, self.lift)
        self.draw()

    # ---------------- playback ----------------
    def _close(self):
        self.playing = False
        if self.job:
            self.after_cancel(self.job)
        self.destroy()

    def toggle(self):
        if self.pos >= self.last:
            self.pos = max(int(self.ei[0]) - 25, 5) if self.N else 5
        self.playing = not self.playing
        self.btn_play.configure(text="⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            self._schedule(30)

    def _schedule(self, delay):
        if self.job:
            self.after_cancel(self.job)
        self.job = self.after(delay, self._tick)

    def _tick(self):
        self.job = None
        if not self.playing:
            return
        new, delay = min(self.pos + int(round(self.speed.get())), self.last), 60
        if self.slow_var.get() and self.N:
            ev = np.r_[self.ei[(self.ei > self.pos) & (self.ei <= new)], self.xi[(self.xi > self.pos) & (self.xi <= new)]]
            if len(ev):
                new, delay = int(ev.min()), 1000          # stop on the event and let the eye catch up
        self.pos = new
        self.draw()
        if self.pos >= self.last:
            self.playing = False
            self.btn_play.configure(text="▶ Play again")
            return
        self._schedule(delay)

    def restart(self):
        self.playing = False
        self.btn_play.configure(text="▶ Play")
        self.pos = max(int(self.ei[0]) - 25, 5) if self.N else 5
        self.draw()

    def next_trade(self):
        k = int(np.searchsorted(self.ei, self.pos, side="right"))
        if k < self.N:
            self.pos = int(self.ei[k])
            self.draw()

    def prev_trade(self):
        k = int(np.searchsorted(self.ei, self.pos, side="left")) - 1
        if k >= 0:
            self.pos = int(self.ei[k])
            self.draw()

    # ---------------- drawing ----------------
    def draw(self):
        pos, W = self.pos, self.WINDOW
        lo = max(0, pos - W + 1)
        x = np.arange(lo, pos + 1)
        o, h, l, c = (a[lo:pos + 1] for a in (self.o, self.h, self.l, self.c))
        ax = self.ax
        ax.clear()
        _style(ax)
        col = np.where(c >= o, GREEN, RED)
        ax.vlines(x, l, h, colors=col, lw=0.9, zorder=3)
        ax.bar(x, np.maximum(np.abs(c - o), (h - l) * 0.03), bottom=np.minimum(o, c), color=col, width=0.7, zorder=3)
        ax.axhline(c[-1], color="#FFFFFF", lw=0.7, ls=":", alpha=0.5)

        ys = [float(l.min()), float(h.max())]
        open_txt = "No open position."
        done = int((self.xi <= pos).sum())
        for k in np.where((self.ei <= pos) & (self.xi >= lo))[0]:
            e, xx, sd = int(self.ei[k]), int(self.xi[k]), int(self.side[k])
            closed = xx <= pos
            right = xx if closed else pos + 3
            a = 0.09 if closed else 0.22
            ax.add_patch(Rectangle((e, self.ent[k]), right - e, self.tgt[k] - self.ent[k], color=GREEN, alpha=a, lw=0, zorder=1))
            ax.add_patch(Rectangle((e, self.ent[k]), right - e, self.stp[k] - self.ent[k], color=RED, alpha=a, lw=0, zorder=1))
            ax.hlines(self.ent[k], e, right, colors=FG, lw=0.8, alpha=0.7, zorder=2)
            ax.scatter([e], [self.ent[k]], marker="^" if sd > 0 else "v", s=130, color=GREEN if sd > 0 else RED,
                       edgecolors="white", linewidths=1.2, zorder=6)
            ax.annotate("BUY" if sd > 0 else "SELL", (e, self.ent[k]), textcoords="offset points",
                        xytext=(0, -16 if sd > 0 else 12), ha="center", fontsize=8, color=FG, fontweight="bold", zorder=7)
            ys += [self.ent[k], self.stp[k], self.tgt[k]]
            if closed:
                good = self.r[k] > 0
                ax.plot([e, xx], [self.ent[k], self.exitp[k]], color=GREEN if good else RED, lw=1.2, ls="--", zorder=4)
                ax.scatter([xx], [self.exitp[k]], marker="o", s=70, color=GREEN if good else RED,
                           edgecolors="white", linewidths=1.2, zorder=6)
                ax.annotate(f"{self.r[k]:+.1f}R", (xx, self.exitp[k]), textcoords="offset points", xytext=(8, 8 if good else -14),
                            fontsize=9, color=GREEN if good else RED, fontweight="bold", zorder=7)
            else:
                ax.annotate(f"TARGET {self.tgt[k]:.6g}", (right, self.tgt[k]), fontsize=7.5, color=GREEN, va="center", ha="left")
                ax.annotate(f"STOP {self.stp[k]:.6g}", (right, self.stp[k]), fontsize=7.5, color=RED, va="center", ha="left")
                open_txt = (f"OPEN {'BUY' if sd > 0 else 'SELL'} @ {self.ent[k]:.6g}   stop {self.stp[k]:.6g}   "
                            f"target {self.tgt[k]:.6g}   now {'+' if sd * (c[-1] - self.ent[k]) >= 0 else '-'}"
                            f"{abs(c[-1] - self.ent[k]) / abs(self.ent[k] - self.stp[k]):.2f}R")
        pad = (max(ys) - min(ys)) * 0.06 or 1
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_xlim(lo - 1, max(lo + W, pos) + 14)
        ax.xaxis.set_major_locator(MaxNLocator(7))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _: self.times.iloc[int(round(v))].strftime("%m-%d %H:%M") if 0 <= int(round(v)) < len(self.df) else ""))
        ax.set_title(f"{self.times.iloc[pos]:%Y-%m-%d %H:%M}   candle {pos + 1}/{len(self.df)}", color=FG, fontsize=10, loc="left")

        ax2 = self.ax2
        ax2.clear()
        _style(ax2)
        ax2.plot(range(self.N + 1), self.cum, color="#3A4152", lw=1)
        cur = float(self.cum[done])
        ax2.plot(range(done + 1), self.cum[:done + 1], color=GREEN if cur >= 0 else RED, lw=2)
        ax2.scatter([done], [cur], color=GREEN if cur >= 0 else RED, s=30, zorder=5)
        ax2.axhline(0, color="#9AA3B5", lw=0.8)
        ax2.set_ylabel("Total R", color=MUTED, fontsize=8)
        ax2.set_xlabel("trade #", color=MUTED, fontsize=8)
        ax2.set_xlim(0, max(self.N, 1))

        wins = int((self.r[:done] > 0).sum())
        self.info.configure(text=f"Closed trades {done}/{self.N}   wins {wins}  losses {done - wins}   total {cur:+.1f}R\n{open_txt}")
        self.canvas.draw_idle()
