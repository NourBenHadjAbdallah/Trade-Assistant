#!/usr/bin/env python3
"""
trade_chat.py - talk with your trade assistant using a LOCAL Ollama model.

Put it in the same folder as trade_ai.py, trade_assistant.py and trade_journal.py.

Setup (once):
    1. Install Ollama:  https://ollama.com
    2. ollama pull qwen2.5:7b          (good in Arabic/French/English; llama3.1:8b also works)
    3. pip install requests pandas numpy

Run:
    python trade_chat.py                       # live Binance data
    python trade_chat.py --demo                # synthetic data
    python trade_chat.py --symbol ETHUSDT --style Swing --model llama3.1:8b

Commands inside the chat:
    /plan [SYMBOL] [STYLE]   build a fresh multi-timeframe plan (STYLE: Scalp, Intraday, Swing, Position)
    /journal                 show your journal stats + coaching notes
    /models                  list models installed in Ollama
    /model NAME              switch model
    /reset                   forget the conversation
    /quit                    exit

Env vars: OLLAMA_HOST (default http://localhost:11434), OLLAMA_MODEL.
"""
import argparse
import json
import os
import sys

import requests

import trade_ai as ai

try:
    import trade_journal as tj
except Exception:          # journal is optional
    tj = None

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

SYSTEM_PROMPT = """You are the trading coach inside a crypto trading app.
The app already computed a multi-timeframe analysis with deterministic code. It is given below as plain text,
plus the user's trade-journal statistics when available.

Rules:
- Use ONLY numbers that appear in the data below. Never invent prices, indicators, news or fundamentals.
  If something is not in the data, say you do not have it.
- Discuss like a mentor: explain what each timeframe is doing, why this strategy was chosen, where the idea is wrong,
  and how much to risk. Answer the user's actual question first, then add context.
- If the user's idea fights the higher timeframe or has poor reward:risk, say so directly and explain why.
- Never promise outcomes. Candle-level order flow is an approximation. At most one short reminder that this is not financial advice.
- If journal stats show a habit (cutting winners, revenge trading, ignoring stops), connect it to the current setup.
- Reply in the same language the user writes in (Arabic, French or English).
- Keep answers short: a few short paragraphs or a short list."""

BRIEFING = ("Explain this setup in plain language: the story across the timeframes, why this strategy, "
            "what must happen before I enter, where I'm wrong, and what mistake I'm most likely to make here.")


# ----------------------------------------------------------------------------
# Ollama client
# ----------------------------------------------------------------------------
def list_models(host=OLLAMA_HOST):
    r = requests.get(f"{host}/api/tags", timeout=5)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def check_ollama(model, host=OLLAMA_HOST):
    """Return the model name to use, or raise RuntimeError with a fix-it message."""
    try:
        models = list_models(host)
    except requests.RequestException:
        raise RuntimeError(f"Cannot reach Ollama at {host}. Start it (open the Ollama app, or run `ollama serve`).")
    if not models:
        raise RuntimeError(f"Ollama is running but has no models. Run:  ollama pull {model}")
    if model in models or any(m.split(":")[0] == model for m in models):
        return model
    print(f"Model '{model}' is not installed. Using '{models[0]}' instead "
          f"(or run `ollama pull {model}`).")
    return models[0]


def build_system(plan=None, journal_text=None):
    s = SYSTEM_PROMPT
    # plain text is easier for small local models than a big JSON blob
    s += "\n\n<analysis>\n" + (ai.render_plan_text(plan) if plan else "No analysis has been run yet.") + "\n</analysis>"
    if journal_text:
        s += "\n\n<journal_stats>\n" + journal_text + "\n</journal_stats>"
    return s


def ask_ollama(question, plan=None, history=None, journal_text=None, model=DEFAULT_MODEL,
               host=OLLAMA_HOST, on_token=None, timeout=300):
    """
    Send one question to Ollama and return the full answer.
    history: list of {"role": "user"|"assistant", "content": str} (last 12 are used)
    on_token: optional callback(str) called for each streamed piece (for live printing / GUI).
    """
    msgs = [{"role": "system", "content": build_system(plan, journal_text)}]
    msgs += list(history or [])[-12:]
    msgs.append({"role": "user", "content": question})
    body = {"model": model, "messages": msgs, "stream": True,
            "options": {"temperature": 0.3, "num_ctx": 8192}}
    parts = []
    try:
        with requests.post(f"{host}/api/chat", json=body, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                try:
                    err = r.json().get("error", r.text)
                except Exception:
                    err = r.text
                raise RuntimeError(f"Ollama error {r.status_code}: {err}")
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(f"Ollama error: {chunk['error']}")
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    parts.append(piece)
                    if on_token:
                        on_token(piece)
                if chunk.get("done"):
                    break
    except requests.ConnectionError:
        raise RuntimeError(f"Lost connection to Ollama at {host}.")
    return "".join(parts).strip()


# ----------------------------------------------------------------------------
# Journal + plan helpers
# ----------------------------------------------------------------------------
def journal_text():
    if tj is None:
        return None
    try:
        df = tj.load()
        if df.empty:                      # empty journal crashes tj.stats(), so skip it
            return None
        s = tj.stats(df)
        return None if s is None else tj.format_stats(s)
    except Exception as e:
        print(f"(journal unavailable: {e})")
        return None


def make_plan(symbol, style, account, risk, demo):
    provider = ai.demo_provider if demo else ai.binance_provider
    return ai.build_plan(provider, symbol.upper(), style, account, risk)


# ----------------------------------------------------------------------------
# Chat loop
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Discuss your trade plan with a local Ollama model")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--style", choices=list(ai.STYLES), default="Intraday")
    p.add_argument("--account", type=float, default=1000.0)
    p.add_argument("--risk", type=float, default=1.0, help="risk per trade, percent of account")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--demo", action="store_true", help="synthetic data instead of Binance")
    a = p.parse_args()

    try:
        model = check_ollama(a.model)
    except RuntimeError as e:
        sys.exit(str(e))

    state = {"symbol": a.symbol.upper(), "style": a.style, "plan": None, "history": [], "model": model}

    def refresh_plan(symbol=None, style=None):
        state["symbol"] = (symbol or state["symbol"]).upper()
        state["style"] = style or state["style"]
        print(f"Analyzing {state['symbol']} ({state['style']})...")
        state["plan"] = make_plan(state["symbol"], state["style"], a.account, a.risk, a.demo)
        state["history"] = []        # new data, new conversation
        print(ai.render_plan_text(state["plan"]))

    def say(question):
        print(f"\nCoach ({state['model']}): ", end="", flush=True)
        try:
            ans = ask_ollama(question, state["plan"], state["history"], journal_text(),
                             model=state["model"], on_token=lambda t: print(t, end="", flush=True))
        except RuntimeError as e:
            print(f"\n[{e}]\n")
            return
        print("\n")
        state["history"] += [{"role": "user", "content": question}, {"role": "assistant", "content": ans}]

    try:
        refresh_plan()
    except Exception as e:
        print(f"Could not build a plan ({e}). You can still chat; try /plan or --demo.")

    print("\n" + "=" * 64)
    print("Ask anything about the plan, your journal, or trading concepts. Type /quit to exit.")
    print("=" * 64)
    if state["plan"]:
        say(BRIEFING)

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.startswith("/"):
            cmd, *rest = q.split()
            cmd = cmd.lower()
            if cmd in ("/quit", "/exit", "/q"):
                break
            elif cmd == "/reset":
                state["history"] = []
                print("Conversation cleared.\n")
            elif cmd == "/models":
                try:
                    print("\n".join(list_models()) + "\n")
                except Exception as e:
                    print(f"[{e}]\n")
            elif cmd == "/model" and rest:
                state["model"] = rest[0]
                print(f"Model set to {rest[0]}\n")
            elif cmd == "/journal":
                print((journal_text() or "No trades logged yet.") + "\n")
            elif cmd == "/plan":
                sym = rest[0] if rest else None
                sty = next((k for k in ai.STYLES if rest[1:] and k.lower() == rest[1].lower()), None)
                try:
                    refresh_plan(sym, sty)
                    say(BRIEFING)
                except Exception as e:
                    print(f"[Could not build plan: {e}]\n")
            else:
                print("Unknown command. Try /plan, /journal, /models, /model NAME, /reset, /quit\n")
            continue
        say(q)


if __name__ == "__main__":
    main()