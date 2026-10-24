import os

import requests
from flask import Flask, request

import news_alerts as na

app = Flask(__name__)

# Set these in Render -> Environment (never in the code):
#   TELEGRAM_TOKEN   token from @BotFather
#   CRON_SECRET      any long random string, protects /cron
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CRON_SECRET = os.environ.get("CRON_SECRET", "")
API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/"
TELEGRAM_API_URL = API + "sendMessage"

COIN_LIST = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT",
             "LTC", "TRX", "TON", "SUI", "MATIC", "ARB", "OP", "SHIB", "ATOM", "NEAR"]

HELP = (
    "Trade Assistant commands (news is always today's news only):\n"
    "/coins               - pick coins from a list (tap to follow/unfollow)\n"
    "/follow BTC ETH SOL  - or type coins yourself\n"
    "/unfollow SOL        - stop following\n"
    "/list                - coins you follow\n"
    "/news BTC            - latest news now\n"
    "/sensitivity 1-10    - lower = more alerts (default 3)"
)


def send(chat_id, text):
    r = requests.post(TELEGRAM_API_URL, json={"chat_id": chat_id, "text": text,
                                              "disable_web_page_preview": True}, timeout=15)
    r.raise_for_status()


def coin_keyboard(chat_id):
    mine = set(na.get_user(chat_id)["coins"])
    extra = [c for c in sorted(mine) if c not in COIN_LIST]
    coins = COIN_LIST + extra
    rows = []
    for i in range(0, len(coins), 4):
        rows.append([{"text": ("✅ " if c in mine else "") + c, "callback_data": "t:" + c}
                     for c in coins[i:i + 4]])
    return {"inline_keyboard": rows}


def send_coin_picker(chat_id):
    requests.post(API + "sendMessage", json={
        "chat_id": chat_id, "text": "Tap coins to follow / unfollow (✅ = following). "
                                    "You get alerts for today's high-impact news.",
        "reply_markup": coin_keyboard(chat_id)}, timeout=15)


def handle_callback(cb):
    chat_id = cb["message"]["chat"]["id"]
    data = cb.get("data", "")
    if data.startswith("t:"):
        now_on, _ = na.toggle(chat_id, data[2:])
        requests.post(API + "editMessageReplyMarkup", json={
            "chat_id": chat_id, "message_id": cb["message"]["message_id"],
            "reply_markup": coin_keyboard(chat_id)}, timeout=15)
        requests.post(API + "answerCallbackQuery", json={
            "callback_query_id": cb["id"],
            "text": ("Following " if now_on else "Unfollowed ") + data[2:]}, timeout=15)


def get_ai_response(user_message):
    """Replace with your real AI chat handler."""
    return f"AI Assistant Received: {user_message}"


def handle_command(chat_id, text):
    cmd, *args = text.split()
    cmd = cmd.split("@")[0].lower()
    if cmd in ("/start", "/help"):
        return HELP
    if cmd == "/coins":
        send_coin_picker(chat_id)
        return ""            # picker already sent
    if cmd == "/follow":
        if not args:
            return "Usage: /follow BTC ETH SOL"
        return "Following: " + ", ".join(na.follow(chat_id, args))
    if cmd == "/unfollow":
        left = na.unfollow(chat_id, args)
        return "Following: " + (", ".join(left) or "nothing")
    if cmd == "/list":
        u = na.get_user(chat_id)
        return f"Following: {', '.join(u['coins']) or 'nothing'}\nSensitivity: {u['threshold']}"
    if cmd == "/sensitivity":
        if args and args[0].isdigit():
            na.set_threshold(chat_id, args[0])
            return f"Alerts now need impact >= {na.get_user(chat_id)['threshold']}/10."
        return "Usage: /sensitivity 1-10"
    if cmd == "/news":
        coins = [a.upper() for a in args] or na.get_user(chat_id)["coins"]
        if not coins:
            return "Follow a coin first: /follow BTC"
        items = na.latest_for(coins)
        return "\n\n---\n\n".join(items) if items else "No news about those coins today yet."
    return None  # not a news command


@app.route("/")
@app.route("/ping")
def health_check():
    return "Trade Assistant Bot is alive!", 200


@app.route("/cron")
def cron():
    """Call every ~10 min from cron-job.org: /cron?key=CRON_SECRET (also keeps Render awake)."""
    if not CRON_SECRET or request.args.get("key") != CRON_SECRET:
        return "forbidden", 403
    try:
        return f"sent {na.run_check(send)}", 200
    except Exception as e:
        return f"error: {e}", 500


@app.route("/webhook", methods=["POST"])
def telegram_webhook():
    data = request.get_json(silent=True) or {}
    if "callback_query" in data:
        try:
            handle_callback(data["callback_query"])
        except Exception as e:
            print("callback error:", e)
        return "OK", 200
    msg = data.get("message")
    if msg and "text" in msg:
        chat_id = msg["chat"]["id"]
        text = msg["text"].strip()
        try:
            reply = handle_command(chat_id, text) if text.startswith("/") else None
            if reply is None:
                reply = get_ai_response(text)
            if reply:
                send(chat_id, reply)
        except Exception as e:
            print("webhook error:", e)
    return "OK", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)