import os
import requests
from flask import Flask, request

app = Flask(__name__)

# Fetch Telegram token from environment variables
TELEGRAM_TOKEN = os.environ.get("8612493118:AAFhiqbRckTrGrVg_cLW3NZFw3a5JPQwrMU")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"


def get_ai_response(user_message):
    """
    Replace this function logic with your actual AI chat handler/response code.
    """
    return f"AI Assistant Received: {user_message}"


@app.route("/")
@app.route("/ping")
def health_check():
    return "Trade Assistant Bot is alive!", 200


@app.route("/webhook", methods=["POST"])
def telegram_webhook():
    data = request.get_json()

    if data and "message" in data:
        chat_id = data["message"]["chat"]["id"]
        incoming_text = data["message"]["text"]

        # Process message with AI logic
        ai_reply = get_ai_response(incoming_text)

        # Send response back to Telegram user
        payload = {"chat_id": chat_id, "text": ai_reply}
        requests.post(TELEGRAM_API_URL, json=payload)

    return "OK", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)