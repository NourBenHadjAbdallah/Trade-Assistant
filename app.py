import os
from flask import Flask

app = Flask(__name__)

@app.route("/")
@app.route("/ping")
def health_check():
    return "Trade Assistant Bot is alive!", 200

if __name__ == "__main__":
    # Render automatically assigns a PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)