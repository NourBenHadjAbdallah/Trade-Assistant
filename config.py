"""
Loads secrets and settings from a local .env file (next to this file) into os.environ.

Import this before reading os.environ. Real environment variables (e.g. Render's dashboard)
always win over .env, so the same code works locally and in deployment.
Copy .env.example to .env and fill in your own values. Never commit .env.
"""
import os

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH, override=False)
except ImportError:
    pass
