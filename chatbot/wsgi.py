"""
WSGI entry point for production servers (gunicorn, Render, Railway, etc.).

Usage:
    gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
"""

from dotenv import load_dotenv

load_dotenv()

from app import app, bootstrap_application  # noqa: E402

bootstrap_application()
