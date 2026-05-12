import sys
import os

# Make the backend package importable from this serverless function
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from app.main import app  # noqa: F401  — Vercel detects the ASGI `app` object
