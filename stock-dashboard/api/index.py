"""
Vercel serverless entry point for the Flask app.
"""
import sys
import os

# Add parent directory to path so we can import app
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Set working directory for templates/static
os.chdir(parent_dir)

from app import app

# Vercel expects 'app' or 'application'
application = app
