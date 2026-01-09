#!/usr/bin/env python3
"""
GEPA Prompt Evolution - Start script
Run this to start the web server
"""

import subprocess
import sys
import os

def main():
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    print("=" * 50)
    print("  GEPA Prompt Evolution")
    print("=" * 50)
    print()
    print("Starting server at http://localhost:8000")
    print("Press Ctrl+C to stop")
    print()

    # Run uvicorn
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.app:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

if __name__ == "__main__":
    main()
