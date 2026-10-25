"""
Portable Cross-Platform Launcher for Diabetes Risk Prediction Application.

Usage:
    python run_app.py
"""

import os
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    # Check for models
    model_path = project_root / "models" / "best_model.joblib"
    if not model_path.exists():
        print("[-] Model artifacts not detected in models/ directory.")
        print("[*] Running end-to-end training pipeline first...")
        from src.train import run_training_pipeline
        run_training_pipeline()

    print("[+] Launching Streamlit Clinical Dashboard on http://localhost:8501 ...")
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", "app/main.py", "--server.port=8501"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
