#!/usr/bin/env python
"""
Pre-deployment optimization script for Render.
This helps prepare your application for cloud deployment.
"""

import os
import sys
from pathlib import Path


def check_environment():
    """Check if running in Render environment."""
    is_render = os.getenv("RENDER") is not None
    print(f"Running on Render: {is_render}")
    if is_render:
        print("✓ Render environment detected")
        print(f"  - Render Service: {os.getenv('RENDER_SERVICE_NAME', 'N/A')}")
        print(f"  - Git Commit: {os.getenv('RENDER_GIT_COMMIT', 'N/A')[:8]}")
    else:
        print("ℹ Not running on Render (local/other environment)")
    return is_render


def check_directories():
    """Ensure necessary directories exist."""
    dirs = [
        ".streamlit",
        "outputs",
        "/tmp/ocr_outputs" if os.path.exists("/tmp") else "outputs"
    ]
    
    for dir_path in dirs:
        if dir_path.startswith("/tmp"):
            continue  # Skip /tmp checks
        
        Path(dir_path).mkdir(exist_ok=True, parents=True)
        print(f"✓ Directory exists: {dir_path}")


def check_dependencies():
    """Verify critical dependencies are installed."""
    critical_modules = [
        "streamlit",
        "torch",
        "transformers",
        "easyocr",
        "pypdfium2",
    ]
    
    missing = []
    for module in critical_modules:
        try:
            __import__(module)
            print(f"✓ {module} installed")
        except ImportError:
            print(f"✗ {module} NOT installed")
            missing.append(module)
    
    return missing


def main():
    """Run all checks."""
    print("=" * 60)
    print("OCR Pipeline - Render Deployment Check")
    print("=" * 60)
    
    print("\n[1/3] Checking environment...")
    is_render = check_environment()
    
    print("\n[2/3] Checking directories...")
    check_directories()
    
    print("\n[3/3] Checking dependencies...")
    missing = check_dependencies()
    
    print("\n" + "=" * 60)
    if missing:
        print(f"⚠ Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return 1
    else:
        print("✓ All checks passed! Ready for deployment.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
