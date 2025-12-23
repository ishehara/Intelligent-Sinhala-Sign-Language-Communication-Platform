"""
Quick start script for training Sinhala Sign Language Recognition model.
Run this to train a model with default settings.

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import subprocess
import sys
import os
from pathlib import Path

def find_venv_python():
    """Find the Python executable in the virtual environment."""
    current_dir = Path.cwd()
    project_root = current_dir.parent.parent.parent  # Go up from src -> ssl-reader -> components -> project root
    
    # Check for venv in project root
    venv_paths = [
        project_root / "venv" / "Scripts" / "python.exe",  # Windows
        project_root / "venv" / "bin" / "python",  # Linux/Mac
        project_root / ".venv" / "Scripts" / "python.exe",  # Windows alternative
        project_root / ".venv" / "bin" / "python",  # Linux/Mac alternative
    ]
    
    for venv_python in venv_paths:
        if venv_python.exists():
            return str(venv_python)
    
    # Fallback to current Python (might be in venv already)
    return sys.executable

def main():
    print("=" * 70)
    print("Sinhala Sign Language Recognition - Quick Start Training")
    print("=" * 70)
    print()
    
    # Check if running from correct directory
    current_dir = Path.cwd()
    if current_dir.name != 'src':
        print("⚠️  Please run this script from the 'src' directory")
        print(f"   Current directory: {current_dir}")
        sys.exit(1)
    
    # Configuration - use absolute paths to avoid path resolution issues
    project_root = current_dir.parent.parent.parent  # Go up from src -> ssl-reader -> components -> project root
    dataset_root = str(project_root / "datasets" / "signVideo")
    cache_dir = "../data/processed"
    
    # Find the correct Python executable
    python_exe = find_venv_python()
    print(f"Using Python: {python_exe}")
    
    print("\nConfiguration:")
    print(f"  Dataset: {dataset_root}")
    print(f"  Cache: {cache_dir}")
    print()
    
    # Ask user for model type
    print("Select model type:")
    print("  1. LSTM (faster, good for quick testing)")
    print("  2. Transformer (better accuracy, slower)")
    print("  3. Hybrid (LSTM + Transformer, best results)")
    
    choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip() or "1"
    
    model_map = {
        "1": "lstm",
        "2": "transformer",
        "3": "hybrid"
    }
    
    model_type = model_map.get(choice, "lstm")
    print(f"\n✓ Selected model: {model_type}")
    
    # Ask about preprocessing
    preprocess = input("\nPreprocess all videos now? (y/n) [default: n]: ").strip().lower() == 'y'
    
    if preprocess:
        print("\n✓ Will preprocess videos (this may take a while)")
    else:
        print("\n✓ Will process videos on-the-fly during training")
    
    # Build command
    cmd = [
        python_exe,
        "train.py",
        "--dataset_root", dataset_root,
        "--cache_dir", cache_dir,
        "--model_type", model_type,
        "--batch_size", "16",
        "--num_epochs", "50",
        "--learning_rate", "0.001",
        "--patience", "10",
        "--max_frames", "60",
        "--hidden_dim", "256",
        "--num_layers", "2",
        "--dropout", "0.3",
        "--num_workers", "4"
    ]
    
    if preprocess:
        cmd.append("--preprocess")
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    print()
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 70)
        print("✓ Training completed successfully!")
        print("=" * 70)
        print("\nTrained model saved in: ../models/")
        print("Training logs saved in: ../logs/")
        print("\nTo test your model, run:")
        print(f"  python inference.py --model_path ../models/checkpoint_best.pth --mode webcam")
        print()
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 70)
        print("✗ Training failed!")
        print("=" * 70)
        print(f"\nError: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Training interrupted by user")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
