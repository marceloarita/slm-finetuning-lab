"""
Day 01 — Environment validation script.
Run this to confirm all dependencies are correctly installed
and that Apple Silicon (MPS) is available for training.
"""

import sys


def check_python():
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 11
    status = "✓" if ok else "✗"
    print(f"{status} Python {version.major}.{version.minor}.{version.micro}")
    if not ok:
        print("  → Python 3.11+ required")
    return ok


def check_torch():
    try:
        import torch
        mps_available = torch.backends.mps.is_available()
        print(f"✓ PyTorch {torch.__version__}")
        if mps_available:
            print("  → MPS (Apple Silicon) available ✓")
        else:
            print("  → MPS not available, will use CPU")
        return True
    except ImportError:
        print("✗ PyTorch not found — run: uv add torch")
        return False


def check_transformers():
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        return True
    except ImportError:
        print("✗ Transformers not found — run: uv add transformers")
        return False


def check_datasets():
    try:
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
        return True
    except ImportError:
        print("✗ Datasets not found — run: uv add datasets")
        return False


def check_mlx():
    try:
        import mlx.core as mx
        print(f"✓ MLX {mx.__version__}")
        return True
    except ImportError:
        print("✗ MLX not found — run: uv add mlx")
        return False


def main():
    print("=" * 40)
    print("SLM Fine-Tuning Lab — Environment Check")
    print("=" * 40)

    results = [
        check_python(),
        check_torch(),
        check_transformers(),
        check_datasets(),
        check_mlx(),
    ]

    print("=" * 40)
    if all(results):
        print("All checks passed. Ready to start Day 1!")
    else:
        failed = results.count(False)
        print(f"{failed} check(s) failed. Fix the issues above before continuing.")
    print("=" * 40)


if __name__ == "__main__":
    main()