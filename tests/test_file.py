import subprocess
import sys
import pytest


def test_package_installation():
    result = subprocess.run([sys.executable, "-m", "pip", "install", "./dist/*.whl"], 
                            capture_output=True, text=True)

    assert result.returncode == 0, f"Installation failed: {result.stderr}"

    try:
        from HousePricingPredictor import nonstandardcode
    except ImportError as e:
        pytest.fail(f"Package import failed: {str(e)}")

