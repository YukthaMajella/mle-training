import subprocess
import sys

import pytest


def test_package_installation():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "./dist/HousePricingPredictor-0.0.1-py3-none-any.whl",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Installation failed: {result.stderr}"

    try:
        import HousePricingPredictor
    except ImportError as e:
        pytest.fail(f"Package import failed: {str(e)}")

    try:
        from HousePricingPredictor import main
    except ImportError as e:
        pytest.fail(f"Failed to import main module: {str(e)}")

    try:
        from HousePricingPredictor import DataIngestion
    except ImportError as e:
        pytest.fail(f"Failed to import DataIngestion module: {str(e)}")

    try:
        from HousePricingPredictor import ModelTraining
    except ImportError as e:
        pytest.fail(f"Failed to import ModelTraining module: {str(e)}")

    try:
        from HousePricingPredictor import ModelScoring
    except ImportError as e:
        pytest.fail(f"Failed to import ModelScoring module: {str(e)}")

    try:
        from HousePricingPredictor.DataIngestion import load_housing_data

        assert callable(load_housing_data)
    except ImportError as e:
        pytest.fail(
            f"Failed to import load_housing_data function from DataIngestion module: {str(e)}"
        )

    try:
        from HousePricingPredictor.ModelTraining import model_training

        assert callable(model_training)
    except ImportError as e:
        pytest.fail(
            f"Failed to import model_training function from ModelTraining module: {str(e)}"
        )

    try:
        from HousePricingPredictor.ModelScoring import model_scoring

        assert callable(model_training)
    except ImportError as e:
        pytest.fail(
            f"Failed to import model_scoring function from ModelScoring module: {str(e)}"
        )
