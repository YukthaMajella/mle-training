"""
This module contains the function to test the installation and module imports of the 
House Pricing Predictor project.

"""

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
            "./dist/house_pricing_predictor_YUKTHAMAJELLA-0.0.1-py3-none-any.whl",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Installation failed: {result.stderr}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "./dist/house_pricing_predictor_yukthamajella-0.0.1.tar.gz",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Installation failed: {result.stderr}"

    try:
        import house_pricing_predictor_YUKTHAMAJELLA
    except ImportError as e:
        pytest.fail(f"Package import failed: {str(e)}")

    try:
        from house_pricing_predictor_YUKTHAMAJELLA import data_ingestion
    except ImportError as e:
        pytest.fail(f"Failed to import DataIngestion module: {str(e)}")

    try:
        from house_pricing_predictor_YUKTHAMAJELLA import model_training
    except ImportError as e:
        pytest.fail(f"Failed to import ModelTraining module: {str(e)}")

    try:
        from house_pricing_predictor_YUKTHAMAJELLA import model_scoring
    except ImportError as e:
        pytest.fail(f"Failed to import ModelScoring module: {str(e)}")

    try:
        from house_pricing_predictor_YUKTHAMAJELLA.data_ingestion import load_housing_data

        assert callable(load_housing_data)
    except ImportError as e:
        pytest.fail(
            f"Failed to import load_housing_data function from DataIngestion module: {str(e)}"
        )

    try:
        from house_pricing_predictor_YUKTHAMAJELLA.model_training import model_training

        assert callable(model_training)
    except ImportError as e:
        pytest.fail(
            f"Failed to import model_training function from ModelTraining module: {str(e)}"
        )

    try:
        from house_pricing_predictor_YUKTHAMAJELLA.model_scoring import model_scoring

        assert callable(model_training)
    except ImportError as e:
        pytest.fail(
            f"Failed to import model_scoring function from ModelScoring module: {str(e)}"
        )
