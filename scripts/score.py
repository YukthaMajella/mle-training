"""
This script handles the model scoring process for the House Pricing Predictor project.

It reads the data to be scored and predicts the prices for it using the trained models.

Modules
-------
- model_scoring: Functions for predicting the target variable for the data.

Usage
-----
python scripts/score.py /path/to/processed_data /path/to/trained_models --log-level
DEBUG --log-path ./logs/score.log --no-console-log

"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from house_pricing_predictor_YUKTHAMAJELLA.logging_config import setup_logging
from house_pricing_predictor_YUKTHAMAJELLA.model_scoring import model_scoring

import mlflow




def score_model(data_path, input_model_path):
    """
    Reads the scoring data, predicts for the data by loading the pickled models and
    stores the predictions as a pickled file.

    Parameters
    ----------
    data_path : str
        The directory path of the processed data.

    input_model_path : str
        The directory path of the pickled model.

    Returns
    -------
    None
        This function doesn't return any value. It saves the predictions for the scored
        data at the specified output path.
    """
    with mlflow.start_run(run_name="model_scoring", nested=True) as run:
        X_test_prepared = pd.read_pickle(f'{data_path}/X_test_prepared.pkl')
        y_test = pd.read_pickle(f'{data_path}/y_test.pkl')
        with open(f'{input_model_path}/final_model.pkl', 'rb') as m:
            final_model = pickle.load(m)

        housing_predictions, final_mse, final_rmse, final_mae = model_scoring(
            final_model, X_test_prepared, y_test
        )
        np.save(f'{data_path}/housing_test_predictions.npy', housing_predictions)
        print(f"Output predictions saved to {data_path}")

        mlflow.log_metric("mse", final_mse)
        mlflow.log_metric("rmse", final_rmse)
        mlflow.log_metric("mae", final_mae)
        mlflow.log_artifact(f'{data_path}/housing_test_predictions.npy')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and preprocess data.")
    parser.add_argument("input_data_path", help="Path to load the processed test data.")
    parser.add_argument("input_model_path", help="Path to load the trained models.")
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level (default: INFO)',
    )
    parser.add_argument(
        '--log-path', type=str, default=None, help='Path to log file (default: None)'
    )
    parser.add_argument(
        '--no-console-log',
        action='store_true',
        help='Disable console logging (default: True)',
    )

    args = parser.parse_args()

    setup_logging(
        log_level=args.log_level,
        log_path=args.log_path,
        console_log=not args.no_console_log,
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting model scoring...")

    try:
        logger.debug("Loading model and scoring data...")
        score_model(args.input_data_path, args.input_model_path)
        logger.info("Scoring completed successfully.")
    except Exception as e:
        logger.error(f"Error during scoring: {e}")
