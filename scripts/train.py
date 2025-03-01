"""
This script handles the model training process for the House Pricing Predictor project.

It reads the training data, trains the model and store it as picked objects.

Modules
-------
- model_training: Functions for training the model using the training dataset.

Usage
-----
python scripts/train.py /path/to/processed_data /path/to/trained_models --log-level 
DEBUG --log-path ./logs/score.log --no-console-log

"""

import argparse
import logging
import os
import pickle
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from house_pricing_predictor_YUKTHAMAJELLA.logging_config import setup_logging
from house_pricing_predictor_YUKTHAMAJELLA.model_scoring import model_scoring
from house_pricing_predictor_YUKTHAMAJELLA.model_training import (
    get_best_model_gridsearch,
    model_training,
)

import mlflow



def train_model(input_data_path, output_path):
    """
    Reads the training data, trains the models and store them as pickled objects.

    Parameters
    ----------
    input_data_path : str
        The directory path of the processed data.

    output_path : str
        The directory path to store the pickled model.

    Returns
    -------
    None
    run_id : str
        The MLflow run ID of the trained model.
    """
    with mlflow.start_run(run_name="model_training", nested=True) as run:
        housing_prepared = pd.read_pickle(f'{input_data_path}/housing_prepared.pkl')
        housing_labels = pd.read_pickle(f'{input_data_path}/housing_labels.pkl')

        lin_reg, tree_reg, rnd_search, grid_search = model_training(
            housing_prepared, housing_labels
        )

        final_model = get_best_model_gridsearch(grid_search, housing_prepared)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(f'{output_path}/final_model.pkl', 'wb') as f:
            pickle.dump(final_model, f)

        print(f"Model saved to {output_path}")

        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)

        mlflow.sklearn.log_model(final_model, "gridsearch_model")

        score = grid_search.best_score_
        mlflow.log_metric("best_score", score)
        print(f"Artifacts saved at: {mlflow.get_artifact_uri()}")
        return run.info.run_id



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using prepared data.")
    parser.add_argument("input_data_path", help="Path to load the processed data.")
    parser.add_argument("output_path", help="Path to store the trained models.")
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
    logger.info("Starting model training...")

    try:
        logger.debug("Loading processed data and training the model...")
        train_model(args.input_data_path, args.output_path)
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
