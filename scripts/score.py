import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import logging
from house_pricing_predictor.logging_config import setup_logging
from house_pricing_predictor.model_scoring import model_scoring

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


# python scripts/score.py /home/mle-training/processed_data /home/mle-training/models --log-level DEBUG --log-path ./logs/score.log --no-console-log


def score_model(data_path, input_model_path):
    X_test_prepared = pd.read_pickle(f'{data_path}/X_test_prepared.pkl')
    y_test = pd.read_pickle(f'{data_path}/y_test.pkl')
    with open(f'{input_model_path}/final_model.pkl', 'rb') as m:
        final_model = pickle.load(m)

    housing_predictions, final_mse, final_rmse, final_mae = model_scoring(
        final_model, X_test_prepared, y_test
    )
    np.save(f'{data_path}/housing_test_predictions.npy', housing_predictions)
    print(f"Output predictions saved to {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and preprocess data.")
    parser.add_argument("input_data_path", help="Path to load the processed test data.")
    parser.add_argument("input_model_path", help="Path to load the trained models.")
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (default: INFO)')
    parser.add_argument('--log-path', type=str, default=None, help='Path to log file (default: None)')
    parser.add_argument('--no-console-log', action='store_true', help='Disable console logging (default: True)')

    args = parser.parse_args()

    setup_logging(log_level=args.log_level, log_path=args.log_path, console_log=not args.no_console_log)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting model scoring...")
    
    try:
        logger.debug("Loading model and scoring data...")
        score_model(args.input_data_path, args.input_model_path)
        logger.info("Scoring completed successfully.")
    except Exception as e:
        logger.error(f"Error during scoring: {e}")
