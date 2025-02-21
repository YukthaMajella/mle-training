import argparse
import os
import pickle
import sys

import pandas as pd
import logging
from house_pricing_predictor.logging_config import setup_logging
from house_pricing_predictor.model_scoring import model_scoring
from house_pricing_predictor.model_training import (
    get_best_model_gridsearch,
    model_training,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


# python scripts/ingest_data.py /home/mle-training/processed_data /home/mle-training/models --log-level DEBUG --log-path ./logs/train.log --no-console-log
def train_model(input_data_path, output_path):

    housing_prepared = pd.read_pickle(f'{input_data_path}/housing_prepared.pkl')
    housing_labels = pd.read_pickle(f'{input_data_path}/housing_labels.pkl')

    lin_reg, tree_reg, rnd_search, grid_search = model_training(
        housing_prepared, housing_labels
    )

    '''with open(f'{output_path}/models/lin_reg_model.pkl', 'wb') as f:
        pickle.dump(lin_reg, f)

    with open(f'{output_path}/models/tree_reg_model.pkl', 'wb') as f:
        pickle.dump(tree_reg, f)

    with open(f'{output_path}/models/rnd_search_model.pkl', 'wb') as f:
        pickle.dump(rnd_search, f)

    with open(f'{output_path}/models/grid_search_model.pkl', 'wb') as f:
        pickle.dump(grid_search, f)'''

    final_model = get_best_model_gridsearch(grid_search, housing_prepared)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(f'{output_path}/final_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)

    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using prepared data.")
    parser.add_argument("input_data_path", help="Path to load the processed data.")
    parser.add_argument("output_path", help="Path to store the trained models.")
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (default: INFO)')
    parser.add_argument('--log-path', type=str, default=None, help='Path to log file (default: None)')
    parser.add_argument('--no-console-log', action='store_true', help='Disable console logging (default: True)')

    args = parser.parse_args()

    setup_logging(log_level=args.log_level, log_path=args.log_path, console_log=not args.no_console_log)

    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")

    try:
        logger.debug("Loading processed data and training the model...")
        train_model(args.input_data_path, args.output_path)
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Error during training: {e}")

