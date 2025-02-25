"""
This script handles the data ingestion process for the House Pricing Predictor project.

It reads input data from various sources, processes it, and stores it in a format that
can be used for model training.

Modules
-------
- data_ingestion: Functions for reading and processing data files.

Usage
-----
python scripts/ingest_data.py /path/to/processed_data --log-level DEBUG --log-path
./logs/ingest_data.log --no-console-log

"""

import argparse
import logging
import os
import sys

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from house_pricing_predictor_YUKTHAMAJELLA.data_ingestion import (
    clean_strat_data,
    fetch_housing_data,
    load_housing_data,
    prepare_test_data,
    prepare_train_data,
    proportions_comparison,
    stratified_split,
)
from house_pricing_predictor_YUKTHAMAJELLA.logging_config import setup_logging

import mlflow
import mlflow.pyfunc




def ingest_data(output_path):
    """
    Ingests raw housing data, preprocesses it, and saves the processed datasets.

    Parameters
    ----------
    output_path : str
        The directory path where the processed data will be saved.

    Returns
    -------
    None
        This function doesn't return any value. It saves the processed data as pickle
        files at the specified output path.
    """

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = os.path.join("datasets", "housing")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    with mlflow.start_run(run_name="data_ingestion", nested=True) as run:

        fetch_housing_data(HOUSING_URL, HOUSING_PATH)
        housing = load_housing_data(HOUSING_PATH)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        strat_train_set, strat_test_set = stratified_split(housing)
        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
        proportions_comparison(housing, strat_test_set, test_set)
        housing, housing_labels, housing_num = clean_strat_data(
            strat_train_set, strat_test_set
        )

        imputer = SimpleImputer(strategy="median")
        imputer.fit(housing_num)
        housing_prepared = prepare_train_data(imputer, housing, housing_num)
        X_test_prepared, y_test = prepare_test_data(strat_test_set, imputer)

        housing_prepared.to_pickle(f'{output_path}/housing_prepared.pkl')
        housing_labels.to_pickle(f'{output_path}/housing_labels.pkl')
        X_test_prepared.to_pickle(f'{output_path}/X_test_prepared.pkl')
        y_test.to_pickle(f'{output_path}/y_test.pkl')
        print(f"Data saved to {output_path}")

        mlflow.log_artifact(f'{output_path}/housing_prepared.pkl')
        mlflow.log_artifact(f'{output_path}/housing_labels.pkl')
        mlflow.log_artifact(f'{output_path}/X_test_prepared.pkl')
        mlflow.log_artifact(f'{output_path}/y_test.pkl')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and preprocess data.")
    parser.add_argument("output_path", help="Path to save the processed data.")
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
    logger.info("Starting data ingestion process...")
    try:
        logger.debug("Ingesting raw data...")
        ingest_data(args.output_path)
        logger.info("Data ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
