import argparse
import os
import pickle
import sys

import pandas as pd

from house_pricing_predictor.model_scoring import model_scoring

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


# python scripts/score.py /home/mle-training/processed_data /home/mle-training/models


def score_model(data_path, input_model_path):
    X_test_prepared = pd.read_pickle(f'{data_path}/X_test_prepared.pkl')
    y_test = pd.read_pickle(f'{data_path}/y_test.pkl')
    with open(f'{input_model_path}/final_model.pkl', 'rb') as m:
        final_model = pickle.load(m)

    housing_predictions, final_mse, final_rmse, final_mae = model_scoring(
        final_model, X_test_prepared, y_test
    )
    housing_predictions.to_pickle(f'{data_path}/housing_test_predictions.pkl')
    print(f"Output predictions saved to {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and preprocess data.")
    parser.add_argument("input_data_path", help="Path to load the processed test data.")
    parser.add_argument("input_model_path", help="Path to load the trained models.")
    args = parser.parse_args()

    score_model(args.input_data_path, args.input_model_path)
