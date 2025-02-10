import argparse
import os
import pickle
import sys

import pandas as pd

from house_pricing_predictor.model_scoring import model_scoring
from house_pricing_predictor.model_training import (
    get_best_model_gridsearch,
    model_training,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


# python scripts/ingest_data.py /home/mle-training/processed_data /home/mle-training/models
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

    args = parser.parse_args()

    train_model(args.input_data_path, args.output_path)
