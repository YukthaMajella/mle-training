import argparse
import os
import sys

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from house_pricing_predictor.data_ingestion import (
    clean_strat_data,
    fetch_housing_data,
    load_housing_data,
    prepare_test_data,
    prepare_train_data,
    proportions_comparison,
    stratified_split,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


# python scripts/ingest_data.py /home/mle-training/processed_data


def ingest_data(output_path):
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = os.path.join("datasets", "housing")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and preprocess data.")
    parser.add_argument("output_path", help="Path to save the processed data.")
    args = parser.parse_args()

    ingest_data(args.output_path)
