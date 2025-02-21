import os
import tarfile

import numpy as np
import pandas as pd
import logging
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

def fetch_housing_data(housing_url, housing_path):
    try:
        logger.info("Extracting data from source...")
        os.makedirs(housing_path, exist_ok=True)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
        logger.debug("Data extracted successfully.")
    except Exception as e:
        logger.error(f"Error while extracting data: {e}")

def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def stratified_split(housing):
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    return strat_train_set, strat_test_set


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


def proportions_comparison(housing, strat_test_set, test_set):
    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )


def add_features(housing):
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    return housing


def data_manipulation(strat_set):
    housing = strat_set.drop("median_house_value", axis=1)
    housing_labels = strat_set["median_house_value"].copy()
    housing_num = housing.drop("ocean_proximity", axis=1)
    return housing, housing_labels, housing_num


def prepare_dataframe(X, housing_num, housing):
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr = add_features(housing_tr)
    housing_cat = housing[["ocean_proximity"]]
    return housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))


def clean_strat_data(strat_train_set, strat_test_set):
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()

    corr_matrix = housing.corr(numeric_only=True)
    corr_matrix["median_house_value"].sort_values(ascending=False)

    housing = add_features(housing)

    housing, housing_labels, housing_num = data_manipulation(strat_train_set)
    return housing, housing_labels, housing_num


def prepare_train_data(imputer, housing, housing_num):
    try:
        logger.info("Preparing training dataframe...")
        X = imputer.transform(housing_num)
        housing_prepared = prepare_dataframe(X, housing_num, housing)
        logger.debug("Training data is prepared successfully.")
    except Exception as e:
        logger.error(f"Error while preparing training data: {e}")
    return housing_prepared


def prepare_test_data(strat_test_set, imputer):
    try:
        logger.info("Preparing test dataframe...")
        X_test, y_test, X_test_num = data_manipulation(strat_test_set)
        X_test_prepared = imputer.transform(X_test_num)
        X_test_prepared = prepare_dataframe(X_test_prepared, X_test_num, X_test)
        logger.debug("Test data is prepared successfully.")
    except Exception as e:
        logger.error(f"Error while preparing test data: {e}")
    return X_test_prepared, y_test
