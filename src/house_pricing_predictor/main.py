import os

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
from house_pricing_predictor.model_scoring import model_scoring
from house_pricing_predictor.model_training import (
    get_best_model_gridsearch,
    model_training,
)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def main():
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    housing = load_housing_data(HOUSING_PATH)
    # train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    strat_train_set, strat_test_set = stratified_split(housing)
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    proportions_comparison(housing, strat_test_set, test_set)
    housing, housing_labels, housing_num = clean_strat_data(
        strat_train_set, strat_test_set
    )

    imputer = SimpleImputer(strategy="median")
    imputer.fit(housing_num)
    housing_prepared = prepare_train_data(imputer, housing, housing_num)

    lin_reg, tree_reg, rnd_search, grid_search = model_training(
        housing_prepared, housing_labels
    )

    # print("Linear Regression")
    lin_mse, lin_rmse, lin_mae = model_scoring(
        lin_reg, housing_prepared, housing_labels
    )

    # print("Decision Tree Regressor")
    tree_mse, tree_rmse, tree_mae = model_scoring(
        tree_reg, housing_prepared, housing_labels
    )

    # print("GridSearch Best Model")
    final_model = get_best_model_gridsearch(grid_search, housing_prepared)

    X_test_prepared, y_test = prepare_test_data(strat_test_set, imputer)

    final_mse, final_rmse, final_mae = model_scoring(
        final_model, X_test_prepared, y_test
    )


if __name__ == "__main__":
    main()
