"""
model_training module contains the function for the model training process for the House 
Pricing Predictor project.

It contains functions to load the training data, train the models and stores it as a
pickled object.

"""

import logging

import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)


def cv_results(model):
    """
    Prints the cross validation results of the t5rained model.

    Parameters
    ----------
    model : object
        The trained machine learning model to be used for making predictions.

    Returns
    -------
    None
        This function doesn't return any value. It prints the model's cross validation
        results.

    """
    cvres = model.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)


def model_training(housing_prepared, housing_labels):
    """
    Trains multiple machine learning models using the provided training data and labels.

    Parameters
    ----------
    housing_prepared : pandas.DataFrame
        The preprocessed training data used to train the models.

    housing_labels : pandas.Series
        The target labels corresponding to the `housing_prepared` dataset.

    Returns
    -------
    tuple
        A tuple containing the following trained models:
        - LinearRegression model
        - DecisionTreeRegressor model
        - RandomizedSearchCV (RandomForestRegressor)
        - GridSearchCV (RandomForestRegressor)

    """
    try:
        logger.info("Training the models...")

        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)

        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(housing_prepared, housing_labels)

        param_distribs = {
            "n_estimators": randint(low=1, high=200),
            "max_features": randint(low=1, high=8),
        }
        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        rnd_search.fit(housing_prepared, housing_labels)
        cv_results(rnd_search)

        param_grid = [
            {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
            {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
        ]
        forest_reg = RandomForestRegressor(random_state=42)

        grid_search = GridSearchCV(
            forest_reg,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
        grid_search.fit(housing_prepared, housing_labels)
        cv_results(grid_search)

        logger.debug("Model training is completed successfully.")
    except Exception as e:
        logger.error(f"Error while model training process: {e}")
    return lin_reg, tree_reg, rnd_search, grid_search


def get_best_model_gridsearch(grid_search, housing_prepared):
    """
    Identifies and returns the best model from the grid search based on cross-validation
    results.

    Parameters
    ----------
    grid_search : sklearn.model_selection.GridSearchCV
        The GridSearchCV object containing the results of the grid search over multiple
        hyperparameters.

    housing_prepared : pandas.DataFrame
        The preprocessed training data.

    Returns
    -------
    sklearn.base.BaseEstimator
        The best estimator (model) selected from the grid search.
    """

    try:
        logger.info("Finging the best models...")
        feature_importances = grid_search.best_estimator_.feature_importances_
        sorted(zip(feature_importances, housing_prepared.columns), reverse=True)
        logger.debug("Best model is selected successfully.")
    except Exception as e:
        logger.error(f"Error while best model selection process: {e}")
    return grid_search.best_estimator_
