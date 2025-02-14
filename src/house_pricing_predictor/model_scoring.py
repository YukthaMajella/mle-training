"""
This module contains the function for the scoring process of the House Pricing Predictor
project.

It contains functions to read the data, load the pickled models and score the data.
"""

import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def model_scoring(model, housing_prepared, housing_labels):
    """
    Reads the scoring data and predicts the target variable for the data by loading the
    pickled models.

    Parameters
    ----------
    model : object
        The trained machine learning model to be used for making predictions.

    housing_prepared : pandas.DataFrame
        The input features of the housing data that have been preprocessed and are ready
        for prediction.

    housing_labels : pandas.Series
        The true target values of the `housing_prepared` data.

    Returns
    -------
    housing_predictions : numpy.ndarray
        The predicted target values for the input data, produced by the model.

    mse : float
        The Mean Squared Error (MSE) between the predicted and actual values.

    rmse : float
        The Root Mean Squared Error (RMSE), which is the square root of MSE.

    mae : float
        The Mean Absolute Error (MAE) between the predicted and actual values.

    """

    try:
        logger.info("Predicting for test data...")
        housing_predictions = model.predict(housing_prepared)
        mse = mean_squared_error(housing_labels, housing_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(housing_labels, housing_predictions)
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("MAE:", mae)
        logger.debug("Model scored for test data successfully.")
    except Exception as e:
        logger.error(f"Error while scoring test data: {e}")
    return housing_predictions, mse, rmse, mae
