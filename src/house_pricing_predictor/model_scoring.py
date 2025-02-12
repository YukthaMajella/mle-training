import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
logger = logging.getLogger(__name__)

def model_scoring(model, housing_prepared, housing_labels):
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

