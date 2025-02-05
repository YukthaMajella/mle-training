import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def model_scoring(model, housing_prepared, housing_labels):
    housing_predictions = model.predict(housing_prepared)
    mse = mean_squared_error(housing_labels, housing_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(housing_labels, housing_predictions)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    return mse, rmse, mae
