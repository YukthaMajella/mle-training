"""
This module contains the function to test the scoring module of the House Pricing 
Predictor project.

"""

import pandas as pd
import pytest

from house_pricing_predictor_YUKTHAMAJELLA.model_scoring import model_scoring
from house_pricing_predictor_YUKTHAMAJELLA.model_training import model_training


def test_model_training():
    eg_tr_data = {
        'longitude': [-121.46, -117.23, -118.25, -120.67, -122.81],
        'latitude': [38.52, 33.09, 36.77, 37.36, 34.06],
        'housing_median_age': [29.0, 7.0, 15.0, 25.0, 10.0],
        'total_rooms': [3873.0, 5320.0, 2105.0, 4572.0, 3811.0],
        'total_bedrooms': [797.0, 855.0, 410.0, 806.0, 715.0],
        'population': [2237.0, 2015.0, 1532.0, 3195.0, 2685.0],
        'households': [706.0, 768.0, 525.0, 987.0, 702.0],
        'median_income': [2.1736, 6.3373, 3.5065, 4.8902, 5.6509],
        'rooms_per_household': [5.485836, 6.927083, 4.0, 4.635, 5.43],
        'bedrooms_per_room': [0.205784, 0.160714, 0.195, 0.176, 0.187],
        'population_per_household': [3.168555, 2.623698, 2.92, 3.24, 3.83],
        'ocean_proximity_INLAND': [True, False, False, True, False],
        'ocean_proximity_ISLAND': [False, False, True, False, False],
        'ocean_proximity_NEAR BAY': [False, False, False, True, False],
        'ocean_proximity_NEAR OCEAN': [False, True, False, False, True],
    }

    df_tr = pd.DataFrame(eg_tr_data)
    assert df_tr.shape[0] == 5

    eg_tr_labels = pd.Series(
        [72100.0, 279600.0, 185000.0, 235000.0, 220500.0],
        index=[12655, 15502, 12656, 15503, 12657],
    )
    assert eg_tr_labels.shape[0] == 5

    lin_reg, tree_reg, rnd_search, grid_search = model_training(df_tr, eg_tr_labels)

    assert lin_reg is not None
    assert tree_reg is not None
    assert rnd_search is not None
    assert grid_search is not None

    eg_test_data = {
        'longitude': [-121.46],
        'latitude': [38.52],
        'housing_median_age': [29.0],
        'total_rooms': [3873.0],
        'total_bedrooms': [797.0],
        'population': [2237.0],
        'households': [706.0],
        'median_income': [2.1736],
        'rooms_per_household': [5.485836],
        'bedrooms_per_room': [0.205784],
        'population_per_household': [3.168555],
        'ocean_proximity_INLAND': [True],
        'ocean_proximity_ISLAND': [False],
        'ocean_proximity_NEAR BAY': [False],
        'ocean_proximity_NEAR OCEAN': [False],
    }

    df_te = pd.DataFrame(eg_test_data)

    y_test_labels = pd.Series([72100.0], index=[12659])

    assert df_te.shape[0] == 1
    assert y_test_labels.shape[0] == 1

    housing_predictions, mse, rmse, mae = model_scoring(lin_reg, df_te, y_test_labels)
    assert housing_predictions is not None
    assert mse is not None
    assert rmse is not None
    assert mae is not None
