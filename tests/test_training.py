"""
This module contains the function to test the training module of the House Pricing
Predictor project.

"""

import os
import sys

import pandas as pd
import pytest

from src.house_pricing_predictor.model_training import model_training

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


def test_model_training():
    eg_data = {
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

    df = pd.DataFrame(eg_data)
    assert df.shape[0] == 5

    eg_labels = pd.Series(
        [72100.0, 279600.0, 185000.0, 235000.0, 220500.0],
        index=[12655, 15502, 12656, 15503, 12657],
    )
    assert eg_labels.shape[0] == 5

    lin_reg, tree_reg, rnd_search, grid_search = model_training(df, eg_labels)

    assert lin_reg is not None
    assert tree_reg is not None
    assert rnd_search is not None
    assert grid_search is not None
