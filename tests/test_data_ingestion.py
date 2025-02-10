import pandas as pd
import pytest

from src.house_pricing_predictor.data_ingestion import add_features


def test_loading_data():
    eg_data = {
        'longitude': [-122.23],
        'latitude': [37.88],
        'housing_median_age': [25],
        'total_rooms': [880],
        'total_bedrooms': [129],
        'population': [322],
        'households': [126],
        'median_income': [8.3252],
        'median_house_value': [452600],
        'ocean_proximity': ['NEAR BAY'],
    }

    df = pd.DataFrame(eg_data)
    assert df.shape[0] == 1

    df = add_features(df)

    assert df["rooms_per_household"].iloc[0] == 880 / 126
    assert df["bedrooms_per_room"].iloc[0] == 129 / 880
    assert df["population_per_household"].iloc[0] == 322 / 126
