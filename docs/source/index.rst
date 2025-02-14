.. house_pricing_predictor documentation master file, created by
   sphinx-quickstart on Thu Feb 13 04:56:31 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

House Pricing Predictor Documentation
=====================================

This is the official documentation for the **House Pricing Predictor** project. 
It aims to predict house prices based on various features such as location, number of rooms, area, etc. 

Overview
--------

The House Pricing Predictor uses machine learning techniques to predict the median house prices in different areas based on a dataset of housing features. The system is built using Python and employs a variety of libraries including scikit-learn, pandas, and NumPy.

Features
--------

- Data ingestion which includes data preprocessing and train/test split with stratified sampling
- Model training using regression algorithms and selecting best model using gridsearch
- Evaluation with metrics like MSE, RMSE, and MAE

Modules
-------

The project is structured into several modules. Each module handles a specific task in the pipeline from data ingestion to model evaluation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   data_ingestion.rst
   model_training.rst
   model_scoring.rst

