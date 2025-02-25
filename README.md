# ** Median housing value prediction **

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## Package Installation
### Setting up MiniConda
'''
conda env create env.yml
conda activate mle-dev2
'''

### Installation of package from .tar.gz and .whl file
The python package can be installed from a .whl (wheel) file or a .tar.gz (source distribution) file using pip. 
Below are the commands for both types of files:
'''
pip install dist/house_pricing_predictor_YUKTHAMAJELLA-0.0.1-py3-none-any.whl
pip install dist/house_pricing_predictor_yukthamajella-0.0.1.tar.gz
'''

### Test the Installation
The following commands can be runned to test the installation of the package. 
'''
pip show house_pricing_predictor_YUKTHAMAJELLA
pip install pytest
pytest tests/test_file.py
pytest tests/test_data_ingestion.py
pytest tests/test_training.py
pytest tests/test_scoring.py
'''

### Running the Application
To execute the application, run the script as follows:
'''
python scripts/ingest_data.py ./processed_data --log-level DEBUG --log-path ./logs/ingest_data.log
python scripts/train.py ./processed_data ./models --log-level INFO --log-path ./logs/train.log --no-console-log
python scripts/score.py ./processed_data ./models --log-level WARNING --log-path ./logs/score.log
'''

### Log Location
Logs for each script will be stored in the ./logs/ directory, with filenames reflecting the script name (e.g., ingest_data.log).

### Running the Help Command to Explore Available Parameters
Use the -h or --help flag to display the available command-line options and their descriptions. This is especially useful to configure data path, model path, log path and other logging configurations.
'''
python scripts/ingest_data.py -h
python scripts/ingest_data.py -h
python scripts/score.py -h
'''