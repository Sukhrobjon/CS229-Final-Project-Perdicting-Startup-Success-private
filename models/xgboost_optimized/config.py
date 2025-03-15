# config.py

"""
Default configuration settings for the startup success prediction model.
"""

# Data parameters
DATA_PATH = "../data/xgboost_data_with_no_nans.csv"
OUTPUT_DIR = "experiments"

# Time splitting parameters
TRAIN_YEARS = [2012, 2013, 2014, 2015, 2016, 2017]
VAL_YEARS = [2018, 2019, 2020]
TEST_YEARS = [2020, 2021, 2022]

# Feature selection
INCLUDE_FEATURES = None  # Set to None to include all feature groups
EXCLUDE_FEATURES = None  # Set to None to exclude no feature groups

# Model parameters
USE_GPU = True
N_TRIALS = 50
K_FOLD = 5
RUN_CV = True

# Experiment parameters
EXPERIMENT_NAME = "startup_time_based_model"

# Predefined experiment configurations
EXPERIMENT_CONFIGS = {
    "baseline": {
        "EXPERIMENT_NAME": "startup_baseline",
        "INCLUDE_FEATURES": None,
        "EXCLUDE_FEATURES": None,
        "N_TRIALS": 40
    },
    "no_econ": {
        "EXPERIMENT_NAME": "startup_no_econ_indicators",
        "INCLUDE_FEATURES": None,
        "EXCLUDE_FEATURES": ["econ_vc"],
        "N_TRIALS": 40
    },
    "only_team": {
        "EXPERIMENT_NAME": "startup_only_team",
        "INCLUDE_FEATURES": ["team_competency"],
        "EXCLUDE_FEATURES": None,
        "N_TRIALS": 40
    },
    "only_lda": {
        "EXPERIMENT_NAME": "startup_only_lda",
        "INCLUDE_FEATURES": ["lda_topics"],
        "EXCLUDE_FEATURES": None,
        "N_TRIALS": 40
    },
    "quick_test": {
        "EXPERIMENT_NAME": "startup_quick_test",
        "N_TRIALS": 5,
        "RUN_CV": False
    }
}