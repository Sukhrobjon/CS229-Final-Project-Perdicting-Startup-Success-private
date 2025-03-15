# utils.py

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.2)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_object(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved to {filepath}")

def load_object(filepath):
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from {filepath}")
    return obj

def generate_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def split_features_by_group(df, target_col='success'):
    feature_groups = {
        'all': [col for col in df.columns if col != target_col],
        'team_competency': [col for col in df.columns if col in 
                           ['founderCount', 'has_previous_startups_total', 
                            'has_board_experience', 'has_executive_experience',
                            'has_advising_experience', 'has_business_education']],
        'econ_vc': [col for col in df.columns if any(x in col for x in 
                   ['gdp', 'consumer', 'treasury', 'vc_'])],
        'temporal': [col for col in df.columns if any(x in col for x in 
                    ['founded', 'company_age', 'days_since'])],
        'location': [col for col in df.columns if any(x in col for x in 
                    ['city', 'state'])],
        'lda_topics': [col for col in df.columns if 'topic_prob_' in col]
    }
    
    return feature_groups

def print_section(title):
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")