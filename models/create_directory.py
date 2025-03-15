import pandas as pd

DATA_PATH = "./data/xgboost_traing_data_without_main_sector.csv"
df = pd.read_csv(DATA_PATH)
print(df.head(1))

# success_counts = df["success"].value_counts(normalize=True) * 100

# print("✅ Percentage distribution of 'success':\n")
# for value, percent in success_counts.items():
#     print(f"{value}: {percent:.2f}%")

# nan_percentage = (df.isna().sum() / len(df)) * 100

# print("✅ Percentage of NaN values per column:\n")
# for col, percent in nan_percentage.items():
#     print(f"{col}: {percent:.2f}%")

# python main.py \      
#   --train_years 2012 2013 2014 2015 2016 2017 \
#   --val_years 2018 2019 \
#   --test_years 2020 2021 2022 \
#   --use_gpu \
#   --n_trials 70 \
#   --run_cv \
#   --add_time_interactions \
#   --train_time_specific \
#   --experiment_name startup_improved_model \
#   --run_shap_analysis
print(df.shape)
df_cleaned = df.dropna(axis=1, how="all")

# Step 2: Drop rows that have any NaN values
df_cleaned = df_cleaned.dropna()

# Step 3: Save the cleaned DataFrame as a new CSV file
file_path = "xgboost_data_with_no_nans.csv"
df_cleaned.to_csv(file_path, index=False)
print(f"file created")
print(df_cleaned.shape)
