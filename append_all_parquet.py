import os
import pandas as pd
import pyarrow.parquet as pq

# Path where your Parquet files are stored
parquet_dir, output_filename = ("/Users/sukhrobjongolibboev/Desktop/aviato_all_data/company_enrich/parquet_files", "merged_company_enrich.parquet")

# List all Parquet files in the directory
parquet_files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith(".parquet")]

# Merge all files
dfs = [pd.read_parquet(file) for file in parquet_files]  # Read all Parquet files
merged_df = pd.concat(dfs, ignore_index=True)  # Append them together

# Save the merged file
output_file = os.path.join(parquet_dir, output_filename)
merged_df.to_parquet(output_file, engine="pyarrow")

print(f"✅ Merged {len(parquet_files)} Parquet files into {output_file}")
