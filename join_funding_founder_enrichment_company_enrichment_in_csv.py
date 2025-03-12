import pandas as pd
from tqdm import tqdm

# Define file paths
funding_csv_path = '/Users/sukhrobjongolibboev/Desktop/Stanford/CS229Winter25/FinalProject/draftApiCall/project/output/funding_1st_20k/company_funding_all_5_chunks.csv'
founders_csv_path = 'founders_with_enrichment_all_5_chunks.csv'
parquet_file_path = '/Users/sukhrobjongolibboev/Desktop/Stanford/CS229Winter25/FinalProject/draftApiCall/project/output/parquet_files/merged_company_enrich.parquet'

# Read the CSV files with progress tracking
print("Loading funding CSV...")
funding_df = pd.read_csv(funding_csv_path)
print(f"Funding CSV loaded. Shape: {funding_df.shape}")

print("Loading founders CSV...")
founders_df = pd.read_csv(founders_csv_path)
print(f"Founders CSV loaded. Shape: {founders_df.shape}")

# Perform the first inner join on company_id
print("Performing first inner join on company_id...")
merged_df = pd.merge(funding_df, founders_df, on='company_id', how='inner')
print(f"First join completed. Shape: {merged_df.shape}")

# Read the Parquet file with progress tracking
print("Loading enrichment Parquet file...")
enrichment_df = pd.read_parquet(parquet_file_path)
print(f"Enrichment Parquet file loaded. Shape: {enrichment_df.shape}")

# Perform the second inner join on company_id
print("Performing second inner join on company_id...")
final_merged_df = pd.merge(merged_df, enrichment_df, on='company_id', how='inner')
print(f"Second join completed. Shape: {final_merged_df.shape}")

# Optionally, save the final merged DataFrame to a CSV file
output_csv_path = '/Users/sukhrobjongolibboev/Desktop/Stanford/CS229Winter25/FinalProject/draftApiCall/project/output/final_merged_data_chunk_1.csv'
print(f"Saving final merged DataFrame to {output_csv_path}...")
final_merged_df.to_csv(output_csv_path, index=False)
print("Final merged DataFrame saved.")

# Print summary of the final merged DataFrame
print(f"Final merged DataFrame shape: {final_merged_df.shape}")
print(final_merged_df.head())