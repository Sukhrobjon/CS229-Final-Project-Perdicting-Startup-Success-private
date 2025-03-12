import pandas as pd

# Define file path
file_path = "/Users/sukhrobjongolibboev/Desktop/Stanford/CS229Winter25/FinalProject/draftApiCall/project/output/parquet_files/merged_company_enrich.parquet"
funding_data = "company_funding.parquet"
founders_enriched = "enriched_founders.parquet"
company_enriched = "merged_company_enrich.parquet"


# Load the Parquet file
df = pd.read_parquet(file_path)

# print(df.columns.to_list)
print(df[df["company_id"] == "A6yFDtYZ64HNgyIJBIrUSa27YSpgBFC"].head())

# print(df.head())
# print(f"✅ Column 'id' renamed to 'company_id' and saved back to {file_path}")

