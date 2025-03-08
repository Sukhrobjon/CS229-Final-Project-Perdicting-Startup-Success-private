import json
from datetime import datetime

def load_json_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {filename}")
        return None

def save_json_file(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=2)

# Load the files
file1 = load_json_file('companies_merged_chunk_7.json')  # First search file
file2 = load_json_file('companies_merged_chunk_8.json')  # Second search file

if not all([file1, file2]):
    print("Error loading one or more files")
    exit(1)

# Create a dictionary to track unique companies
unique_companies = {}
duplicates_count = 0

# Process companies from both files
files_to_process = [file1, file2]

for file_data in files_to_process:
    for company in file_data['companies']:
        company_id = company['id']
        
        if company_id not in unique_companies:
            unique_companies[company_id] = company
        else:
            duplicates_count += 1

# Create new metadata
new_metadata = {
    "total_companies": len(unique_companies),
    "files_merged": len(files_to_process),
    "duplicates_found": duplicates_count,
    "timestamp": datetime.now().isoformat(),
    "processing_stats": {
        "had_errors": False
    }
}

# Create the final output structure
output_data = {
    "metadata": new_metadata,
    "companies": list(unique_companies.values())
}

# Save to new file
save_json_file(output_data, 'companies_merge_50k_chunk_5.json')

print(f"Processing complete:")
print(f"Total unique companies: {len(unique_companies)}")
print(f"Duplicates found: {duplicates_count}")
print(f"Output saved to: merged_companies.json")