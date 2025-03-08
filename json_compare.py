import json

# Read the JSON files
def load_json_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Load both JSON files
search_data = load_json_file('search.json')
founders_data = load_json_file('founders.json')

# Get list of company IDs from search.json
search_company_ids = [company['id'] for company in search_data['companies']]

# Get list of company IDs from founders.json (these are the keys)
founders_company_ids = list(founders_data.keys())

# Find matching IDs
matching_ids = set(search_company_ids) & set(founders_company_ids)

# Print results
print(f"Total companies in search.json: {len(search_company_ids)}")
print(f"Total companies in founders.json: {len(founders_company_ids)}")
print(f"Number of matching company IDs: {len(matching_ids)}")

# Optional: Print the matching IDs
print("\nMatching company IDs:")
for company_id in matching_ids:
    print(company_id)