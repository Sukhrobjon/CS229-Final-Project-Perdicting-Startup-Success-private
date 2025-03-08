import json
from datetime import datetime
import math

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

# Load the merged file
merged_data = load_json_file('companies_with_founders_non_empty.json')

if not merged_data:
    print("Error loading merged file")
    exit(1)

# Calculate chunks
companies = merged_data['companies']
total_companies = len(companies)
chunk_size = 20000
num_full_chunks = 7

# Create chunks
chunks = []
for i in range(num_full_chunks):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size
    chunks.append(companies[start_idx:end_idx])

# Add remaining companies to the last chunk
remaining_companies = companies[num_full_chunks * chunk_size:]
chunks.append(remaining_companies)

# Save each chunk with appropriate metadata
for i, chunk in enumerate(chunks, 1):
    chunk_metadata = {
        "total_companies": len(chunk),
        "chunk_number": i,
        "total_chunks": 8,
        "timestamp": datetime.now().isoformat(),
        "processing_stats": {
            "had_errors": False
        }
    }

    chunk_data = {
        "metadata": chunk_metadata,
        "companies": chunk
    }

    filename = f"companies_merged_chunk_{i}.json"
    save_json_file(chunk_data, filename)
    print(f"Chunk {i} saved with {len(chunk)} companies to {filename}")

print("\nProcessing complete:")
print(f"Total companies processed: {total_companies}")
print(f"Created 8 chunks:")
for i in range(8):
    print(f"Chunk {i+1}: {len(chunks[i])} companies")