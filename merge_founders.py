import json
import glob
from datetime import datetime
import os
from pathlib import Path

def load_json_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def save_json_file(data, filename):
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=2)

def merge_founders_chunks():
    # Define input and output paths
    input_pattern = 'output/fouders_1st_20k/founders_chunk_*.json'
    output_dir = 'output/fouders_1st_20k'
    
    # Get all founders chunk files
    chunk_files = sorted(glob.glob(input_pattern))
    
    if not chunk_files:
        print(f"No founder chunk files found in {input_pattern}!")
        return

    merged_data = {}
    total_founders = 0
    total_companies = 0
    companies_with_founders = 0
    companies_without_founders = 0
    failed_calls = 0
    processing_time = 0

    print(f"Found {len(chunk_files)} chunk files to process...")

    # Process each chunk file
    for chunk_file in chunk_files:
        print(f"Processing {os.path.basename(chunk_file)}...")
        data = load_json_file(chunk_file)
        if not data:
            continue

        # Extract metadata from the chunk
        chunk_metadata = data.get('metadata', {})
        failed_calls += chunk_metadata.get('failed_calls', 0)
        processing_time += float(chunk_metadata.get('uptime', 0)) if 'uptime' in chunk_metadata else 0

        # Merge company data
        for company_id, company_data in data.items():
            if company_id != 'metadata':
                merged_data[company_id] = company_data
                total_companies += 1
                if company_data.get('founders'):
                    companies_with_founders += 1
                    total_founders += len(company_data['founders'])
                else:
                    companies_without_founders += 1

    # Calculate average founders per company
    avg_founders = total_founders / companies_with_founders if companies_with_founders > 0 else 0

    # Create new metadata
    new_metadata = {
        "metadata": {
            "processed_companies": total_companies,
            "total_founders": total_founders,
            "processing_time": f"{processing_time/60:.1f} minutes",
            "success_rate": f"{((total_companies - failed_calls)/total_companies * 100):.1f}%",
            "timestamp": datetime.now().isoformat(),
            "source_files": [os.path.basename(f) for f in chunk_files],
            "api_calls": {
                "total": total_companies,
                "successful": total_companies - failed_calls,
                "failed": failed_calls
            },
            "error_counts": {
                "400": {"count": 0, "description": "Bad Request"},
                "401": {"count": 0, "description": "Unauthorized"},
                "403": {"count": 0, "description": "Forbidden"},
                "404": {"count": 0, "description": "Not Found"},
                "429": {"count": 0, "description": "Rate Limited"},
                "500": {"count": 0, "description": "Server Error"},
                "timeout": {"count": 0, "description": "Request Timeout"},
                "other": {"count": failed_calls, "description": "Other Errors"}
            },
            "total_companies": total_companies,
            "companies_with_founders": companies_with_founders,
            "companies_without_founders": companies_without_founders,
            "average_founders_per_company": avg_founders
        }
    }

    # Combine metadata and company data
    final_data = {**new_metadata, **merged_data}

    # Create output filename in the same directory
    output_filename = os.path.join(output_dir, f'founders_merged_{total_companies}_companies.json')
    save_json_file(final_data, output_filename)
    
    print(f"\nMerge completed successfully!")
    print(f"Total companies processed: {total_companies}")
    print(f"Total founders found: {total_founders}")
    print(f"Companies with founders: {companies_with_founders}")
    print(f"Companies without founders: {companies_without_founders}")
    print(f"Average founders per company: {avg_founders:.2f}")
    print(f"Output saved to: {output_filename}")

if __name__ == "__main__":
    merge_founders_chunks()