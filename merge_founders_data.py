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
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=2)

def merge_json_files():
    # Define input and output paths
    input_pattern = 'output/temp/*.json'
    output_dir = 'output/temp'
    
    # Get all JSON files
    json_files = sorted(glob.glob(input_pattern))
    
    if not json_files:
        print(f"No JSON files found in {input_pattern}!")
        return

    merged_data = {}
    total_processed = 0
    total_founders = 0
    total_companies = 0
    companies_with_founders = 0
    companies_without_founders = 0
    total_failed_calls = 0
    total_successful_calls = 0
    processing_time = 0.0

    print(f"Found {len(json_files)} files to process...")

    # Process each file
    for json_file in json_files:
        print(f"Processing {os.path.basename(json_file)}...")
        data = load_json_file(json_file)
        if not data:
            continue

        # Extract metadata
        file_metadata = data.get('metadata', {})
        total_processed += file_metadata.get('processed_companies', 0)
        total_founders += file_metadata.get('total_founders', 0)
        processing_time += float(file_metadata.get('processing_time', '0').split()[0]) # Extract minutes
        total_failed_calls += file_metadata.get('api_calls', {}).get('failed', 0)
        total_successful_calls += file_metadata.get('api_calls', {}).get('successful', 0)

        # Merge company data
        for key, value in data.items():
            if key != 'metadata':
                merged_data[key] = value
                total_companies += 1
                if value.get('founders'):
                    companies_with_founders += 1
                else:
                    companies_without_founders += 1

    # Calculate averages and rates
    avg_founders = total_founders / companies_with_founders if companies_with_founders > 0 else 0
    success_rate = (total_successful_calls / (total_successful_calls + total_failed_calls)) * 100 if (total_successful_calls + total_failed_calls) > 0 else 0

    # Create new metadata
    new_metadata = {
        "metadata": {
            "processed_companies": total_companies,
            "total_founders": total_founders,
            "processing_time": f"{processing_time:.1f} minutes",
            "success_rate": f"{success_rate:.1f}%",
            "timestamp": datetime.now().isoformat(),
            "source_files": [os.path.basename(f) for f in json_files],
            "api_calls": {
                "total": total_successful_calls + total_failed_calls,
                "successful": total_successful_calls,
                "failed": total_failed_calls
            },
            "error_counts": {
                "400": {"count": 0, "description": "Bad Request"},
                "401": {"count": 0, "description": "Unauthorized"},
                "403": {"count": 0, "description": "Forbidden"},
                "404": {"count": 0, "description": "Not Found"},
                "429": {"count": 0, "description": "Rate Limited"},
                "500": {"count": 0, "description": "Server Error"},
                "timeout": {"count": 0, "description": "Request Timeout"},
                "other": {"count": total_failed_calls, "description": "Other Errors"}
            },
            "total_companies": total_companies,
            "companies_with_founders": companies_with_founders,
            "companies_without_founders": companies_without_founders,
            "average_founders_per_company": avg_founders
        }
    }

    # Combine metadata and company data
    final_data = {**new_metadata, **merged_data}

    # Create output filename
    output_filename = os.path.join(output_dir, f'merged_all_{total_companies}_companies.json')
    save_json_file(final_data, output_filename)
    
    print(f"\nMerge completed successfully!")
    print(f"Total files processed: {len(json_files)}")
    print(f"Total companies: {total_companies}")
    print(f"Total founders: {total_founders}")
    print(f"Companies with founders: {companies_with_founders}")
    print(f"Companies without founders: {companies_without_founders}")
    print(f"Average founders per company: {avg_founders:.2f}")
    print(f"Overall success rate: {success_rate:.1f}%")
    print(f"Total processing time: {processing_time:.1f} minutes")
    print(f"Output saved to: {output_filename}")

if __name__ == "__main__":
    merge_json_files()