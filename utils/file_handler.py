import csv
import json
import os
from typing import List, Dict
from utils.config import Config

class FileHandler:
    @staticmethod
    def read_company_ids_from_json(json_file_path: str) -> List[str]:
        """Read unique company IDs from a JSON file"""
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                
                # Get companies from the correct location in the structure
                companies = data.get('companies', [])
                
                # Extract unique IDs
                unique_ids = {company['id'] for company in companies if 'id' in company}
                company_ids = list(unique_ids)
                
                total_companies = len(companies)
                duplicates = total_companies - len(unique_ids)
                
                print(f"\nFile: {json_file_path}")
                print(f"Total companies found: {total_companies}")
                print(f"Duplicate IDs removed: {duplicates}")
                print(f"Unique company IDs: {len(company_ids)}")
                
                # Print first few IDs for verification
                if company_ids:
                    print("\nFirst few company IDs:")
                    for id in company_ids[:5]:
                        print(f"- {id}")
                
                return company_ids
        except Exception as e:
            print(f"Error reading JSON file: {str(e)}")
            return []

    @staticmethod
    def ensure_output_directory(directory: str = "output"):
        """Ensure output directory exists"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created output directory: {directory}")

    @staticmethod
    def extract_person_ids_from_founders(founders_file: str) -> List[str]:
        """Extract person IDs from founders JSON file"""
        try:
            if not os.path.exists(founders_file):
                print(f"Error: Founders file not found at {founders_file}")
                return []

            with open(founders_file, 'r') as f:
                founders_data = json.load(f)
                person_ids = []
                for company_data in founders_data.values():
                    for founder in company_data.get('founders', []):
                        person_id = founder.get('id')
                        if person_id:
                            person_ids.append(person_id)
                
                # Remove duplicates while maintaining order
                person_ids = list(dict.fromkeys(person_ids))
                print(f"Successfully extracted {len(person_ids)} unique person IDs")
                return person_ids
                
        except Exception as e:
            print(f"Error extracting person IDs: {str(e)}")
            return []