import json
from typing import List, Dict, Any
import os

class FileHandler:
    @staticmethod
    def read_company_ids_from_json(json_file_path: str) -> List[str]:
        """Read company IDs from a JSON file containing items array"""
        company_ids = []
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                
                # Extract company IDs from items array
                if 'items' in data:
                    for item in data['items']:
                        if 'id' in item:
                            company_ids.append(item['id'])
                    
                    total_count = data.get('count', {}).get('value', 'unknown')
                    print(f"Successfully read {len(company_ids)} company IDs from {json_file_path}")
                    print(f"Total companies in data: {total_count}")
                else:
                    print("Error: No 'items' array found in JSON file")
                    
        except FileNotFoundError:
            print(f"Error: JSON file not found at {json_file_path}")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {json_file_path}")
        except Exception as e:
            print(f"Error reading JSON file: {str(e)}")
            
        return company_ids

    @staticmethod
    def extract_person_ids_from_founders(founders_file: str) -> List[str]:
        """Extract person IDs from founders JSON file"""
        person_ids = []
        try:
            if not os.path.exists(founders_file):
                print(f"Error: Founders file not found at {founders_file}")
                return person_ids

            with open(founders_file, 'r') as f:
                founders_data = json.load(f)
                for company_data in founders_data.values():
                    for founder in company_data.get('founders', []):
                        person_id = founder.get('id')
                        if person_id:
                            person_ids.append(person_id)
            
            # Remove duplicates while maintaining order
            person_ids = list(dict.fromkeys(person_ids))
            print(f"Successfully extracted {len(person_ids)} unique person IDs from founders data")
            
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {founders_file}")
        except Exception as e:
            print(f"Error extracting person IDs: {str(e)}")
        
        return person_ids

    @staticmethod
    def ensure_output_directory(directory: str = "output"):
        """Ensure output directory exists"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created output directory: {directory}")