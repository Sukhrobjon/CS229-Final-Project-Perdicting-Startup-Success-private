import json
import os
from typing import Dict, Any, Optional
import requests
from utils.config import Config

class SearchProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.default_dsl = {
            "offset": 0,
            "limit": 1,
            "filters": [
                {
                    "AND": [
                        {
                            "founded": {
                                "operation": "gte",
                                "value": "1998-12-31T23:59:59.999Z"
                            }
                        },
                        {
                            "founded": {
                                "operation": "lte",
                                "value": "2024-12-31T23:59:59.999Z"
                            }
                        },
                        {
                            "country": {
                                "operation": "eq",
                                "value": "United States"
                            }
                        }
                    ]
                }
            ]
        }

    def search_companies(self, dsl: Optional[Dict[str, Any]] = None) -> Dict:
        """Make a search request to Aviato API"""
        url = f"{Config.BASE_URL}/company/search"
        
        if dsl is None:
            dsl = self.default_dsl
        
        payload = {"dsl": dsl}
        
        print("\nRequest Details:")
        print(f"URL: {url}")
        print("Payload:", json.dumps(payload, indent=2))
        
        response = requests.post(url, headers=self.headers, json=payload)
        
        print("\nResponse Status:", response.status_code)
        
        if response.status_code != 200:
            print("Response Content:", response.text)
            
        response.raise_for_status()
        return response.json()

    def save_results(self, data: Dict) -> None:
        """Save search results to JSON"""
        # Ensure output directory exists
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        # Save in output directory
        output_file = os.path.join(Config.OUTPUT_DIR, Config.COMPANIES_FILE)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nSearch results saved to {output_file}")
        print(f"Total results: {data.get('count', {}).get('value', 'N/A')}")
        
        # Also save a copy in root directory for backward compatibility
        with open(Config.COMPANIES_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Backup copy saved to {Config.COMPANIES_FILE}")