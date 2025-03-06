import json
from typing import Dict, Any, Optional
import requests
import time
from datetime import datetime
from utils.config import Config
from utils.file_manager import FileManager

class SearchProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.file_manager = FileManager(Config.OUTPUT_DIR)
        self.progress_file = os.path.join(Config.OUTPUT_DIR, "search_progress.json")
        self.last_company_name = None
        self.total_available = 0
        self.total_processed = 0
        self.api_calls_made = 0

    def get_search_dsl(self, offset: int) -> Dict:
        """Generate DSL with pagination and sorting"""
        filters = [
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

        # Add name filter for subsequent batches
        if self.last_company_name:
            filters[0]["AND"].append({
                "name": {
                    "operation": "gt",
                    "value": self.last_company_name
                }
            })

        return {
            "offset": offset,
            "limit": 250,  # Maximum efficient batch size
            "sort": [
                {"name": {"order": "asc"}}
            ],
            "filters": filters
        }

    def search_companies(self, dsl: Dict) -> Dict:
        """Make a search request to Aviato API"""
        url = f"{Config.BASE_URL}/company/search"
        payload = {"dsl": dsl}
        
        response = requests.post(url, headers=self.headers, json=payload)
        self.api_calls_made += 1
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
        response.raise_for_status()
        return response.json()

    def save_progress(self):
        """Save current progress"""
        progress = {
            "total_available": self.total_available,
            "total_processed": self.total_processed,
            "api_calls_made": self.api_calls_made,
            "last_company_name": self.last_company_name,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def load_progress(self) -> Dict:
        """Load previous progress"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.total_processed = progress.get('total_processed', 0)
                    self.api_calls_made = progress.get('api_calls_made', 0)
                    self.last_company_name = progress.get('last_company_name')
                    return progress
            except Exception as e:
                print(f"Error loading progress: {e}")
        return {}

    def process_companies(self, target_companies: int = 1000):
        """Process companies with resume capability"""
        print(f"\nStarting company search process for {target_companies} companies")
        self.load_progress()
        
        companies_buffer = []
        start_offset = self.total_processed

        try:
            while self.total_processed < target_companies:
                # Get batch of companies
                dsl = self.get_search_dsl(len(companies_buffer))
                response = self.search_companies(dsl)
                
                # Update total on first call
                if self.total_available == 0:
                    self.total_available = int(response.get('count', {}).get('value', 0))
                    print(f"\nTotal available companies: {self.total_available}")

                items = response.get('items', [])
                if not items:
                    break

                companies_buffer.extend(items)
                self.last_company_name = items[-1].get('name')
                
                # Save batch when buffer reaches or exceeds batch size
                if len(companies_buffer) >= self.file_manager.batch_size:
                    filename, end_offset = self.file_manager.save_batch(
                        companies_buffer[:self.file_manager.batch_size],
                        start_offset
                    )
                    
                    # Update counters
                    self.total_processed += self.file_manager.batch_size
                    companies_buffer = companies_buffer[self.file_manager.batch_size:]
                    start_offset = end_offset
                    
                    # Save progress
                    self.save_progress()
                    
                    # Progress update
                    print(f"\nProgress: {self.total_processed}/{target_companies} "
                          f"({(self.total_processed/target_companies)*100:.1f}%)")
                    print(f"API calls made: {self.api_calls_made}")

                time.sleep(1)  # Rate limiting

            # Save any remaining companies
            if companies_buffer:
                self.file_manager.save_batch(companies_buffer, start_offset)
                self.total_processed += len(companies_buffer)
                self.save_progress()

            # Merge all files if processing is complete
            if self.total_processed >= target_companies:
                print("\nProcessing complete. Merging files...")
                merged_file = self.file_manager.merge_files(target_companies)
                print(f"Final merged file: {merged_file}")

        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            # Save progress and current buffer on error
            if companies_buffer:
                self.file_manager.save_batch(companies_buffer, start_offset)
            self.save_progress()
            raise

        finally:
            # Print final statistics
            print("\nSearch Processing Complete:")
            print("------------------------")
            print(f"Total Companies Found: {self.total_available}")
            print(f"Total Companies Processed: {self.total_processed}")
            print(f"Total API Calls Made: {self.api_calls_made}")
            print(f"Files Generated: {len(os.listdir(Config.OUTPUT_DIR))}")
            print("------------------------")