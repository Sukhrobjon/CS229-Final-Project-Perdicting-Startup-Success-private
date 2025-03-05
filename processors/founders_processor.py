import requests
import json
import os
from typing import Dict, Optional, List
import time
from processors.base_processor import BaseProcessor
from utils.config import Config

class FoundersProcessor(BaseProcessor):
    def __init__(self, api_key: str, batch_size: int = 1000):
        super().__init__(api_key, batch_size)
        self.progress_interval = 100  # Show progress every 100 companies

    def get_founders(self, company_id: str, per_page: int = 10, page: int = 0) -> Optional[Dict]:
        """Get company founders"""
        url = f"https://data.api.aviato.co/company/{company_id}/founders"
        params = {
            "perPage": per_page,
            "page": page
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching founders for {company_id}: {response.status_code}")
            return None

    def save_batch(self):
        """Save current batch of founders data"""
        filename = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['founders'])
        self.append_to_json(filename)

    def process_companies(self, company_ids: List[str]):
        """Process list of companies for founders"""
        total_companies = len(company_ids)
        start_time = time.time()
        successful_processes = 0
        founders_found = 0
        
        print(f"\nStarting Founders processing for {total_companies} companies...")

        for i, company_id in enumerate(company_ids, 1):
            try:
                founders_data = self.get_founders(company_id)
                if founders_data:
                    self.data[company_id] = founders_data
                    successful_processes += 1
                    # Count founders found
                    founders_found += len(founders_data.get('founders', []))
                    
                    # Show periodic progress for large datasets
                    if i % self.progress_interval == 0:
                        elapsed_time = time.time() - start_time
                        avg_time_per_company = elapsed_time / i
                        remaining_companies = total_companies - i
                        estimated_remaining_time = remaining_companies * avg_time_per_company
                        
                        print(f"\nProgress: {i}/{total_companies} companies processed "
                              f"({(i/total_companies)*100:.1f}%)")
                        print(f"Founders found so far: {founders_found}")
                        print(f"Estimated time remaining: {estimated_remaining_time/60:.1f} minutes")

                    # Don't check batch size for small datasets
                    if total_companies > self.batch_size:
                        self.check_and_save_batch()
                
                if i < total_companies:
                    time.sleep(1)  # Rate limiting
                    
            except Exception as e:
                print(f"Error processing company {company_id}: {str(e)}")
                continue
        
        # Final save if we have data
        if self.data:
            self.save_batch()

        # Get unique founder IDs for final summary
        unique_founders = len(self.get_all_founder_ids())

        # Final summary
        elapsed_time = time.time() - start_time
        print(f"\nFounders processing completed:")
        print(f"Total companies processed: {total_companies}")
        print(f"Successful processes: {successful_processes}")
        print(f"Total founders found: {founders_found}")
        print(f"Unique founders: {unique_founders}")
        print(f"Total time taken: {elapsed_time/60:.1f} minutes")
        print(f"Data saved to: {os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['founders'])}")

    def get_all_founder_ids(self) -> List[str]:
        """Extract all founder IDs from the collected data"""
        founder_ids = []
        for company_data in self.data.values():
            for founder in company_data.get('founders', []):
                founder_id = founder.get('id')
                if founder_id:
                    founder_ids.append(founder_id)
        return list(set(founder_ids))  # Remove duplicates