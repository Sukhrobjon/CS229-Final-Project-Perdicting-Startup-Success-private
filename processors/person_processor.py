import requests
import json
import os
from typing import Dict, Optional, List
import time
from processors.base_processor import BaseProcessor
from utils.config import Config

class PersonEnrichmentProcessor(BaseProcessor):
    def __init__(self, api_key: str, batch_size: int = 1000):
        super().__init__(api_key, batch_size)
        self.max_batch_size = 20  # Maximum IDs per API request
        self.progress_interval = 5  # Show progress every 5 batches (100 persons)

    def get_person_bulk_enrichment(self, person_ids: List[str]) -> Optional[List[Dict]]:
        """Get enriched person data in bulk including experience, education, and degrees"""
        url = "https://data.api.aviato.co/person/bulk/enrich"
        all_results = []
        successful_persons = 0
        start_time = time.time()
        
        total_ids = len(person_ids)
        total_batches = (total_ids + self.max_batch_size - 1) // self.max_batch_size
        
        print(f"\nStarting Person Enrichment processing for {total_ids} persons...")
        
        # Only show batch info for larger datasets
        if total_batches > 1:
            print(f"Processing in {total_batches} batches (max {self.max_batch_size} persons per batch)")
        
        for i in range(0, total_ids, self.max_batch_size):
            batch_ids = person_ids[i:i + self.max_batch_size]
            current_batch = i // self.max_batch_size + 1
            
            payload = {
                "lookups": [{"id": person_id} for person_id in batch_ids],
                "include": ["EXPERIENCE", "EDUCATION", "DEGREES", "LOCATION_DETAILS"]
            }
            
            headers = {
                **self.headers,
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.post(url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    all_results.extend(data)
                    
                    # Store results
                    for person_data in data:
                        person_id = person_data.get('person', {}).get('id')
                        if person_id:
                            self.data[person_id] = person_data
                            successful_persons += 1
                    
                    # Show periodic progress for larger datasets
                    if total_batches > 1 and current_batch % self.progress_interval == 0:
                        elapsed_time = time.time() - start_time
                        avg_time_per_batch = elapsed_time / current_batch
                        remaining_batches = total_batches - current_batch
                        estimated_remaining_time = remaining_batches * avg_time_per_batch
                        
                        print(f"\nProgress: Batch {current_batch}/{total_batches} "
                              f"({(current_batch/total_batches)*100:.1f}%)")
                        print(f"Persons processed: {successful_persons}/{total_ids}")
                        print(f"Estimated time remaining: {estimated_remaining_time/60:.1f} minutes")
                    
                    if i + self.max_batch_size < total_ids:
                        time.sleep(1)  # Rate limiting
                else:
                    print(f"\nError in batch {current_batch}: Status {response.status_code}")
                    
            except Exception as e:
                print(f"\nError processing batch {current_batch}: {str(e)}")
                continue
        
        # Final summary
        elapsed_time = time.time() - start_time
        print(f"\nPerson Enrichment processing completed:")
        print(f"Total persons processed: {total_ids}")
        print(f"Successful enrichments: {successful_persons}")
        if total_batches > 1:
            print(f"Total batches processed: {total_batches}")
        print(f"Total time taken: {elapsed_time/60:.1f} minutes")
        
        return all_results if all_results else None

    def save_batch(self):
        """Save current batch of person enrichment data"""
        filename = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['person_enrichment'])
        self.append_to_json(filename)

    def process_companies(self, company_ids: List[str]):
        """Not used in this processor"""
        raise NotImplementedError("This processor doesn't process companies directly")

    def process_person_ids(self, person_ids: List[str]):
        """Process list of person IDs for enrichment"""
        try:
            # Check both possible locations for founders file
            founders_file = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['founders'])
            root_founders_file = Config.OUTPUT_FILES['founders']
            
            if os.path.exists(founders_file):
                print(f"Using founders data from output directory")
                founders_data_path = founders_file
            elif os.path.exists(root_founders_file):
                print(f"Using founders data from root directory")
                founders_data_path = root_founders_file
            else:
                print("Founders data not found. Skipping person enrichment.")
                return

            total_persons = len(person_ids)
            print(f"\nStarting Person Enrichment processing for {total_persons} persons...")
            
            enrichment_data = self.get_person_bulk_enrichment(person_ids)
            
            # Always save data if we have any results, regardless of size
            if self.data:
                print(f"\nSaving enrichment data for {len(self.data)} persons...")
                self.save_batch()
                print(f"Data saved to: {os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['person_enrichment'])}")
            else:
                print("No enrichment data was collected to save.")
            
            # Reset after saving
            self.data = {}
                
        except Exception as e:
            print(f"Error processing person enrichment: {str(e)}")
            # Try to save any data we might have collected before the error
            if self.data:
                print("\nAttempting to save partial data...")
                self.save_batch()
                print("Partial data saved.")