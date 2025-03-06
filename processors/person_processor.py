import requests
import json
import os
import glob
from typing import Dict, Optional, List
import time
from datetime import datetime
from statistics import mean, median
from processors.base_processor import BaseProcessor
from utils.config import Config
from utils.rate_limiter import RateLimiter

class PersonEnrichmentProcessor(BaseProcessor):
    def __init__(self, api_key: str, batch_size: int = 1000):
        super().__init__(api_key, batch_size)
        # API and batch configuration
        self.max_batch_size = 20  # API limit for bulk enrichment
        self.progress_interval = 5
        self.rate_limiter = RateLimiter(max_requests=10000)
        self.session = requests.Session()
        
        # Progress tracking
        self.progress_file = os.path.join(Config.OUTPUT_DIR, "person_enrichment_progress.json")
        
        # API tracking
        self.api_calls_made = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.had_errors = False
        
        # Buffer configuration
        self.buffer_size = 100
        self.chunk_size = 10000
        self.buffer = {}
        
        # Performance tracking
        self.total_persons = 0
        self.request_times = []
        self.last_request_time = time.time()
        
        # Initialize session headers
        self.session.headers.update({
            **self.headers,
            "Content-Type": "application/json"
        })

    def get_performance_metrics(self) -> Dict:
        """Calculate current performance metrics"""
        if not self.request_times:
            return {
                "avg_request_time": 0,
                "median_request_time": 0,
                "min_request_time": 0,
                "max_request_time": 0,
                "current_rate": 0
            }
            
        recent_times = self.request_times[-1000:]  # Last 1000 requests
        avg_time = mean(recent_times)
        
        return {
            "avg_request_time": avg_time,
            "median_request_time": median(recent_times),
            "min_request_time": min(recent_times),
            "max_request_time": max(recent_times),
            "current_rate": 60 / avg_time if avg_time > 0 else 0
        }

    def save_progress(self, current_index: int, batch_ids: List[str]):
        """Save current progress"""
        metrics = self.get_performance_metrics()
        
        progress = {
            "last_index": current_index,
            "last_batch_ids": batch_ids,
            "persons_processed": self.successful_calls,
            "api_calls_made": self.api_calls_made,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "total_persons": self.total_persons,
            "had_errors": self.had_errors,
            "performance_metrics": metrics,
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
                    self.api_calls_made = progress.get('api_calls_made', 0)
                    self.successful_calls = progress.get('successful_calls', 0)
                    self.failed_calls = progress.get('failed_calls', 0)
                    self.total_persons = progress.get('total_persons', 0)
                    self.had_errors = progress.get('had_errors', False)
                    return progress
            except Exception as e:
                print(f"Error loading progress: {e}")
        return {}

    def append_buffer_to_temp(self):
        """Append current buffer to temp file"""
        if not self.buffer:
            return
            
        temp_file = os.path.join(Config.OUTPUT_DIR, "person_enrichment_temp.json")
        
        try:
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    existing_data = json.load(f)
                existing_data.update(self.buffer)
                data_to_save = existing_data
            else:
                data_to_save = self.buffer

            with open(temp_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            self.buffer = {}
            
        except Exception as e:
            print(f"Error appending to temp file: {str(e)}")
            self.had_errors = True

    def save_batch(self):
        """Implementation of abstract method from BaseProcessor"""
        self.append_buffer_to_temp()

    def save_chunk_if_needed(self, persons_processed: int):
        """Create new chunk file at threshold"""
        if persons_processed % self.chunk_size == 0:
            chunk_number = persons_processed // self.chunk_size
            temp_file = os.path.join(Config.OUTPUT_DIR, "person_enrichment_temp.json")
            chunk_file = os.path.join(Config.OUTPUT_DIR, f"person_enrichment_chunk_{chunk_number}.json")
            
            if os.path.exists(temp_file):
                os.rename(temp_file, chunk_file)
                print(f"\nCreated chunk file: {chunk_file}")

    def merge_final_data(self):
        """Merge all data into final file"""
        final_data = {}
        
        try:
            chunk_pattern = os.path.join(Config.OUTPUT_DIR, "person_enrichment_chunk_*.json")
            chunk_files = sorted(glob.glob(chunk_pattern))
            
            if chunk_files:
                print("\nMerging chunk files...")
                for i, chunk_file in enumerate(chunk_files, 1):
                    print(f"Processing chunk file {i}/{len(chunk_files)}")
                    with open(chunk_file, 'r') as f:
                        chunk_data = json.load(f)
                        final_data.update(chunk_data)
                    os.remove(chunk_file)
            
            temp_file = os.path.join(Config.OUTPUT_DIR, "person_enrichment_temp.json")
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    temp_data = json.load(f)
                    final_data.update(temp_data)
                os.remove(temp_file)
            
            final_file = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['person_enrichment'])
            with open(final_file, 'w') as f:
                json.dump(final_data, f, indent=2)
            
            print("\nFinal Output Summary:")
            print("------------------------")
            print(f"Output file: {final_file}")
            print(f"Total persons processed: {len(final_data)}")
            print("------------------------")
            
        except Exception as e:
            print(f"Error during final merge: {str(e)}")
            self.had_errors = True

    def get_person_bulk_enrichment(self, person_ids: List[str]) -> Optional[List[Dict]]:
        """Get enriched person data in bulk including experience, education, and degrees"""
        url = "https://data.api.aviato.co/person/bulk/enrich"
        all_results = []
        successful_persons = 0
        start_time = time.time()
        
        total_ids = len(person_ids)
        total_batches = (total_ids + self.max_batch_size - 1) // self.max_batch_size
        
        print(f"\nStarting Person Enrichment processing for {total_ids} persons...")
        
        for i in range(0, total_ids, self.max_batch_size):
            batch_start_time = time.time()
            self.rate_limiter.wait_if_needed()
            
            batch_ids = person_ids[i:i + self.max_batch_size]
            current_batch = i // self.max_batch_size + 1
            
            payload = {
                "lookups": [{"id": person_id} for person_id in batch_ids],
                "include": ["EXPERIENCE", "EDUCATION", "DEGREES", "LOCATION_DETAILS"]
            }
            
            try:
                response = self.session.post(url, json=payload)
                request_time = time.time() - batch_start_time
                self.request_times.append(request_time)
                self.api_calls_made += 1
                
                if response.status_code == 200:
                    self.successful_calls += 1
                    data = response.json()
                    all_results.extend(data)
                    
                    for person_data in data:
                        person_id = person_data.get('person', {}).get('id')
                        if person_id:
                            self.buffer[person_id] = person_data
                            successful_persons += 1
                            self.total_persons += 1
                    
                    if len(self.buffer) >= self.buffer_size:
                        self.append_buffer_to_temp()
                    
                    if current_batch % self.progress_interval == 0:
                        self.save_progress(i, batch_ids)
                        metrics = self.get_performance_metrics()
                        
                        elapsed_time = time.time() - start_time
                        remaining_batches = total_batches - current_batch
                        estimated_remaining_time = (elapsed_time / current_batch) * remaining_batches
                        
                        print(f"\nProgress: Batch {current_batch}/{total_batches} "
                              f"({(current_batch/total_batches)*100:.1f}%)")
                        print(f"Persons processed: {successful_persons}/{total_ids}")
                        print(f"Current Rate: {metrics['current_rate']:.1f} calls/minute")
                        print(f"Estimated time remaining: {estimated_remaining_time/60:.1f} minutes")
                
                elif response.status_code == 429:  # Rate limit hit
                    retry_after = int(response.headers.get('Retry-After', 60))
                    print(f"\nRate limit hit. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    return self.get_person_bulk_enrichment(batch_ids)
                else:
                    self.failed_calls += 1
                    self.had_errors = True
                    print(f"\nError in batch {current_batch}: Status {response.status_code}")
                    
            except Exception as e:
                self.failed_calls += 1
                self.had_errors = True
                print(f"\nError processing batch {current_batch}: {str(e)}")
                continue
        
        return all_results if all_results else None

    def process_person_ids(self, person_ids: List[str]):
        """Process list of person IDs for enrichment"""
        total_persons = len(person_ids)
        start_time = time.time()
        
        # Load previous progress
        progress = self.load_progress()
        start_index = progress.get('last_index', 0)
        
        if start_index > 0:
            print(f"\nResuming from person {start_index + 1}")
            print(f"Previous progress: {self.successful_calls} persons processed")
        
        try:
            enrichment_data = self.get_person_bulk_enrichment(person_ids[start_index:])
            
            if self.buffer:
                self.append_buffer_to_temp()
            
            self.merge_final_data()
            
            # Remove progress file after successful completion
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
                
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
            if self.buffer:
                self.append_buffer_to_temp()
            self.save_progress(start_index + len(self.buffer), person_ids[start_index:start_index + len(self.buffer)])
        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            if self.buffer:
                self.append_buffer_to_temp()
            self.save_progress(start_index + len(self.buffer), person_ids[start_index:start_index + len(self.buffer)])
        finally:
            metrics = self.get_performance_metrics()
            elapsed_time = time.time() - start_time
            
            print(f"\nPerson Enrichment Processing Complete:")
            print("------------------------")
            print(f"Persons Processed: {self.successful_calls}/{total_persons}")
            print(f"API Calls: {self.api_calls_made} (Success: {self.successful_calls}, Failed: {self.failed_calls})")
            print(f"Average Rate: {self.api_calls_made/(elapsed_time/60):.1f} calls/minute")
            print(f"Best Rate Achieved: {metrics['current_rate']:.1f} calls/minute")
            print(f"Average Request Time: {metrics['avg_request_time']:.3f} seconds")
            print(f"Total Time: {elapsed_time/60:.1f} minutes")
            print(f"Status: {'Completed with errors' if self.had_errors else 'Clean'}")
            print(f"Output File: {os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['person_enrichment'])}")
            print("------------------------")

    def process_companies(self, company_ids: List[str]):
        """Not used in this processor"""
        raise NotImplementedError("This processor doesn't process companies directly")