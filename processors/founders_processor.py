import requests
import json
import os
import glob
from typing import Dict, Optional, List
import time
from datetime import datetime
from processors.base_processor import BaseProcessor
from utils.config import Config
from utils.rate_limiter import RateLimiter

class FoundersProcessor(BaseProcessor):
    def __init__(self, api_key: str, batch_size: int = 1000):
        super().__init__(api_key, batch_size)
        self.progress_interval = 500  # Show progress every 500 companies
        self.rate_limiter = RateLimiter(max_requests=10000)  # Increased rate limit
        self.api_calls_made = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.had_errors = False
        
        # Buffer configuration
        self.buffer_size = 50    # Companies in memory before append
        self.chunk_size = 10000  # Companies per chunk file
        self.buffer = {}         # Current companies in memory
        self.total_founders = 0  # Track total founders found

    def get_founders(self, company_id: str) -> Optional[Dict]:
        """Get company founders"""
        self.rate_limiter.wait_if_needed()
        
        url = f"https://data.api.aviato.co/company/{company_id}/founders"
        params = {
            "perPage": 100,  # Use maximum efficient page size
            "page": 0
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            self.api_calls_made += 1
            
            if response.status_code == 429:  # Rate limit hit
                print("Rate limit reached, waiting...")
                time.sleep(int(response.headers.get('Retry-After', 5)))
                return self.get_founders(company_id)
            
            if response.status_code == 200:
                self.successful_calls += 1
                data = response.json()
                if data and 'founders' in data:
                    self.total_founders += len(data['founders'])
                return data
            else:
                self.failed_calls += 1
                self.had_errors = True
                print(f"Error fetching founders for {company_id}: {response.status_code}")
                return None
                
        except Exception as e:
            self.failed_calls += 1
            self.had_errors = True
            print(f"Exception fetching founders for {company_id}: {str(e)}")
            return None

    def save_progress(self, current_index: int, company_id: str):
        """Save current progress"""
        progress = {
            "last_index": current_index,
            "last_company_id": company_id,
            "companies_processed": self.successful_calls,
            "api_calls_made": self.api_calls_made,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "total_founders": self.total_founders,
            "had_errors": self.had_errors,
            "last_updated": datetime.now().isoformat()
        }
        
        progress_file = os.path.join(Config.OUTPUT_DIR, "founders_progress.json")
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def load_progress(self) -> Dict:
        """Load previous progress"""
        progress_file = os.path.join(Config.OUTPUT_DIR, "founders_progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    self.api_calls_made = progress.get('api_calls_made', 0)
                    self.successful_calls = progress.get('successful_calls', 0)
                    self.failed_calls = progress.get('failed_calls', 0)
                    self.total_founders = progress.get('total_founders', 0)
                    self.had_errors = progress.get('had_errors', False)
                    return progress
            except Exception as e:
                print(f"Error loading progress: {e}")
        return {}

    def save_batch(self):
        """Implementation of abstract method from BaseProcessor"""
        self.append_buffer_to_temp()

    def append_buffer_to_temp(self):
        """Append current buffer to temp file"""
        if not self.buffer:
            return
            
        temp_file = os.path.join(Config.OUTPUT_DIR, "founders_temp.json")
        
        try:
            # Read existing data if file exists
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    existing_data = json.load(f)
                existing_data.update(self.buffer)
                data_to_save = existing_data
            else:
                data_to_save = self.buffer

            # Write updated data
            with open(temp_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            self.buffer = {}  # Clear buffer
            
        except Exception as e:
            print(f"Error appending to temp file: {str(e)}")
            self.had_errors = True

    def save_chunk_if_needed(self, companies_processed: int):
        """Create new chunk file at 10k threshold"""
        if companies_processed % self.chunk_size == 0:
            chunk_number = companies_processed // self.chunk_size
            temp_file = os.path.join(Config.OUTPUT_DIR, "founders_temp.json")
            chunk_file = os.path.join(Config.OUTPUT_DIR, f"founders_chunk_{chunk_number}.json")
            
            if os.path.exists(temp_file):
                os.rename(temp_file, chunk_file)
                print(f"\nCreated chunk file: {chunk_file}")

    def merge_final_data(self):
        """Merge all data into final file"""
        final_data = {}
        
        try:
            # First, check for chunk files (if we had >10k companies)
            chunk_pattern = os.path.join(Config.OUTPUT_DIR, "founders_chunk_*.json")
            chunk_files = sorted(glob.glob(chunk_pattern))
            
            # Process chunk files if they exist
            if chunk_files:
                print("\nMerging chunk files...")
                for i, chunk_file in enumerate(chunk_files, 1):
                    print(f"Processing chunk file {i}/{len(chunk_files)}")
                    with open(chunk_file, 'r') as f:
                        chunk_data = json.load(f)
                        final_data.update(chunk_data)
                    os.remove(chunk_file)  # Clean up chunk file
            
            # Check for temp file (for remaining companies or if total was <10k)
            temp_file = os.path.join(Config.OUTPUT_DIR, "founders_temp.json")
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    temp_data = json.load(f)
                    final_data.update(temp_data)
                os.remove(temp_file)  # Clean up temp file
            
            # Save final merged file
            final_file = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['founders'])
            with open(final_file, 'w') as f:
                json.dump(final_data, f, indent=2)
            
            print(f"\nAll data merged into: {final_file}")
            print(f"Total companies in final file: {len(final_data)}")
            
        except Exception as e:
            print(f"Error during final merge: {str(e)}")
            self.had_errors = True

    def get_all_founder_ids(self) -> List[str]:
        """Extract all founder IDs from the collected data"""
        founder_ids = set()  # Use set for efficiency
        for company_data in self.data.values():
            for founder in company_data.get('founders', []):
                founder_id = founder.get('id')
                if founder_id:
                    founder_ids.add(founder_id)
        return list(founder_ids)

    def process_companies(self, company_ids: List[str]):
        """Process list of companies for founders"""
        total_companies = len(company_ids)
        start_time = time.time()
        companies_processed = 0
        
        # Load previous progress
        progress = self.load_progress()
        start_index = progress.get('last_index', 0)
        
        if start_index > 0:
            print(f"\nResuming from company {start_index + 1}")
            print(f"Previous progress: {self.successful_calls} companies processed")
            print(f"Total founders found so far: {self.total_founders}")
        
        print(f"\nStarting Founders processing for {total_companies} companies...")
        print(f"Buffer size: {self.buffer_size} companies")
        print(f"Chunk size: {self.chunk_size} companies")

        try:
            for i, company_id in enumerate(company_ids[start_index:], start_index + 1):
                try:
                    founders_data = self.get_founders(company_id)
                    if founders_data:
                        self.buffer[company_id] = founders_data
                        companies_processed += 1
                        
                        # Append to temp file when buffer is full
                        if len(self.buffer) >= self.buffer_size:
                            self.append_buffer_to_temp()
                            
                        # Create new chunk file at 10k threshold
                        self.save_chunk_if_needed(companies_processed)
                        
                        # Save progress periodically
                        if i % self.progress_interval == 0:
                            self.save_progress(i, company_id)
                    
                    # Show progress every 500 companies
                    if i % self.progress_interval == 0:
                        elapsed_time = time.time() - start_time
                        avg_time_per_company = elapsed_time / (i - start_index)
                        remaining_companies = total_companies - i
                        estimated_remaining_time = remaining_companies * avg_time_per_company
                        current_rate = self.api_calls_made / (elapsed_time / 60)
                        
                        print(f"\nProgress: {i}/{total_companies} ({(i/total_companies)*100:.1f}%)")
                        print(f"API Calls: {self.api_calls_made} (Success: {self.successful_calls}, Failed: {self.failed_calls})")
                        print(f"Total Founders Found: {self.total_founders}")
                        print(f"Current Rate: {current_rate:.1f} calls/minute")
                        print(f"Est. time remaining: {estimated_remaining_time/60:.1f} minutes")
                    
                except Exception as e:
                    print(f"Error processing company {company_id}: {str(e)}")
                    self.save_progress(i, company_id)  # Save progress on error
                    continue

            # Save any remaining data in buffer
            if self.buffer:
                self.append_buffer_to_temp()

            # Merge all data into final file
            self.merge_final_data()

            # Clear progress file after successful completion
            progress_file = os.path.join(Config.OUTPUT_DIR, "founders_progress.json")
            if os.path.exists(progress_file):
                os.remove(progress_file)

        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
            if self.buffer:
                self.append_buffer_to_temp()
            self.save_progress(i, company_id)
        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            if self.buffer:
                self.append_buffer_to_temp()
            self.save_progress(i, company_id)
        finally:
            # Final summary
            elapsed_time = time.time() - start_time
            print(f"\nFounders Processing Complete:")
            print("------------------------")
            print(f"Companies Processed: {self.successful_calls}/{total_companies}")
            print(f"API Calls: {self.api_calls_made} (Success: {self.successful_calls}, Failed: {self.failed_calls})")
            print(f"Total Founders Found: {self.total_founders}")
            print(f"Total time: {elapsed_time/60:.1f} minutes")
            print(f"Average rate: {self.api_calls_made/(elapsed_time/60):.1f} calls/minute")
            print(f"Status: {'Completed with errors' if self.had_errors else 'Clean'}")
            print("------------------------")