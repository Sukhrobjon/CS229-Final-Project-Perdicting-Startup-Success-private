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

class FoundersProcessor(BaseProcessor):
    def __init__(self, api_key: str, batch_size: int = 1000):
        super().__init__(api_key, batch_size)
        self.progress_interval = 500
        self.rate_limiter = RateLimiter(max_requests=10000)
        self.session = requests.Session()
        
        # Progress tracking
        self.progress_file = os.path.join(Config.OUTPUT_DIR, "founders_progress.json")
        
        # API tracking
        self.api_calls_made = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.had_errors = False
        
        # Buffer configuration
        self.buffer_size = 50
        self.chunk_size = 10000
        self.buffer = {}
        
        # Performance tracking
        self.total_founders = 0
        self.request_times = []
        self.last_request_time = time.time()
        
        # Error tracking
        self.error_counts = {
            400: {'count': 0, 'description': 'Bad Request'},
            401: {'count': 0, 'description': 'Unauthorized'},
            403: {'count': 0, 'description': 'Forbidden'},
            404: {'count': 0, 'description': 'Not Found'},
            429: {'count': 0, 'description': 'Rate Limited'},
            500: {'count': 0, 'description': 'Server Error'},
            'timeout': {'count': 0, 'description': 'Request Timeout'},
            'other': {'count': 0, 'description': 'Other Errors'}
        }
        
        # Initialize session
        self.session.headers.update(self.headers)

    def get_founders(self, company_id: str) -> Optional[Dict]:
        """Get company founders - Optimized version"""
        start_time = time.time()
        self.rate_limiter.wait_if_needed()
        
        url = f"https://data.api.aviato.co/company/{company_id}/founders"
        params = {
            "perPage": 100,
            "page": 0
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            request_time = time.time() - start_time
            self.request_times.append(request_time)
            self.api_calls_made += 1
            
            if response.status_code == 200:
                self.successful_calls += 1
                data = response.json()
                if data and 'founders' in data:
                    self.total_founders += len(data['founders'])
                return data
            else:
                self.failed_calls += 1
                self.had_errors = True
                
                # Track error type
                if response.status_code in self.error_counts:
                    self.error_counts[response.status_code]['count'] += 1
                else:
                    self.error_counts['other']['count'] += 1
                    
                # Log detailed error for debugging
                if response.status_code >= 400:
                    error_details = f"Company {company_id}: Status {response.status_code}"
                    try:
                        error_details += f", Response: {response.json()}"
                    except:
                        error_details += f", Response: {response.text[:200]}"
                    
                    with open(os.path.join(Config.OUTPUT_DIR, "founders_errors.log"), 'a') as f:
                        f.write(f"{datetime.now().isoformat()}: {error_details}\n")
                
                return None
                    
        except requests.exceptions.Timeout:
            self.failed_calls += 1
            self.had_errors = True
            self.error_counts['timeout']['count'] += 1
            return None
        except Exception as e:
            self.failed_calls += 1
            self.had_errors = True
            self.error_counts['other']['count'] += 1
            
            # Log unexpected errors
            with open(os.path.join(Config.OUTPUT_DIR, "founders_errors.log"), 'a') as f:
                f.write(f"{datetime.now().isoformat()}: Company {company_id}: Unexpected error: {str(e)}\n")
            return None

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
            
        recent_times = self.request_times[-1000:]  # Look at last 1000 requests
        avg_time = mean(recent_times)
        
        return {
            "avg_request_time": avg_time,
            "median_request_time": median(recent_times),
            "min_request_time": min(recent_times),
            "max_request_time": max(recent_times),
            "current_rate": 60 / avg_time if avg_time > 0 else 0
        }

    def save_progress(self, current_index: int, company_id: str):
        """Save current progress"""
        metrics = self.get_performance_metrics()
        
        progress = {
            "last_index": current_index,
            "last_company_id": company_id,
            "companies_processed": self.successful_calls,
            "api_calls_made": self.api_calls_made,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "total_founders": self.total_founders,
            "had_errors": self.had_errors,
            "error_counts": self.error_counts,
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
                    self.total_founders = progress.get('total_founders', 0)
                    self.had_errors = progress.get('had_errors', False)
                    self.error_counts = progress.get('error_counts', self.error_counts)
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

    def save_chunk_if_needed(self, companies_processed: int):
        """Create new chunk file at 10k threshold"""
        if companies_processed % self.chunk_size == 0:
            chunk_number = companies_processed // self.chunk_size
            temp_file = os.path.join(Config.OUTPUT_DIR, "founders_temp.json")
            chunk_file = os.path.join(Config.OUTPUT_DIR, f"founders_chunk_{chunk_number}.json")
            
            if os.path.exists(temp_file):
                os.rename(temp_file, chunk_file)

    def merge_final_data(self):
        """Merge all data into final file"""
        final_data = {}
        
        try:
            chunk_pattern = os.path.join(Config.OUTPUT_DIR, "founders_chunk_*.json")
            chunk_files = sorted(glob.glob(chunk_pattern))
            
            if chunk_files:
                print("\nMerging chunk files...")
                for i, chunk_file in enumerate(chunk_files, 1):
                    with open(chunk_file, 'r') as f:
                        chunk_data = json.load(f)
                        final_data.update(chunk_data)
                    os.remove(chunk_file)
            
            temp_file = os.path.join(Config.OUTPUT_DIR, "founders_temp.json")
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    temp_data = json.load(f)
                    final_data.update(temp_data)
                os.remove(temp_file)
            
            final_file = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['founders'])
            with open(final_file, 'w') as f:
                json.dump(final_data, f, indent=2)
            
            print("\nFinal Output Summary:")
            print("------------------------")
            print(f"Output file: {final_file}")
            print(f"Total companies: {len(final_data)}")
            print(f"Total founders: {self.total_founders}")
            print("------------------------")
            
        except Exception as e:
            print(f"Error during final merge: {str(e)}")
            self.had_errors = True

    def print_error_summary(self):
        """Print summary of errors encountered"""
        if self.failed_calls > 0:
            print("\nError Summary:")
            print("-------------")
            for status, data in self.error_counts.items():
                if data['count'] > 0:
                    print(f"{data['description']} ({status}): {data['count']} occurrences")
            print(f"Error details logged to: {os.path.join(Config.OUTPUT_DIR, 'founders_errors.log')}")
            print("-------------")

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

        try:
            for i, company_id in enumerate(company_ids[start_index:], start_index + 1):
                founders_data = self.get_founders(company_id)
                if founders_data:
                    self.buffer[company_id] = founders_data
                    companies_processed += 1
                    
                    if len(self.buffer) >= self.buffer_size:
                        self.append_buffer_to_temp()
                    
                    self.save_chunk_if_needed(companies_processed)
                    
                    # Show progress every 500 companies
                    if i % self.progress_interval == 0:
                        self.save_progress(i, company_id)
                        metrics = self.get_performance_metrics()
                        elapsed_time = time.time() - start_time
                        remaining_companies = total_companies - i
                        estimated_remaining_time = (elapsed_time / i) * remaining_companies
                        
                        print(f"\nProgress: {i}/{total_companies} ({(i/total_companies)*100:.1f}%)")
                        print(f"API Calls: {self.api_calls_made} (Success: {self.successful_calls}, Failed: {self.failed_calls})")
                        print(f"Total Founders Found: {self.total_founders}")
                        print(f"Current Rate: {metrics['current_rate']:.1f} calls/minute")
                        print(f"Avg Request Time: {metrics['avg_request_time']:.3f} seconds")
                        print(f"Est. time remaining: {estimated_remaining_time/60:.1f} minutes")

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
            self.save_progress(i, company_id)
        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            if self.buffer:
                self.append_buffer_to_temp()
            self.save_progress(i, company_id)
        finally:
            metrics = self.get_performance_metrics()
            elapsed_time = time.time() - start_time
            
            print(f"\nFounders Processing Complete:")
            print("------------------------")
            print(f"Companies Processed: {self.successful_calls}/{total_companies}")
            print(f"API Calls: {self.api_calls_made} (Success: {self.successful_calls}, Failed: {self.failed_calls})")
            print(f"Total Founders Found: {self.total_founders}")
            print(f"Average Rate: {self.api_calls_made/(elapsed_time/60):.1f} calls/minute")
            print(f"Best Rate Achieved: {metrics['current_rate']:.1f} calls/minute")
            print(f"Average Request Time: {metrics['avg_request_time']:.3f} seconds")
            print(f"Total Time: {elapsed_time/60:.1f} minutes")
            print(f"Status: {'Completed with errors' if self.had_errors else 'Clean'}")
            print(f"Output File: {os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['founders'])}")
            print("------------------------")
            
            # Print error summary if there were any failures
            if self.failed_calls > 0:
                self.print_error_summary()