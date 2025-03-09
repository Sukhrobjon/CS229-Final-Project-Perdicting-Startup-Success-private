# /processors/person_processor.py

import requests
import json
import os
import glob
from typing import Dict, Optional, List
import time
from datetime import datetime
from statistics import mean, median
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from processors.base_processor import BaseProcessor
from utils.config import Config
from utils.rate_limiter import RateLimiter

class PersonEnrichmentProcessor(BaseProcessor):
    def __init__(self, api_key: str, batch_size: int = 1000):
        super().__init__(api_key, batch_size)
        
        # Concurrent Processing Configuration
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,    # Match ThreadPoolExecutor workers
            pool_maxsize=20,
            max_retries=0,         # We handle retries ourselves
            pool_block=False
        )
        self.session.mount('https://', adapter)
        self.session.headers.update({
            **self.headers,
            "Content-Type": "application/json"
        })
        
        # Processing Configuration
        self.max_batch_size = Config.PERSON_BATCH_SIZE  # API limit for bulk enrichment
        self.progress_interval = 100
        self.rate_limiter = RateLimiter(
            max_requests=20,        # Optimal concurrent requests
            time_window=1           # 1 second window
        )
        
        # Buffer and Chunk Configuration
        self.buffer_size = 50
        self.chunk_size = 1000
        self.buffer = {}
        
        # Progress Tracking
        self.progress_file = os.path.join(Config.OUTPUT_DIR, "person_enrichment_progress.json")
        self.start_time = time.time()
        self.api_calls_made = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.had_errors = False
        self.total_persons = 0
        self.request_times = []
        
        # Error Tracking
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
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Set up logging with separate error tracking"""
        self.logger = logging.getLogger('person_enrichment')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        
        # Main log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        log_file = os.path.join(
            Config.LOGS_DIR, 
            f'person_enrichment_{timestamp}.log'
        )
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
        
        # Separate error log
        error_log = os.path.join(
            Config.LOGS_DIR, 
            f'person_enrichment_errors_{timestamp}.log'
        )
        error_handler = logging.FileHandler(error_log)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(error_handler)

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
                    
                    self.logger.info(f"Progress loaded: {self.successful_calls} persons processed")
                    return progress
            except Exception as e:
                self.logger.error(f"Error loading progress: {str(e)}")
        return {}

    def save_progress(self, current_index: int):
        """Save current progress"""
        metrics = self.get_performance_metrics()
        
        progress = {
            "last_index": current_index,
            "persons_processed": self.successful_calls,
            "api_calls_made": self.api_calls_made,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "total_persons": self.total_persons,
            "had_errors": self.had_errors,
            "error_counts": self.error_counts,
            "performance_metrics": metrics,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
            self.logger.info(f"Progress saved at index {current_index}")
        except Exception as e:
            self.logger.error(f"Error saving progress: {str(e)}")

    def save_chunk_if_needed(self, persons_processed: int):
        """Create new chunk file at threshold"""
        if persons_processed % self.chunk_size == 0:
            chunk_number = persons_processed // self.chunk_size
            temp_file = os.path.join(Config.OUTPUT_DIR, "person_enrichment_temp.json")
            chunk_file = os.path.join(
                Config.OUTPUT_DIR, 
                f"person_enrichment_chunk_{chunk_number}.json"
            )
            
            try:
                if os.path.exists(temp_file):
                    os.rename(temp_file, chunk_file)
                    self.logger.info(f"Created chunk file: {chunk_file}")
            except Exception as e:
                self.logger.error(f"Error creating chunk file: {str(e)}")

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
            self.logger.info(f"Buffer appended to temp file: {temp_file}")
            
        except Exception as e:
            self.logger.error(f"Error appending to temp file: {str(e)}")
            self.had_errors = True

    
    def get_person_bulk_enrichment(self, person_ids: List[str]) -> Optional[List[Dict]]:
        """Get enriched person data in bulk with enhanced error handling and retries"""
        max_retries = 3
        retry_delay = 1  # seconds
        url = Config.ENDPOINTS['person_enrich']
        
        for attempt in range(max_retries):
            start_time = time.time()
            
            try:
                # Check kill switch conditions
                should_kill, reason = self.rate_limiter.should_kill_process()
                if should_kill:
                    self.logger.error(f"Kill switch activated: {reason}")
                    raise SystemExit(f"Process terminated: {reason}")
                
                self.rate_limiter.wait_if_needed()
                
                payload = {
                    "lookups": [{"id": person_id} for person_id in person_ids],
                    "include": ["EXPERIENCE", "EDUCATION", "DEGREES", "LOCATION_DETAILS"]
                }
                
                response = self.session.post(url, json=payload, timeout=15)
                request_time = time.time() - start_time
                self.request_times.append(request_time)
                self.api_calls_made += 1
                
                if response.status_code == 200:
                    self.successful_calls += 1
                    self.rate_limiter.record_result(success=True)
                    data = response.json()
                    
                    # Process successful response
                    for person_data in data:
                        person_id = person_data.get('person', {}).get('id')
                        if person_id:
                            self.buffer[person_id] = person_data
                            self.total_persons += 1
                    
                    self.logger.info(
                        f"Successfully processed batch of {len(person_ids)} persons. "
                        f"Time: {request_time:.2f}s"
                    )
                    return data
                    
                elif response.status_code >= 500:
                    # Server error - retry
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        self.logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed. "
                            f"Status: {response.status_code}. Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                    
                    # Final attempt failed
                    self._handle_error(response.status_code, request_time)
                    return None
                    
                elif response.status_code == 429:  # Rate limit hit
                    retry_after = int(response.headers.get('Retry-After', 60))
                    self.logger.warning(f"Rate limit hit. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    return self.get_person_bulk_enrichment(person_ids)
                
                else:
                    # Other errors - don't retry
                    self._handle_error(response.status_code, request_time, response.text)
                    return None
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    self.logger.warning(
                        f"Timeout on attempt {attempt + 1}/{max_retries}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
                
                self._handle_error('timeout', time.time() - start_time)
                return None
                
            except Exception as e:
                self._handle_error('other', time.time() - start_time, str(e))
                return None
        
        return None

    def _handle_error(self, error_type, request_time: float, details: str = ""):
        """Centralized error handling"""
        self.failed_calls += 1
        self.had_errors = True
        
        if isinstance(error_type, int):
            if error_type in self.error_counts:
                self.error_counts[error_type]['count'] += 1
            else:
                self.error_counts['other']['count'] += 1
        else:
            self.error_counts[error_type]['count'] += 1
        
        self.rate_limiter.record_result(
            success=False,
            is_timeout=error_type == 'timeout',
            is_rate_limit=error_type == 429
        )
        
        error_msg = (
            f"Error processing batch: {error_type}, "
            f"Time: {request_time:.2f}s"
        )
        if details:
            error_msg += f", Details: {details[:200]}"
        
        self.logger.error(error_msg)

    def get_performance_metrics(self) -> Dict:
        """Calculate current performance metrics"""
        if not self.request_times:
            return {
                "avg_request_time": 0,
                "median_request_time": 0,
                "min_request_time": 0,
                "max_request_time": 0,
                "current_rate": 0,
                "success_rate": 0,
                "error_rate": 0
            }
        
        recent_times = self.request_times[-1000:]  # Last 1000 requests
        rate_limiter_stats = self.rate_limiter.get_stats()
        
        return {
            "avg_request_time": mean(recent_times),
            "median_request_time": median(recent_times),
            "min_request_time": min(recent_times),
            "max_request_time": max(recent_times),
            "current_rate": rate_limiter_stats['current_rate'],
            "success_rate": self.successful_calls / max(self.api_calls_made, 1),
            "error_rate": rate_limiter_stats['error_rate'],
            "total_persons": self.total_persons,
            "rate_limiter_stats": rate_limiter_stats
        }

    def extract_person_ids_from_founders_data(self, founders_file: str) -> List[str]:
        """Extract unique person IDs from founders data file"""
        try:
            with open(founders_file, 'r') as f:
                data = json.load(f)
                person_ids = set()
                
                # Skip metadata entry
                for company_id, company_data in data.items():
                    if company_id != 'metadata' and isinstance(company_data, dict):
                        for founder in company_data.get('founders', []):
                            if founder.get('id'):
                                person_ids.add(founder['id'])
                
                unique_ids = list(person_ids)
                
                self.logger.info(
                    f"Extracted {len(unique_ids)} unique person IDs from founders data"
                )
                return unique_ids
                
        except Exception as e:
            self.logger.error(f"Error extracting person IDs from founders data: {str(e)}")
            return []

    def process_founders_data(self):
        """Main entry point for processing founders data"""
        # Validate configuration
        if not Config.validate_person_enrichment_config():
            raise ValueError("Invalid configuration for person enrichment processing")

        self.logger.info(f"Reading founders data from: {Config.FOUNDERS_DATA_PATH}")
        print(f"\nReading founders data from: {Config.FOUNDERS_DATA_PATH}")
        
        # Extract person IDs from founders data
        person_ids = self.extract_person_ids_from_founders_data(Config.FOUNDERS_DATA_PATH)
        
        if not person_ids:
            error_msg = "No person IDs found in founders data"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        print(f"\nFound {len(person_ids)} unique person IDs to process")
        self.logger.info(f"Found {len(person_ids)} unique person IDs to process")
        
        # Process the extracted person IDs
        self.process_person_ids(person_ids)

    def process_person_ids(self, person_ids: List[str]):
        """Process list of person IDs with concurrent execution"""
        total_persons = len(person_ids)
        self.start_time = time.time()
        persons_processed = 0
        current_index = 0
        
        # Load previous progress
        progress = self.load_progress()
        start_index = progress.get('last_index', 0)
        
        if start_index > 0:
            self.logger.info(f"Resuming from person {start_index + 1}")
            print(f"\nResuming from person {start_index + 1}")
            print(f"Previous progress: {self.successful_calls} persons processed")
        
        try:
            # Process in batches using ThreadPoolExecutor
            batch_size = 100  # Size of batches to submit to thread pool
            
            with ThreadPoolExecutor(max_workers=20) as executor:
                for i in range(start_index, total_persons, batch_size):
                    batch_end = min(i + batch_size, total_persons)
                    current_batch = person_ids[i:batch_end]
                    current_index = batch_end
                    
                    # Split batch into smaller chunks for API limits
                    futures = []
                    for j in range(0, len(current_batch), self.max_batch_size):
                        api_batch = current_batch[j:j + self.max_batch_size]
                        futures.append(
                            executor.submit(self.get_person_bulk_enrichment, api_batch)
                        )
                    
                    # Process completed futures
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                persons_processed += len(result)
                                
                                if len(self.buffer) >= self.buffer_size:
                                    self.append_buffer_to_temp()
                        except Exception as e:
                            self.logger.error(f"Error processing future: {str(e)}")
                    
                    # Save progress and show stats periodically
                    if current_index % self.progress_interval == 0:
                        self._save_and_report_progress(
                            current_index, 
                            total_persons, 
                            persons_processed
                        )
        
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
            self._handle_shutdown(current_index)
        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}")
            self._handle_shutdown(current_index)
        finally:
            self._cleanup_and_finalize()

    def _save_and_report_progress(self, current_index: int, total_persons: int, 
                                persons_processed: int):
        """Save progress and display current statistics"""
        self.save_progress(current_index)
        self.save_chunk_if_needed(persons_processed)
        
        # Calculate and display metrics
        metrics = self.get_performance_metrics()
        elapsed_time = time.time() - self.start_time
        remaining_persons = total_persons - current_index
        estimated_remaining_time = (elapsed_time / current_index) * remaining_persons if current_index > 0 else 0
        
        status = (
            f"\nProgress: {current_index}/{total_persons} "
            f"({(current_index/total_persons)*100:.1f}%)\n"
            f"Current Rate: {metrics['current_rate']:.1f} requests/second\n"
            f"Success Rate: {metrics['success_rate']:.2%}\n"
            f"Total Persons Processed: {persons_processed}\n"
            f"Avg Request Time: {metrics['avg_request_time']:.3f} seconds\n"
            f"Est. time remaining: {estimated_remaining_time/60:.1f} minutes"
        )
        
        self.logger.info(status)
        print(status)
        
        if self.failed_calls > 0:
            self.print_error_summary()

    def _handle_shutdown(self, current_index: int):
        """Handle graceful shutdown"""
        try:
            if self.buffer:
                print("\nSaving current buffer...")
                self.append_buffer_to_temp()
            
            print("\nSaving final progress...")
            self.save_progress(current_index)
            
            # Perform emergency merge
            self.emergency_merge_files()
            
            # Print final statistics
            self._print_final_stats()
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            # Try emergency merge even if shutdown handling fails
            try:
                self.emergency_merge_files()
            except Exception as merge_error:
                self.logger.error(f"Emergency merge failed: {str(merge_error)}")

    def _cleanup_and_finalize(self):
        """Final cleanup and merging of files"""
        try:
            if self.buffer:
                self.append_buffer_to_temp()
            
            self.merge_final_data()
            
            # Remove progress file after successful completion
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
            
            self._print_final_stats()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            self.emergency_merge_files()

    def _print_final_stats(self):
        """Print final processing statistics"""
        elapsed_time = time.time() - self.start_time
        metrics = self.get_performance_metrics()
        
        summary = (
            "\nProcessing Complete:\n"
            "-------------------\n"
            f"Total Runtime: {elapsed_time/60:.1f} minutes\n"
            f"Persons Processed: {self.successful_calls}\n"
            f"Success Rate: {metrics['success_rate']:.2%}\n"
            f"Average Rate: {self.successful_calls/(elapsed_time):.1f} persons/second\n"
            f"Total API Calls: {self.api_calls_made}\n"
            f"Failed Calls: {self.failed_calls}\n"
            f"Status: {'Completed with errors' if self.had_errors else 'Clean'}\n"
            f"Output File: {Config.get_output_path('person_enrichment')}"
        )
        
        print(summary)
        self.logger.info(summary)
        
        if self.had_errors:
            self.print_error_summary()

    def print_error_summary(self):
        """Print summary of errors encountered"""
        if self.failed_calls > 0 or self.had_errors:
            error_summary = "\nError Summary:\n-------------\n"
            for status, data in self.error_counts.items():
                if data['count'] > 0:
                    error_summary += f"{data['description']} ({status}): {data['count']} occurrences\n"
            error_summary += f"Error details logged to: {os.path.join(Config.LOGS_DIR)}\n-------------"
            
            self.logger.warning(error_summary)
            print(error_summary)

    # Required abstract method implementations
    def save_batch(self):
        """Implementation of abstract method from BaseProcessor"""
        self.append_buffer_to_temp()

    def process_companies(self, company_ids: List[str]):
        """Not used in this processor"""
        raise NotImplementedError("This processor doesn't process companies directly")

    def emergency_merge_files(self):
        """Emergency merge of all files in case of unexpected exit"""
        try:
            self.logger.info("Attempting emergency file merge...")
            print("\nAttempting emergency file merge...")
            
            final_data = {}
            
            # Merge chunk files
            chunk_pattern = os.path.join(Config.OUTPUT_DIR, "person_enrichment_chunk_*.json")
            chunk_files = sorted(glob.glob(chunk_pattern))
            
            if chunk_files:
                self.logger.info(f"Found {len(chunk_files)} chunk files to merge")
                for chunk_file in chunk_files:
                    try:
                        with open(chunk_file, 'r') as f:
                            chunk_data = json.load(f)
                            final_data.update(chunk_data)
                        os.remove(chunk_file)
                    except Exception as e:
                        self.logger.error(f"Error processing chunk file {chunk_file}: {str(e)}")
            
            # Merge temp file if exists
            temp_file = os.path.join(Config.OUTPUT_DIR, "person_enrichment_temp.json")
            if os.path.exists(temp_file):
                try:
                    with open(temp_file, 'r') as f:
                        temp_data = json.load(f)
                        final_data.update(temp_data)
                    os.remove(temp_file)
                except Exception as e:
                    self.logger.error(f"Error processing temp file: {str(e)}")
            
            # Save merged data
            output_file = Config.get_output_path('person_enrichment')
            with open(output_file, 'w') as f:
                json.dump(final_data, f, indent=2)
            
            self.logger.info("Emergency merge completed successfully")
            print(f"\nEmergency merge completed. Data saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error during emergency merge: {str(e)}")
            print(f"\nError during emergency merge: {str(e)}")
