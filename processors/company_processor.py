# /processors/company_processor.py

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

class CompanyEnrichmentProcessor(BaseProcessor):
    def __init__(self, api_key: str, batch_size: int = 1000):
        super().__init__(api_key, batch_size)
        
        # Configuration - Optimal Medium Settings
        self.progress_interval = 1000
        self.rate_limiter = RateLimiter(
            max_requests=20,    # Optimal concurrent requests
            time_window=1       # 1 second window
        )
        
        # Initialize session with concurrent settings
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,    # Allow 20 concurrent connections
            pool_maxsize=20,        # Maximum 20 concurrent connections
            max_retries=0,          # We handle retries ourselves
            pool_block=False        # Don't block when pool is full
        )
        self.session.mount('https://', adapter)
        self.session.headers.update(self.headers)
        
        # Progress tracking
        self.progress_file = os.path.join(Config.OUTPUT_DIR, "enrichment_progress.json")
        
        # API tracking
        self.api_calls_made = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.had_errors = False
        
        # Buffer configuration
        self.buffer_size = 1000
        self.chunk_size = 1000
        self.buffer = {}
        
        # Performance tracking
        self.request_times = []
        self.last_request_time = time.time()
        self.start_time = time.time()
        
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
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Set up logging with enhanced error tracking"""
        self.logger = logging.getLogger('company_enrichment_processor')
        self.logger.setLevel(logging.INFO)
        
        log_dir = os.path.join(Config.OUTPUT_DIR, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Main log file
        log_file = os.path.join(
            log_dir, 
            f'company_enrichment_{datetime.now().strftime("%Y%m%d_%H%M")}.log'
        )
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
        
        # Separate error log
        error_log = os.path.join(
            log_dir, 
            f'company_enrichment_errors_{datetime.now().strftime("%Y%m%d_%H%M")}.log'
        )
        error_handler = logging.FileHandler(error_log)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(error_handler)

    def filter_company_data(self, data: Dict) -> Dict:
        """Filter company data to keep only desired fields"""
        desired_fields = {
            "id", "name", "country", "region", "locality", "URLs", 
            "industryList", "alternateNames", "description", "tagline",
            "founded", "headcount", "financingStatus", "ownershipStatus",
            "status", "latestDealType", "latestDealDate", "latestDealAmount",
            "investorCount", "legalName", "leadInvestorCount", "totalFunding",
            "fundingRoundCount", "lastRoundValuation", "targetMarketList",
            "isAcquired", "isGovernment", "isNonProfit", "isExited",
            "isShutDown", "customerTypes"
        }
        return {k: v for k, v in data.items() if k in desired_fields}

    def get_company_enrichment(self, company_id: str) -> Optional[Dict]:
        """Get enriched company data with enhanced error handling and retry logic"""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            start_time = time.time()
            
            try:
                # Check kill switch conditions
                should_kill, reason = self.rate_limiter.should_kill_process()
                if should_kill:
                    self.logger.error(f"Kill switch activated: {reason}")
                    raise SystemExit(f"Process terminated: {reason}")
                
                self.rate_limiter.wait_if_needed()
                
                url = "https://data.api.aviato.co/company/enrich"
                params = {"id": company_id}
                
                response = self.session.get(url, params=params, timeout=15)
                request_time = time.time() - start_time
                self.request_times.append(request_time)
                self.api_calls_made += 1
                
                if response.status_code == 200:
                    self.successful_calls += 1
                    self.rate_limiter.record_result(success=True)
                    data = response.json()
                    
                    filtered_data = self.filter_company_data(data)
                    self.logger.info(
                        f"Successfully processed company {company_id}. "
                        f"Time: {request_time:.2f}s"
                    )
                    return {company_id: filtered_data}
                    
                elif response.status_code >= 500:
                    # Server error - retry
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        self.logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for company {company_id}. "
                            f"Status: {response.status_code}. Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                    
                    # Final attempt failed
                    self.failed_calls += 1
                    self.had_errors = True
                    self.error_counts[500]['count'] += 1
                    self.rate_limiter.record_result(success=False)
                    
                    self.logger.error(
                        f"All retries failed for company {company_id}. "
                        f"Status: {response.status_code}"
                    )
                    return None
                
                else:
                    # Other errors - don't retry
                    self.failed_calls += 1
                    self.had_errors = True
                    
                    if response.status_code in self.error_counts:
                        self.error_counts[response.status_code]['count'] += 1
                    else:
                        self.error_counts['other']['count'] += 1
                    
                    self.rate_limiter.record_result(
                        success=False,
                        is_rate_limit=response.status_code == 429
                    )
                    
                    error_details = (
                        f"Company {company_id}: Status {response.status_code}, "
                        f"Time: {request_time:.2f}s, "
                        f"Response: {response.text[:200]}"
                    )
                    self.logger.error(error_details)
                    
                    return None
                    
            except (requests.exceptions.Timeout, ConnectionResetError) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for company {company_id}. "
                        f"Error: {str(e)}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
                
                self.failed_calls += 1
                self.had_errors = True
                self.error_counts['timeout' if isinstance(e, requests.exceptions.Timeout) else 'other']['count'] += 1
                self.rate_limiter.record_result(success=False, is_timeout=True)
                
                self.logger.error(
                    f"Final attempt failed for company {company_id}. "
                    f"Error: {str(e)}. Time: {time.time() - start_time:.2f}s"
                )
                return None
                
            except Exception as e:
                self.failed_calls += 1
                self.had_errors = True
                self.error_counts['other']['count'] += 1
                self.rate_limiter.record_result(success=False)
                
                self.logger.error(
                    f"Unexpected error for company {company_id}: {str(e)}. "
                    f"Time: {time.time() - start_time:.2f}s"
                )
                return None
        
        return None
    

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
            
        recent_times = self.request_times[-1000:]  # Look at last 1000 requests
        avg_time = mean(recent_times)
        
        # Get rate limiter stats
        rate_limiter_stats = self.rate_limiter.get_stats()
        
        return {
            "avg_request_time": avg_time,
            "median_request_time": median(recent_times),
            "min_request_time": min(recent_times),
            "max_request_time": max(recent_times),
            "current_rate": rate_limiter_stats['current_rate'],
            "success_rate": self.successful_calls / max(self.api_calls_made, 1),
            "error_rate": rate_limiter_stats['error_rate'],
            "rate_limiter_stats": rate_limiter_stats
        }

    def save_progress(self, current_index: int, company_id: str):
        """Save current progress with enhanced metrics"""
        metrics = self.get_performance_metrics()
        rate_limiter_stats = self.rate_limiter.get_stats()
        
        progress = {
            "last_index": current_index,
            "last_company_id": company_id,
            "companies_processed": self.successful_calls,
            "api_calls_made": self.api_calls_made,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "had_errors": self.had_errors,
            "error_counts": self.error_counts,
            "performance_metrics": metrics,
            "rate_limiter_stats": rate_limiter_stats,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
            self.logger.info(f"Progress saved at index {current_index}")
        except Exception as e:
            self.logger.error(f"Error saving progress: {str(e)}")

    def load_progress(self) -> Dict:
        """Load previous progress"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.api_calls_made = progress.get('api_calls_made', 0)
                    self.successful_calls = progress.get('successful_calls', 0)
                    self.failed_calls = progress.get('failed_calls', 0)
                    self.had_errors = progress.get('had_errors', False)
                    self.error_counts = progress.get('error_counts', self.error_counts)
                    
                    self.logger.info(f"Progress loaded: {self.successful_calls} companies processed")
                    return progress
            except Exception as e:
                self.logger.error(f"Error loading progress: {str(e)}")
        return {}

    def save_batch(self):
        """Implementation of abstract method from BaseProcessor"""
        try:
            self.append_buffer_to_temp()
            self.logger.info(f"Batch saved: {len(self.buffer)} companies")
        except Exception as e:
            self.logger.error(f"Error saving batch: {str(e)}")

    def append_buffer_to_temp(self):
        """Append current buffer to temp file"""
        if not self.buffer:
            return
            
        temp_file = os.path.join(Config.OUTPUT_DIR, "enrichment_temp.json")
        
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

    def save_chunk_if_needed(self, companies_processed: int):
        """Create new chunk file at threshold"""
        if companies_processed % self.chunk_size == 0:
            chunk_number = companies_processed // self.chunk_size
            temp_file = os.path.join(Config.OUTPUT_DIR, "enrichment_temp.json")
            chunk_file = os.path.join(Config.OUTPUT_DIR, f"enrichment_chunk_{chunk_number}.json")
            
            try:
                if os.path.exists(temp_file):
                    os.rename(temp_file, chunk_file)
                    self.logger.info(f"Created chunk file: {chunk_file}")
            except Exception as e:
                self.logger.error(f"Error creating chunk file: {str(e)}")

    def merge_final_data(self):
        """Merge all data into final file with metadata"""
        final_data = {
            "metadata": {
                "processed_companies": self.successful_calls,
                "processing_time": f"{(time.time() - self.start_time)/60:.1f} minutes",
                "success_rate": f"{(self.successful_calls/max(self.api_calls_made, 1))*100:.1f}%",
                "timestamp": datetime.now().isoformat(),
                "source_file": Config.COMPANIES_FILE,
                "api_calls": {
                    "total": self.api_calls_made,
                    "successful": self.successful_calls,
                    "failed": self.failed_calls
                },
                "error_counts": self.error_counts
            }
        }
        
        try:
            # Merge chunk files
            chunk_pattern = os.path.join(Config.OUTPUT_DIR, "enrichment_chunk_*.json")
            chunk_files = sorted(glob.glob(chunk_pattern))
            
            if chunk_files:
                self.logger.info("\nMerging chunk files...")
                for i, chunk_file in enumerate(chunk_files, 1):
                    with open(chunk_file, 'r') as f:
                        chunk_data = json.load(f)
                        for key, value in chunk_data.items():
                            if key != 'metadata':
                                final_data[key] = value
                    os.remove(chunk_file)
                    self.logger.info(f"Processed chunk file {i}/{len(chunk_files)}")
            
            # Merge temp file
            temp_file = os.path.join(Config.OUTPUT_DIR, "enrichment_temp.json")
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    temp_data = json.load(f)
                    for key, value in temp_data.items():
                        if key != 'metadata':
                            final_data[key] = value
                os.remove(temp_file)
            
            # Save final file
            final_file = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['company_enrichment'])
            with open(final_file, 'w') as f:
                json.dump(final_data, f, indent=2)
            
            # Print summary
            summary = {
                "Output file": final_file,
                "Total companies": len(final_data) - 1,  # Subtract 1 for metadata
                "Success rate": f"{(self.successful_calls/max(self.api_calls_made, 1))*100:.1f}%",
                "Processing time": f"{(time.time() - self.start_time)/60:.1f} minutes"
            }
            
            print("\nFinal Output Summary:")
            print("------------------------")
            for key, value in summary.items():
                print(f"{key}: {value}")
            print("------------------------")
            
        except Exception as e:
            self.logger.error(f"Error during final merge: {str(e)}")
            self.had_errors = True
            print(f"\nError during final merge: {str(e)}")

    def emergency_merge_files(self):
        """Emergency merge of all files in case of unexpected exit"""
        try:
            self.logger.info("Attempting emergency file merge...")
            print("\nAttempting emergency file merge...")
            
            final_data = {
                "metadata": {
                    "emergency_merge": True,
                    "processed_companies": self.successful_calls,
                    "processing_time": f"{(time.time() - self.start_time)/60:.1f} minutes",
                    "success_rate": f"{(self.successful_calls/max(self.api_calls_made, 1))*100:.1f}%",
                    "timestamp": datetime.now().isoformat(),
                    "source_file": Config.COMPANIES_FILE,
                    "api_calls": {
                        "total": self.api_calls_made,
                        "successful": self.successful_calls,
                        "failed": self.failed_calls
                    },
                    "error_counts": self.error_counts
                }
            }
            
            # Merge all chunk files
            chunk_pattern = os.path.join(Config.OUTPUT_DIR, "enrichment_chunk_*.json")
            chunk_files = sorted(glob.glob(chunk_pattern))
            
            if chunk_files:
                self.logger.info(f"Found {len(chunk_files)} chunk files to merge")
                print(f"Found {len(chunk_files)} chunk files to merge")
                
                for chunk_file in chunk_files:
                    try:
                        with open(chunk_file, 'r') as f:
                            chunk_data = json.load(f)
                            for key, value in chunk_data.items():
                                if key != 'metadata':
                                    final_data[key] = value
                        os.remove(chunk_file)
                    except Exception as e:
                        self.logger.error(f"Error processing chunk file {chunk_file}: {str(e)}")
            
            # Check for temp file
            temp_file = os.path.join(Config.OUTPUT_DIR, "enrichment_temp.json")
            if os.path.exists(temp_file):
                try:
                    with open(temp_file, 'r') as f:
                        temp_data = json.load(f)
                        for key, value in temp_data.items():
                            if key != 'metadata':
                                final_data[key] = value
                    os.remove(temp_file)
                except Exception as e:
                    self.logger.error(f"Error processing temp file: {str(e)}")
            
            # Save merged data
            final_file = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['company_enrichment'])
            with open(final_file, 'w') as f:
                json.dump(final_data, f, indent=2)
            
            self.logger.info("Emergency merge completed successfully")
            print("\nEmergency merge completed successfully")
            
            # Print summary
            summary = {
                "Output file": final_file,
                "Total companies": len(final_data) - 1,
                "Success rate": f"{(self.successful_calls/max(self.api_calls_made, 1))*100:.1f}%",
                "Processing time": f"{(time.time() - self.start_time)/60:.1f} minutes"
            }
            
            print("\nEmergency Merge Summary:")
            print("------------------------")
            for key, value in summary.items():
                print(f"{key}: {value}")
            print("------------------------")
            
        except Exception as e:
            self.logger.error(f"Error during emergency merge: {str(e)}")
            print(f"\nError during emergency merge: {str(e)}")

    def handle_shutdown(self, current_index: int, last_company_id: str):
        """Handle graceful shutdown with emergency merge"""
        try:
            if self.buffer:
                print("\nSaving current buffer...")
                self.append_buffer_to_temp()
            
            print("\nSaving final progress...")
            self.save_progress(current_index, last_company_id)
            
            # Perform emergency merge
            self.emergency_merge_files()
            
            # Get final statistics
            elapsed_time = time.time() - self.start_time
            print("\nProcessing Summary at Shutdown:")
            print("------------------------------")
            print(f"Total Runtime: {elapsed_time/60:.1f} minutes")
            print(f"Companies Processed: {self.successful_calls}")
            print(f"Success Rate: {(self.successful_calls/max(self.api_calls_made, 1))*100:.1f}%")
            print(f"Average Rate: {self.successful_calls/elapsed_time:.1f} companies/second")
            
            if self.had_errors:
                self.print_error_summary()
                
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            print(f"Error during shutdown: {str(e)}")
            
            # Try emergency merge even if shutdown handling fails
            try:
                self.emergency_merge_files()
            except Exception as merge_error:
                self.logger.error(f"Emergency merge failed during shutdown: {str(merge_error)}")
                print(f"Emergency merge failed during shutdown: {str(merge_error)}")

    def print_error_summary(self):
        """Print summary of errors encountered"""
        if self.failed_calls > 0 or self.had_errors:
            error_summary = "\nError Summary:\n-------------\n"
            for status, data in self.error_counts.items():
                if data['count'] > 0:
                    error_summary += f"{data['description']} ({status}): {data['count']} occurrences\n"
            error_summary += f"Error details logged to: {os.path.join(Config.OUTPUT_DIR, 'logs')}\n-------------"
            
            self.logger.warning(error_summary)
            print(error_summary)

    def process_companies(self, company_ids: List[str]):
        """Process list of companies with concurrent processing"""
        total_companies = len(company_ids)
        self.start_time = time.time()
        companies_processed = 0
        current_index = 0
        last_company_id = None
        
        # Load previous progress
        progress = self.load_progress()
        start_index = progress.get('last_index', 0)
        
        if start_index > 0:
            self.logger.info(f"Resuming from company {start_index + 1}")
            print(f"\nResuming from company {start_index + 1}")
            print(f"Previous progress: {self.successful_calls} companies processed")
        
        self.logger.info(f"Starting Company Enrichment processing for {total_companies} companies...")
        print(f"\nStarting Company Enrichment processing for {total_companies} companies...")

        try:
            # Process in batches of 1000 companies
            batch_size = 1000
            
            with ThreadPoolExecutor(max_workers=20) as executor:
                for i in range(start_index, total_companies, batch_size):
                    batch_end = min(i + batch_size, total_companies)
                    batch = company_ids[i:batch_end]
                    last_company_id = batch[-1]
                    current_index = i + len(batch)
                    
                    # Submit batch to thread pool
                    futures = [
                        executor.submit(self.get_company_enrichment, company_id)
                        for company_id in batch
                    ]
                    
                    # Process completed futures
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                self.buffer.update(result)
                                companies_processed += 1
                                
                                if len(self.buffer) >= self.buffer_size:
                                    self.append_buffer_to_temp()
                        except Exception as e:
                            self.logger.error(f"Error processing future: {str(e)}")
                    
                    # Save progress and show stats
                    if current_index % self.progress_interval == 0:
                        self.save_progress(current_index, last_company_id)
                        self.save_chunk_if_needed(companies_processed)
                        
                        # Show progress
                        metrics = self.get_performance_metrics()
                        elapsed_time = time.time() - self.start_time
                        remaining_companies = total_companies - current_index
                        estimated_remaining_time = (elapsed_time / (current_index - start_index)) * remaining_companies if current_index > start_index else 0
                        
                        status = (
                            f"\nProgress: {current_index}/{total_companies} "
                            f"({(current_index/total_companies)*100:.1f}%)\n"
                            f"Current Rate: {metrics['current_rate']:.1f} requests/second\n"
                            f"Success Rate: {metrics['success_rate']:.2%}\n"
                            f"Avg Request Time: {metrics['avg_request_time']:.3f} seconds\n"
                            f"Est. time remaining: {estimated_remaining_time/60:.1f} minutes"
                        )
                        
                        self.logger.info(status)
                        print(status)
                        
                        if self.failed_calls > 0:
                            self.print_error_summary()

            # Final cleanup
            if self.buffer:
                self.append_buffer_to_temp()
            
            self.merge_final_data()
            
            # Print final statistics
            elapsed_time = time.time() - self.start_time
            print("\nProcessing Complete:")
            print("-------------------")
            print(f"Total Runtime: {elapsed_time/60:.1f} minutes")
            print(f"Companies Processed: {companies_processed}")
            print(f"Success Rate: {(self.successful_calls/max(self.api_calls_made, 1))*100:.1f}%")
            print(f"Average Rate: {companies_processed/elapsed_time:.1f} companies/second")
            
            if self.had_errors:
                self.print_error_summary()
            
            # Remove progress file after successful completion
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)

        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
            self.handle_shutdown(current_index, last_company_id)
        except SystemExit as e:
            print(f"\nProcess terminated: {str(e)}")
            self.handle_shutdown(current_index, last_company_id)
        except Exception as e:
            self.logger.error(f"\nError during processing: {str(e)}")
            print(f"\nError during processing: {str(e)}")
            self.handle_shutdown(current_index, last_company_id)
        finally:
            # Ensure files are merged even if handle_shutdown fails
            try:
                if not os.path.exists(os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['company_enrichment'])):
                    self.emergency_merge_files()
            except Exception as e:
                self.logger.error(f"Final emergency merge attempt failed: {str(e)}")
                print(f"Final emergency merge attempt failed: {str(e)}")
            print("\nProcessing finished.")