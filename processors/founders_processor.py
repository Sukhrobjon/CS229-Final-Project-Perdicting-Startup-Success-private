# /processors/founders_processor.py

import requests
import json
import os
import glob
from typing import Dict, Optional, List
import time
from datetime import datetime
from statistics import mean, median
import logging
from processors.base_processor import BaseProcessor
from utils.config import Config
from utils.rate_limiter import EnhancedRateLimiter
from utils.retry_handler import RetryHandler

class FoundersProcessor(BaseProcessor):
    def __init__(self, api_key: str, batch_size: int = 1000):
        super().__init__(api_key, batch_size)
        self.progress_interval = 500
        self.rate_limiter = EnhancedRateLimiter(max_requests=150)
        self.retry_handler = RetryHandler()
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
        
        # Initialize session
        self.session.headers.update(self.headers)
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Set up logging for the processor"""
        self.logger = logging.getLogger('founders_processor')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(Config.OUTPUT_DIR, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        log_file = os.path.join(log_dir, f'founders_processor_{datetime.now().strftime("%Y%m%d_%H%M")}.log')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def get_founders(self, company_id: str) -> Optional[Dict]:
        """Get company founders - Enhanced version with comprehensive error handling"""
        start_time = time.time()
        
        # Check kill switch conditions before making request
        should_kill, reason = self.rate_limiter.should_kill_process()
        if should_kill:
            self.logger.error(f"Kill switch activated: {reason}")
            raise SystemExit(f"Process terminated: {reason}")
        
        self.rate_limiter.wait_if_needed()
        
        url = f"https://data.api.aviato.co/company/{company_id}/founders"
        params = {
            "perPage": 100,
            "page": 0
        }
        
        def make_request():
            """Inner function to make the actual request"""
            return self.session.get(url, params=params, timeout=10)
        
        try:
            response = make_request()
            request_time = time.time() - start_time
            self.request_times.append(request_time)
            self.api_calls_made += 1
            
            if response.status_code == 200:
                self.successful_calls += 1
                self.rate_limiter.record_result(success=True)
                data = response.json()
                if data and 'founders' in data:
                    self.total_founders += len(data['founders'])
                return data
                
            elif response.status_code == 429:
                # Rate limit hit
                self.rate_limiter.record_result(success=False, is_rate_limit=True)
                return self.retry_handler.handle_rate_limit(
                    response,
                    lambda: self.get_founders(company_id)
                )
                
            elif response.status_code >= 500:
                # Server errors
                self.rate_limiter.record_result(success=False)
                return self.retry_handler.handle_server_error(
                    response.status_code,
                    lambda: self.get_founders(company_id)
                )
                
            else:
                # Other errors (400, 401, 403, 404)
                self.failed_calls += 1
                self.had_errors = True
                self.rate_limiter.record_result(success=False)
                
                if response.status_code in self.error_counts:
                    self.error_counts[response.status_code]['count'] += 1
                else:
                    self.error_counts['other']['count'] += 1
                
                error_details = (
                    f"Company {company_id}: Status {response.status_code}, "
                    f"Response: {response.text[:200]}"
                )
                self.logger.error(error_details)
                self.retry_handler.log_error('api_error', error_details)
                
                return None
                    
        except requests.exceptions.Timeout:
            self.failed_calls += 1
            self.had_errors = True
            self.error_counts['timeout']['count'] += 1
            self.rate_limiter.record_result(success=False, is_timeout=True)
            
            return self.retry_handler.handle_network_error(
                Exception("Timeout"),
                lambda: self.get_founders(company_id)
            )
            
        except requests.exceptions.RequestException as e:
            self.failed_calls += 1
            self.had_errors = True
            self.error_counts['other']['count'] += 1
            self.rate_limiter.record_result(success=False)
            
            return self.retry_handler.handle_network_error(
                e,
                lambda: self.get_founders(company_id)
            )
            
        except Exception as e:
            self.failed_calls += 1
            self.had_errors = True
            self.error_counts['other']['count'] += 1
            self.rate_limiter.record_result(success=False)
            
            error_details = f"Company {company_id}: Unexpected error: {str(e)}"
            self.logger.error(error_details)
            self.retry_handler.log_error('unexpected_error', error_details)
            
            return None

    def get_performance_metrics(self) -> Dict:
        """Calculate current performance metrics with retry statistics"""
        if not self.request_times:
            return {
                "avg_request_time": 0,
                "median_request_time": 0,
                "min_request_time": 0,
                "max_request_time": 0,
                "current_rate": 0,
                "success_rate": 0,
                "error_rate": 0,
                "retry_statistics": self.retry_handler.get_stats()
            }
            
        recent_times = self.request_times[-1000:]  # Look at last 1000 requests
        avg_time = mean(recent_times)
        
        # Get rate limiter stats
        rate_limiter_stats = self.rate_limiter.get_stats()
        retry_stats = self.retry_handler.get_stats()
        
        return {
            "avg_request_time": avg_time,
            "median_request_time": median(recent_times),
            "min_request_time": min(recent_times),
            "max_request_time": max(recent_times),
            "current_rate": rate_limiter_stats['current_rate'],
            "success_rate": self.successful_calls / max(self.api_calls_made, 1),
            "error_rate": rate_limiter_stats['error_rate'],
            "retry_statistics": retry_stats
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
            "total_founders": self.total_founders,
            "had_errors": self.had_errors,
            "error_counts": self.error_counts,
            "performance_metrics": metrics,
            "rate_limiter_stats": rate_limiter_stats,
            "disk_usage": self.rate_limiter.check_disk_usage(),
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
                    self.total_founders = progress.get('total_founders', 0)
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
            self.logger.info(f"Buffer appended to temp file: {temp_file}")
            
        except Exception as e:
            self.logger.error(f"Error appending to temp file: {str(e)}")
            self.had_errors = True

    def save_chunk_if_needed(self, companies_processed: int):
        """Create new chunk file at threshold"""
        if companies_processed % self.chunk_size == 0:
            chunk_number = companies_processed // self.chunk_size
            temp_file = os.path.join(Config.OUTPUT_DIR, "founders_temp.json")
            chunk_file = os.path.join(Config.OUTPUT_DIR, f"founders_chunk_{chunk_number}.json")
            
            try:
                if os.path.exists(temp_file):
                    os.rename(temp_file, chunk_file)
                    self.logger.info(f"Created chunk file: {chunk_file}")
            except Exception as e:
                self.logger.error(f"Error creating chunk file: {str(e)}")

    def merge_final_data(self):
        """Merge all data into final file"""
        final_data = {}
        
        try:
            chunk_pattern = os.path.join(Config.OUTPUT_DIR, "founders_chunk_*.json")
            chunk_files = sorted(glob.glob(chunk_pattern))
            
            if chunk_files:
                self.logger.info("\nMerging chunk files...")
                for i, chunk_file in enumerate(chunk_files, 1):
                    with open(chunk_file, 'r') as f:
                        chunk_data = json.load(f)
                        final_data.update(chunk_data)
                    os.remove(chunk_file)
                    self.logger.info(f"Processed chunk file {i}/{len(chunk_files)}")
            
            temp_file = os.path.join(Config.OUTPUT_DIR, "founders_temp.json")
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    temp_data = json.load(f)
                    final_data.update(temp_data)
                os.remove(temp_file)
            
            final_file = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['founders'])
            with open(final_file, 'w') as f:
                json.dump(final_data, f, indent=2)
            
            self.logger.info("\nFinal Output Summary:")
            self.logger.info(f"Output file: {final_file}")
            self.logger.info(f"Total companies: {len(final_data)}")
            self.logger.info(f"Total founders: {self.total_founders}")
            
            print("\nFinal Output Summary:")
            print("------------------------")
            print(f"Output file: {final_file}")
            print(f"Total companies: {len(final_data)}")
            print(f"Total founders: {self.total_founders}")
            print("------------------------")
            
        except Exception as e:
            self.logger.error(f"Error during final merge: {str(e)}")
            self.had_errors = True
            print(f"\nError during final merge: {str(e)}")

    def print_error_summary(self):
        """Print summary of errors encountered"""
        if self.failed_calls > 0:
            error_summary = "\nError Summary:\n-------------\n"
            for status, data in self.error_counts.items():
                if data['count'] > 0:
                    error_summary += f"{data['description']} ({status}): {data['count']} occurrences\n"
            error_summary += f"Error details logged to: {os.path.join(Config.OUTPUT_DIR, 'logs')}\n-------------"
            
            self.logger.warning(error_summary)
            print(error_summary)
            
    def handle_shutdown(self, current_index: int, company_id: str):
        """Handle graceful shutdown"""
        self.logger.info("Initiating graceful shutdown...")
        
        if self.buffer:
            self.logger.info("Saving current buffer...")
            print("\nSaving current buffer...")
            self.append_buffer_to_temp()
        
        self.logger.info("Saving final progress...")
        print("\nSaving progress...")
        self.save_progress(current_index, company_id)
        
        # Get final statistics
        stats = self.rate_limiter.get_stats()
        metrics = self.get_performance_metrics()
        
        summary = f"\nFinal Statistics:\n"
        summary += f"Companies Processed: {self.successful_calls}\n"
        summary += f"Error Rate: {stats['error_rate']:.2%}\n"
        summary += f"Total Founders Found: {self.total_founders}\n"
        summary += f"Average Request Time: {metrics['avg_request_time']:.3f} seconds\n"
        summary += f"Total Runtime: {(time.time() - self.start_time) / 60:.1f} minutes"
        
        self.logger.info(summary)
        print(summary)
        
        if self.had_errors:
            self.print_error_summary()

    def print_stats(self, current_index: int, total_companies: int, start_time: float):
        """Print current statistics"""
        stats = self.rate_limiter.get_stats()
        metrics = self.get_performance_metrics()
        elapsed_time = time.time() - start_time
        remaining_companies = total_companies - current_index
        estimated_remaining_time = (elapsed_time / current_index) * remaining_companies if current_index > 0 else 0
        
        status = (
            f"\nProgress: {current_index}/{total_companies} "
            f"({(current_index/total_companies)*100:.1f}%)\n"
            f"Current Rate: {stats['current_rate']:.1f} requests/second\n"
            f"Success Rate: {metrics['success_rate']:.2%}\n"
            f"Error Rate: {stats['error_rate']:.2%}\n"
            f"Total Founders Found: {self.total_founders}\n"
            f"Avg Request Time: {metrics['avg_request_time']:.3f} seconds\n"
            f"Est. time remaining: {estimated_remaining_time/60:.1f} minutes"
        )
        
        # Add retry statistics if any retries occurred
        retry_stats = metrics.get('retry_statistics', {})
        if retry_stats.get('total_retries', 0) > 0:
            status += f"\nTotal Retries: {retry_stats['total_retries']}"
            for retry_type, count in retry_stats['retry_counts'].items():
                if count > 0:
                    status += f"\n  - {retry_type}: {count}"
        
        self.logger.info(status)
        print(status)

    def process_companies(self, company_ids: List[str]):
        """Process list of companies for founders with enhanced monitoring"""
        total_companies = len(company_ids)
        start_time = time.time()
        companies_processed = 0
        
        # Load previous progress
        progress = self.load_progress()
        start_index = progress.get('last_index', 0)
        
        if start_index > 0:
            self.logger.info(f"Resuming from company {start_index + 1}")
            print(f"\nResuming from company {start_index + 1}")
            print(f"Previous progress: {self.successful_calls} companies processed")
            print(f"Total founders found so far: {self.total_founders}")
        
        self.logger.info(f"Starting Founders processing for {total_companies} companies...")
        print(f"\nStarting Founders processing for {total_companies} companies...")

        try:
            for i, company_id in enumerate(company_ids[start_index:], start_index + 1):
                try:
                    founders_data = self.get_founders(company_id)
                    if founders_data:
                        self.buffer[company_id] = founders_data
                        companies_processed += 1
                        
                        if len(self.buffer) >= self.buffer_size:
                            self.append_buffer_to_temp()
                        
                        self.save_chunk_if_needed(companies_processed)
                    
                    # Show progress and stats every 500 companies
                    if i % self.progress_interval == 0:
                        self.save_progress(i, company_id)
                        self.print_stats(i, total_companies, start_time)
                        
                except SystemExit as e:
                    self.logger.error(f"Kill switch activated: {str(e)}")
                    print(f"\nKill switch activated: {str(e)}")
                    self.handle_shutdown(i, company_id)
                    return

            if self.buffer:
                self.append_buffer_to_temp()

            self.merge_final_data()

            # Remove progress file after successful completion
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)

        except KeyboardInterrupt:
            self.logger.warning("\nProcess interrupted by user")
            print("\nProcess interrupted by user")
            self.handle_shutdown(i, company_id)
        except Exception as e:
            self.logger.error(f"\nError during processing: {str(e)}")
            print(f"\nError during processing: {str(e)}")
            self.handle_shutdown(i, company_id)