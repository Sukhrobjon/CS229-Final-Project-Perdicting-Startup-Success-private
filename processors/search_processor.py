import json
import os
import glob
import logging
from typing import Dict, Any, Optional, List, Tuple
import requests
import time
from datetime import datetime
from processors.base_processor import BaseProcessor
from utils.config import Config
from utils.rate_limiter import RateLimiter

class SearchProcessor(BaseProcessor):
    def __init__(self, api_key: str, batch_size: int = 1000):
        super().__init__(api_key, batch_size)
        # Buffer and chunking configuration
        self.buffer = {}
        self.buffer_size = 50
        self.chunk_size = 10000
        self.progress_interval = 500
        self.rate_limiter = RateLimiter(max_requests=1000)
        
        # API tracking
        self.api_calls_made = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.had_errors = False
        
        # Search specific
        self.last_company_name = None
        self.total_available = 0
        self.total_processed = 0
        self.current_chunk_start_id = None
        
        # Timing tracking
        self.start_time = None
        self.last_thousand_time = None
        self.thousand_company_intervals = []
        self.target_companies = None
        
        # Duplicate tracking
        self.company_ids_seen = set()
        self.company_names_seen = set()
        
        # Metadata tracking
        self.metadata = {
            "total_companies": 0,
            "target_companies": 0,
            "files_merged": 0,
            "duplicates_found": 0,
            "start_time": None,
            "companies_processed": 0,
            "chunks_created": 0
        }
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        log_file = os.path.join(Config.OUTPUT_DIR, f'search_processor_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Search Processor initialized")
    def check_duplicates(self, companies: List[Dict]) -> List[Dict]:
        """Track duplicates similar to FileManager"""
        duplicates = []
        for company in companies:
            company_id = company.get('id')
            company_name = company.get('name')
            
            if company_id in self.company_ids_seen:
                duplicates.append({
                    'id': company_id,
                    'name': company_name,
                    'type': 'id_duplicate'
                })
            if company_name in self.company_names_seen:
                duplicates.append({
                    'id': company_id,
                    'name': company_name,
                    'type': 'name_duplicate'
                })
                
            self.company_ids_seen.add(company_id)
            if company_name:
                self.company_names_seen.add(company_name)
            
        return duplicates

    def get_search_dsl(self, offset: int) -> Dict:
        """Generate DSL with pagination and sorting"""
        filters = [
            {
                "AND": [
                    {
                        "founded": {
                            "operation": "gte",
                            "value": "2011-12-31T23:59:59.999Z"
                        }
                    },
                    {
                        "founded": {
                            "operation": "lte",
                            "value": "2022-12-31T23:59:59.999Z"
                        }
                    },
                    {
                        "country": {
                            "operation": "eq",
                            "value": "United States"
                        }
                    },
                    {
                        "companyFoundingLinks.personID": {
                            "operation": "noteq",
                            "value": None
                        }
                    }
                ]
            }
        ]

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
        self.rate_limiter.wait_if_needed()
        url = f"{Config.BASE_URL}/company/search"
        payload = {"dsl": dsl}
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            self.api_calls_made += 1
            
            if response.status_code == 429:  # Rate limit hit
                retry_after = int(response.headers.get('Retry-After', 60))
                self.logger.warning(f"Rate limit hit. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self.search_companies(dsl)
            
            if response.status_code != 200:
                self.failed_calls += 1
                self.had_errors = True
                self.logger.error(f"API Error: {response.status_code}")
                self.logger.error(f"Response: {response.text}")
                
            response.raise_for_status()
            self.successful_calls += 1
            
            # Log rate stats periodically
            if self.api_calls_made % 100 == 0:
                self.log_rate_limiter_stats()
                
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Request failed: {str(e)}", exc_info=True)
            raise

    def log_rate_limiter_stats(self):
        """Log current rate limiter statistics"""
        if self.start_time:
            current_rate = self.api_calls_made / ((time.time() - self.start_time) / 60)
            self.logger.info(f"Current API call rate: {current_rate:.1f} calls/minute")
            self.logger.info(f"Rate limiter status: {self.rate_limiter.get_status()}")

    def log_thousand_milestone(self):
        """Log statistics for every 1000 companies processed"""
        current_time = time.time()
        
        if self.last_thousand_time is None:
            self.last_thousand_time = self.start_time
        
        interval_time = current_time - self.last_thousand_time
        self.thousand_company_intervals.append(interval_time)
        
        avg_time_per_thousand = sum(self.thousand_company_intervals) / len(self.thousand_company_intervals)
        remaining_thousands = (self.target_companies - self.total_processed) / 1000
        estimated_remaining_time = remaining_thousands * avg_time_per_thousand
        
        self.logger.info("\n=== Thousand Companies Milestone Report ===")
        self.logger.info(f"Companies processed: {self.total_processed:,}")
        self.logger.info(f"Time for last 1000: {interval_time/60:.1f} minutes")
        self.logger.info(f"Average time per 1000: {avg_time_per_thousand/60:.1f} minutes")
        self.logger.info(f"Estimated time remaining: {estimated_remaining_time/60:.1f} minutes")
        self.logger.info(f"Current success rate: {(self.successful_calls/self.api_calls_made)*100:.1f}%")
        self.log_rate_limiter_stats()
        self.logger.info("==========================================\n")
        
        self.last_thousand_time = current_time

    def save_progress(self):
        """Save current progress with detailed metadata"""
        progress = {
            "metadata": {
                "total_available": self.total_available,
                "target_companies": self.target_companies,
                "total_processed": self.total_processed,
                "files_merged": len(glob.glob(os.path.join(Config.OUTPUT_DIR, "search_chunk_*.json"))),
                "api_calls_made": self.api_calls_made,
                "successful_calls": self.successful_calls,
                "failed_calls": self.failed_calls,
                "duplicates_found": len(self.company_ids_seen) + len(self.company_names_seen) - self.total_processed,
                "last_company_name": self.last_company_name,
                "current_chunk_start_id": self.current_chunk_start_id,
                "chunks_created": self.metadata["chunks_created"],
                "had_errors": self.had_errors,
                "last_updated": datetime.now().isoformat()
            },
            "performance": {
                "elapsed_time": time.time() - self.start_time if self.start_time else 0,
                "average_rate": self.total_processed / ((time.time() - self.start_time) / 60) if self.start_time else 0,
                "thousand_intervals": self.thousand_company_intervals
            }
        }
        
        progress_file = os.path.join(Config.OUTPUT_DIR, "search_progress.json")
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        self.logger.debug("Progress saved with detailed metadata")
    def load_progress(self) -> Dict:
        """Load previous progress"""
        progress_file = os.path.join(Config.OUTPUT_DIR, "search_progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    metadata = progress.get('metadata', {})
                    self.total_processed = metadata.get('total_processed', 0)
                    self.api_calls_made = metadata.get('api_calls_made', 0)
                    self.successful_calls = metadata.get('successful_calls', 0)
                    self.failed_calls = metadata.get('failed_calls', 0)
                    self.had_errors = metadata.get('had_errors', False)
                    self.last_company_name = metadata.get('last_company_name')
                    self.current_chunk_start_id = metadata.get('current_chunk_start_id')
                    self.metadata["chunks_created"] = metadata.get('chunks_created', 0)
                    
                    # Load performance data
                    performance = progress.get('performance', {})
                    self.thousand_company_intervals = performance.get('thousand_intervals', [])
                    
                    self.logger.info("Previous progress loaded successfully")
                    return progress
            except Exception as e:
                self.logger.error(f"Error loading progress: {e}", exc_info=True)
        return {}

    def save_batch(self):
        """Implementation of abstract method from BaseProcessor"""
        self.append_buffer_to_temp()

    def append_buffer_to_temp(self):
        """Append current buffer to temp file"""
        if not self.buffer:
            return
            
        temp_file = os.path.join(Config.OUTPUT_DIR, "search_temp.json")
        
        try:
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    existing_data = json.load(f)
                    if 'companies' in existing_data:
                        companies_data = {company['id']: company for company in existing_data['companies']}
                    else:
                        companies_data = existing_data
                companies_data.update(self.buffer)
            else:
                companies_data = self.buffer

            # Check for duplicates before saving
            duplicates = self.check_duplicates(list(self.buffer.values()))
            if duplicates:
                self.logger.warning(f"Found {len(duplicates)} duplicates in current batch")
                self.metadata["duplicates_found"] += len(duplicates)

            # Save with metadata
            data_to_save = {
                "metadata": {
                    "companies_count": len(companies_data),
                    "timestamp": datetime.now().isoformat(),
                    "duplicates_found": len(duplicates) if duplicates else 0
                },
                "companies": list(companies_data.values())
            }

            with open(temp_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            self.logger.debug(f"Appended {len(self.buffer)} companies to temp file")
            self.buffer = {}
            
        except Exception as e:
            self.logger.error(f"Error appending to temp file: {str(e)}", exc_info=True)
            self.had_errors = True

    def save_chunk_if_needed(self, companies_processed: int):
        """Create new chunk file at 10k threshold"""
        if companies_processed % self.chunk_size == 0:
            temp_file = os.path.join(Config.OUTPUT_DIR, "search_temp.json")
            if os.path.exists(temp_file):
                try:
                    with open(temp_file, 'r') as f:
                        data = json.load(f)
                        if data and 'companies' in data:
                            companies = data['companies']
                            if companies:
                                first_company = companies[0]
                                last_company = companies[-1]
                                chunk_name = f"search_chunk_{first_company['id']}_to_{last_company['id']}.json"
                                chunk_file = os.path.join(Config.OUTPUT_DIR, chunk_name)
                                
                                # Update metadata for chunk
                                data['metadata'].update({
                                    "chunk_number": self.metadata["chunks_created"] + 1,
                                    "start_id": first_company['id'],
                                    "end_id": last_company['id'],
                                    "final_timestamp": datetime.now().isoformat()
                                })
                                
                                with open(chunk_file, 'w') as f:
                                    json.dump(data, f, indent=2)
                                
                                os.remove(temp_file)
                                self.metadata["chunks_created"] += 1
                                self.logger.info(f"Created chunk file: {chunk_file}")
                                self.logger.info(f"Chunk contains {len(companies)} companies")
                except Exception as e:
                    self.logger.error(f"Error creating chunk file: {str(e)}", exc_info=True)
                    self.had_errors = True

    def merge_final_data(self, target_companies: int):
        """Merge all data into final file"""
        final_companies = []
        duplicates_found = 0
        
        try:
            self.logger.info("Starting final data merge process")
            chunk_pattern = os.path.join(Config.OUTPUT_DIR, "search_chunk_*.json")
            chunk_files = sorted(glob.glob(chunk_pattern))
            
            # Reset duplicate tracking for final merge
            self.company_ids_seen.clear()
            self.company_names_seen.clear()
            
            if chunk_files:
                self.logger.info(f"Found {len(chunk_files)} chunk files to merge")
                for i, chunk_file in enumerate(chunk_files, 1):
                    self.logger.info(f"Processing chunk file {i}/{len(chunk_files)}: {os.path.basename(chunk_file)}")
                    with open(chunk_file, 'r') as f:
                        chunk_data = json.load(f)
                        companies = chunk_data.get('companies', [])
                        
                        # Check for duplicates
                        duplicates = self.check_duplicates(companies)
                        duplicates_found += len(duplicates)
                        
                        final_companies.extend(companies)
                    os.remove(chunk_file)
                    self.logger.debug(f"Removed processed chunk file: {chunk_file}")
            
            # Process remaining temp file
            temp_file = os.path.join(Config.OUTPUT_DIR, "search_temp.json")
            if os.path.exists(temp_file):
                self.logger.info("Processing remaining data from temp file")
                with open(temp_file, 'r') as f:
                    temp_data = json.load(f)
                    companies = temp_data.get('companies', [])
                    
                    # Check for duplicates
                    duplicates = self.check_duplicates(companies)
                    duplicates_found += len(duplicates)
                    
                    final_companies.extend(companies)
                os.remove(temp_file)
            
            # Prepare final metadata
            final_metadata = {
                "total_companies": len(final_companies),
                "target_companies": target_companies,
                "files_merged": len(chunk_files),
                "duplicates_found": duplicates_found,
                "timestamp": datetime.now().isoformat(),
                "processing_stats": {
                    "api_calls_made": self.api_calls_made,
                    "successful_calls": self.successful_calls,
                    "failed_calls": self.failed_calls,
                    "had_errors": self.had_errors
                }
            }
            
            # Save final merged file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            final_file = os.path.join(
                Config.OUTPUT_DIR, 
                f"companies_merged_{target_companies}_{timestamp}.json"
            )
            
            final_data = {
                "metadata": final_metadata,
                "companies": final_companies
            }
            
            with open(final_file, 'w') as f:
                json.dump(final_data, f, indent=2)
            
            self.logger.info(f"Final merge complete. Total companies: {len(final_companies)}")
            self.logger.info(f"Final file saved as: {final_file}")
            
            return final_file
            
        except Exception as e:
            self.logger.error(f"Error during final merge: {str(e)}", exc_info=True)
            self.had_errors = True
            return None

    def process_companies(self, target_companies: int = 1000):
        """Process companies with resume capability"""
        self.target_companies = target_companies
        self.start_time = time.time() if not self.start_time else self.start_time
        self.logger.info(f"Starting company search process for {target_companies:,} companies")
        self.load_progress()
        
        try:
            # First call to get total count and first batch
            dsl = self.get_search_dsl(0)
            response = self.search_companies(dsl)
            
            self.total_available = int(response.get('count', {}).get('value', 0))
            self.logger.info(f"Total available companies: {self.total_available:,}")
            
            # Process first batch
            items = response.get('items', [])
            if items:
                for item in items:
                    company_id = item.get('id')
                    if not self.current_chunk_start_id:
                        self.current_chunk_start_id = company_id
                    self.buffer[company_id] = item
                
                self.total_processed = len(items)
                self.last_company_name = items[-1].get('name')
                
                if len(self.buffer) >= self.buffer_size:
                    self.append_buffer_to_temp()
                
                self.logger.info(f"Processed first batch: {self.total_processed:,} companies")

            # Process remaining in batches
            while self.total_processed < target_companies:
                remaining = target_companies - self.total_processed
                batch_size = min(250, remaining)
                
                self.logger.info(f"Fetching next batch ({self.total_processed + 1:,} - {self.total_processed + batch_size:,})")
                
                dsl = self.get_search_dsl(self.total_processed)
                dsl['limit'] = batch_size
                
                try:
                    response = self.search_companies(dsl)
                    items = response.get('items', [])
                    
                    if not items:
                        self.logger.warning("No more companies available")
                        break

                    for item in items:
                        company_id = item.get('id')
                        self.buffer[company_id] = item
                    
                    self.total_processed += len(items)
                    self.last_company_name = items[-1].get('name')
                    
                    if len(self.buffer) >= self.buffer_size:
                        self.append_buffer_to_temp()
                    
                    self.save_chunk_if_needed(self.total_processed)
                    
                    # Add thousand milestone logging
                    if (self.total_processed % 1000) < 250:
                        self.log_thousand_milestone()
                    
                    # Enhanced progress logging
                    if self.total_processed % self.progress_interval == 0:
                        self.save_progress()
                        current_time = time.time()
                        elapsed_time = current_time - self.start_time
                        
                        self.logger.info("\n=== Progress Update ===")
                        self.logger.info(f"Companies processed: {self.total_processed:,}/{target_companies:,} "
                                       f"({(self.total_processed/target_companies)*100:.1f}%)")
                        self.logger.info(f"API calls: {self.api_calls_made:,} "
                                       f"(Success: {self.successful_calls:,}, Failed: {self.failed_calls:,})")
                        self.logger.info(f"Success rate: {(self.successful_calls/self.api_calls_made)*100:.1f}%")
                        self.logger.info(f"Time elapsed: {elapsed_time/60:.1f} minutes")
                        self.logger.info(f"Average processing rate: {self.total_processed/(elapsed_time/60):.1f} companies/minute")
                        self.log_rate_limiter_stats()
                        self.logger.info("=====================\n")
                    
                    time.sleep(1)  # Rate limiting
                    
                except KeyboardInterrupt:
                    self.logger.warning("Process interrupted by user")
                    if self.buffer:
                        self.append_buffer_to_temp()
                    self.save_progress()
                    raise
                except Exception as e:
                    self.logger.error(f"Error processing batch: {str(e)}", exc_info=True)
                    if self.buffer:
                        self.append_buffer_to_temp()
                    self.save_progress()
                    raise

            # Save any remaining buffer
            if self.buffer:
                self.append_buffer_to_temp()

            # Merge files if processing is complete
            if self.total_processed >= target_companies:
                self.logger.info("Processing complete. Starting file merge...")
                merged_file = self.merge_final_data(target_companies)
                if merged_file:
                    self.logger.info(f"Final merged file: {merged_file}")
                
                # Clean up progress file
                progress_file = os.path.join(Config.OUTPUT_DIR, "search_progress.json")
                if os.path.exists(progress_file):
                    os.remove(progress_file)
                    self.logger.info("Cleaned up progress file")

        except KeyboardInterrupt:
            self.logger.warning("Process interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}", exc_info=True)
        finally:
            elapsed_time = time.time() - self.start_time
            self.logger.info("\n=== Final Processing Summary ===")
            self.logger.info(f"Total Companies Found: {self.total_available:,}")
            self.logger.info(f"Total Companies Processed: {self.total_processed:,}")
            self.logger.info(f"API Calls: {self.api_calls_made:,} (Success: {self.successful_calls:,}, Failed: {self.failed_calls:,})")
            self.logger.info(f"Average Rate: {self.api_calls_made/(elapsed_time/60):.1f} calls/minute")
            self.logger.info(f"Total Processing Time: {elapsed_time/60:.1f} minutes")
            self.logger.info(f"Final Success Rate: {(self.successful_calls/self.api_calls_made)*100:.1f}%")
            self.logger.info(f"Duplicates Found: {self.metadata['duplicates_found']}")
            self.logger.info(f"Status: {'Completed with errors' if self.had_errors else 'Clean'}")
            self.logger.info("===============================")