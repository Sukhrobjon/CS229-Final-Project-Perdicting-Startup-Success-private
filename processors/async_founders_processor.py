# /processors/async_founders_processor.py

import asyncio
import aiohttp
from typing import Dict, List, Optional, Set, NamedTuple
from collections import defaultdict, deque
import time
import json
import os
from datetime import datetime
import logging
from utils.config import Config
from utils.async_rate_limiter import AsyncRateLimiter

class QueueConfig(NamedTuple):
    name: str
    concurrency: int
    timeout: int
    batch_size: int

class AsyncFoundersProcessor:
    def __init__(self, api_key: str, batch_size: int = 1000):
        self.api_key = api_key
        self.batch_size = batch_size
        self.headers = {"Authorization": f"Bearer {api_key}"}
        
        # Queue configurations
        self.queues = {
            'fast': QueueConfig('fast', concurrency=40, timeout=10, batch_size=400),
            'medium': QueueConfig('medium', concurrency=20, timeout=15, batch_size=400),
            'slow': QueueConfig('slow', concurrency=5, timeout=20, batch_size=200)
        }
        
        # Queue tracking
        self.queue_results = {
            'fast': {'success': 0, 'failure': 0, 'credits_used': 0},
            'medium': {'success': 0, 'failure': 0, 'credits_used': 0},
            'slow': {'success': 0, 'failure': 0, 'credits_used': 0}
        }
        
        # Failed IDs for each queue
        self.failed_ids = {
            'fast': set(),
            'medium': set(),
            'slow': set()
        }
        
        # Configuration
        self.progress_interval = 100  # Reduced for more frequent updates
        self.rate_limiter = AsyncRateLimiter(max_requests=150)
        
        # State tracking
        self.total_founders = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.buffer = {}
        self.start_time = time.time()
        self.credits_used = 0
        
        # Error tracking
        self.error_tracking = {
            'rate_limits': {'count': 0, 'timestamps': [], 'retry_after_values': []},
            'timeouts': {'count': 0, 'timestamps': [], 'response_times': []},
            'api_errors': defaultdict(lambda: {'count': 0, 'timestamps': [], 'messages': []})
        }
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Set up logging with queue-specific tracking"""
        self.logger = logging.getLogger('async_founders_processor')
        self.logger.setLevel(logging.INFO)
        
        log_dir = os.path.join(Config.OUTPUT_DIR, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Add queue-specific log files
        for queue_name in self.queues.keys():
            queue_log = os.path.join(log_dir, f'queue_{queue_name}_{datetime.now().strftime("%Y%m%d_%H%M")}.log')
            handler = logging.FileHandler(queue_log)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

    def print_queue_stats(self, queue_name: str):
        """Print statistics for a specific queue"""
        stats = self.queue_results[queue_name]
        total = stats['success'] + stats['failure']
        success_rate = (stats['success'] / total * 100) if total > 0 else 0
        
        print(f"\n{queue_name.upper()} Queue Statistics:")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Successful Calls: {stats['success']}")
        print(f"Failed Calls: {stats['failure']}")
        print(f"Credits Used: {stats['credits_used']}")

    def update_queue_stats(self, queue_name: str, success: bool):
        """Update statistics for a queue"""
        if success:
            self.queue_results[queue_name]['success'] += 1
        else:
            self.queue_results[queue_name]['failure'] += 1
        self.queue_results[queue_name]['credits_used'] += 1
    
    async def process_with_queue(self, session: aiohttp.ClientSession, company_id: str, 
                               queue_config: QueueConfig) -> Optional[Dict]:
        """Process a single company with queue-specific settings"""
        try:
            await self.rate_limiter.acquire()
            
            timeout = aiohttp.ClientTimeout(
                total=queue_config.timeout,
                connect=queue_config.timeout // 2,
                sock_read=queue_config.timeout // 2
            )
            
            url = f"https://data.api.aviato.co/company/{company_id}/founders"
            params = {"perPage": 20, "page": 0}
            
            start_time = time.time()
            
            async with session.get(url, params=params, headers=self.headers, 
                                 timeout=timeout) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    self.successful_calls += 1
                    
                    if data and 'founders' in data:
                        self.total_founders += len(data['founders'])
                    
                    self.update_queue_stats(queue_config.name, success=True)
                    self.logger.info(
                        f"[{queue_config.name}] Success for {company_id}. "
                        f"Response time: {response_time:.2f}s"
                    )
                    
                    return {company_id: data}
                    
                elif response.status == 429:  # Rate limit
                    retry_after = int(response.headers.get('Retry-After', 60))
                    self.error_tracking['rate_limits']['count'] += 1
                    self.error_tracking['rate_limits']['retry_after_values'].append(retry_after)
                    
                    self.logger.warning(
                        f"[{queue_config.name}] Rate limit for {company_id}. "
                        f"Retry-After: {retry_after}s"
                    )
                    
                    # Move to slower queue
                    self.failed_ids[queue_config.name].add(company_id)
                    return None
                    
                else:
                    self.failed_calls += 1
                    error_body = await response.text()
                    self.error_tracking['api_errors'][str(response.status)]['count'] += 1
                    
                    self.update_queue_stats(queue_config.name, success=False)
                    self.logger.error(
                        f"[{queue_config.name}] Error {response.status} for {company_id}. "
                        f"Response: {error_body[:200]}"
                    )
                    
                    # Move to slower queue
                    self.failed_ids[queue_config.name].add(company_id)
                    return None
                    
        except asyncio.TimeoutError:
            self.failed_calls += 1
            self.error_tracking['timeouts']['count'] += 1
            self.error_tracking['timeouts']['response_times'].append(queue_config.timeout)
            
            self.update_queue_stats(queue_config.name, success=False)
            self.logger.error(f"[{queue_config.name}] Timeout for {company_id}")
            
            # Move to slower queue
            self.failed_ids[queue_config.name].add(company_id)
            return None
            
        except Exception as e:
            self.failed_calls += 1
            self.update_queue_stats(queue_config.name, success=False)
            self.logger.error(
                f"[{queue_config.name}] Unexpected error for {company_id}: {str(e)}"
            )
            
            # Move to slower queue
            self.failed_ids[queue_config.name].add(company_id)
            return None

    async def process_queue(self, session: aiohttp.ClientSession, 
                          company_ids: List[str], queue_config: QueueConfig):
        """Process a batch of companies with queue-specific concurrency"""
        self.logger.info(
            f"Processing {len(company_ids)} companies in {queue_config.name} queue "
            f"with concurrency {queue_config.concurrency}"
        )
        
        # Process in smaller chunks to maintain control
        for i in range(0, len(company_ids), queue_config.batch_size):
            batch = company_ids[i:i + queue_config.batch_size]
            
            # Create tasks with limited concurrency
            semaphore = asyncio.Semaphore(queue_config.concurrency)
            
            async def process_with_semaphore(company_id: str):
                async with semaphore:
                    return await self.process_with_queue(session, company_id, queue_config)
            
            tasks = [process_with_semaphore(cid) for cid in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update buffer with successful results
            for result in results:
                if isinstance(result, dict):
                    self.buffer.update(result)
            
            # Print progress
            processed = i + len(batch)
            if processed % self.progress_interval == 0:
                self.print_queue_stats(queue_config.name)
                print(f"Processed {processed}/{len(company_ids)} in {queue_config.name} queue")

    async def save_results(self):
        """Save current results to file"""
        if not self.buffer:
            return
            
        output_file = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['founders'])
        try:
            with open(output_file, 'w') as f:
                json.dump(self.buffer, f, indent=2)
            self.logger.info(f"Saved {len(self.buffer)} results to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")

    def print_final_summary(self):
        """Print comprehensive summary of all queues"""
        print("\nFinal Processing Summary:")
        print("========================")
        
        total_processed = 0
        total_success = 0
        total_credits = 0
        
        for queue_name, stats in self.queue_results.items():
            total = stats['success'] + stats['failure']
            success_rate = (stats['success'] / total * 100) if total > 0 else 0
            total_processed += total
            total_success += stats['success']
            total_credits += stats['credits_used']
            
            print(f"\n{queue_name.upper()} Queue:")
            print(f"Processed: {total}")
            print(f"Success Rate: {success_rate:.1f}%")
            print(f"Credits Used: {stats['credits_used']}")
        
        overall_success_rate = (total_success / total_processed * 100) if total_processed > 0 else 0
        print("\nOverall Statistics:")
        print(f"Total Processed: {total_processed}")
        print(f"Total Successful: {total_success}")
        print(f"Overall Success Rate: {overall_success_rate:.1f}%")
        print(f"Total Credits Used: {total_credits}")
        print(f"Total Founders Found: {self.total_founders}")
        
        # Print error summary
        if self.error_tracking['rate_limits']['count'] > 0:
            print(f"\nRate Limits: {self.error_tracking['rate_limits']['count']}")
        if self.error_tracking['timeouts']['count'] > 0:
            print(f"Timeouts: {self.error_tracking['timeouts']['count']}")
        for status, data in self.error_tracking['api_errors'].items():
            if data['count'] > 0:
                print(f"API Error {status}: {data['count']}")

    async def process_companies(self, company_ids: List[str]):
        """Main processing function using queue-based approach"""
        total_companies = len(company_ids)
        self.start_time = time.time()
        
        print(f"\nProcessing {total_companies} companies using queue-based approach...")
        
        try:
            connector = aiohttp.TCPConnector(
                limit=None,  # Let semaphores handle concurrency
                force_close=False,
                enable_cleanup_closed=True
            )
            
            async with aiohttp.ClientSession(connector=connector) as session:
                # Process through fast queue
                fast_queue = company_ids[:self.queues['fast'].batch_size]
                await self.process_queue(session, fast_queue, self.queues['fast'])
                
                # Move failures and remaining companies to medium queue
                medium_queue = list(self.failed_ids['fast'])
                if len(company_ids) > self.queues['fast'].batch_size:
                    medium_queue.extend(
                        company_ids[self.queues['fast'].batch_size:
                                  self.queues['fast'].batch_size + self.queues['medium'].batch_size]
                    )
                
                if medium_queue:
                    await self.process_queue(session, medium_queue, self.queues['medium'])
                
                # Move failures and remaining companies to slow queue
                slow_queue = list(self.failed_ids['medium'])
                remaining_start = self.queues['fast'].batch_size + self.queues['medium'].batch_size
                if len(company_ids) > remaining_start:
                    slow_queue.extend(company_ids[remaining_start:])
                
                if slow_queue:
                    await self.process_queue(session, slow_queue, self.queues['slow'])
                
                # Save results
                await self.save_results()
                
                # Print final summary
                elapsed = time.time() - self.start_time
                self.print_final_summary()
                print(f"\nTotal processing time: {elapsed/60:.1f} minutes")
                
                # Save detailed analysis
                analysis = {
                    'queue_results': self.queue_results,
                    'error_tracking': self.error_tracking,
                    'performance': {
                        'total_time': elapsed,
                        'companies_per_second': total_companies / elapsed,
                        'total_founders': self.total_founders
                    }
                }
                
                analysis_file = os.path.join(
                    Config.OUTPUT_DIR, 
                    f"founders_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                )
                
                with open(analysis_file, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
                print(f"\nDetailed analysis saved to: {analysis_file}")
                
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
            await self.save_results()
            self.print_final_summary()
        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}")
            print(f"\nError during processing: {str(e)}")
            await self.save_results()
            self.print_final_summary()