# /processors/async_founders_processor.py

import asyncio
import aiohttp
from typing import Dict, List, Optional, Set
import time
import json
import os
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
from utils.config import Config
from utils.async_rate_limiter import AsyncRateLimiter

class AsyncFoundersProcessor:
    def __init__(self, api_key: str, batch_size: int = 1000):
        self.api_key = api_key
        self.batch_size = batch_size
        self.headers = {"Authorization": f"Bearer {api_key}"}
        
        # Configuration
        self.progress_interval = 500
        self.rate_limiter = AsyncRateLimiter(max_requests=150)
        
        # State tracking
        self.total_founders = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.buffer = {}
        self.start_time = time.time()
        
        # Analysis tracking
        self.failure_analysis = {
            'failed_companies': defaultdict(int),  # company_id -> failure count
            'failure_timestamps': [],              # list of failure timestamps
            'concurrent_stats': defaultdict(lambda: {'total': 0, 'failed': 0}),  # concurrent_requests -> stats
            'sequential_failures': 0,              # track consecutive failures
            'max_sequential_failures': 0,          # max consecutive failures seen
            'current_concurrent': 0,               # current number of concurrent requests
            'success_by_batch': [],               # success rate per batch
        }
        
        # Keep track of in-flight requests
        self.active_requests = set()
        
        # Track successful companies for comparison
        self.successful_companies = set()
        
        # Error tracking
        if not hasattr(self, 'error_counts'):
            self.error_counts = {
                400: {'count': 0, 'description': 'Bad Request'},
                401: {'count': 0, 'description': 'Unauthorized'},
                403: {'count': 0, 'description': 'Forbidden'},
                404: {'count': 0, 'description': 'Not Found'},
                429: {'count': 0, 'description': 'Rate Limited'},
                500: {'count': 0, 'description': 'Server Error'},
                502: {'count': 0, 'description': 'Bad Gateway'},
                503: {'count': 0, 'description': 'Service Unavailable'},
                504: {'count': 0, 'description': 'Gateway Timeout'},
                'timeout': {'count': 0, 'description': 'Request Timeout'},
                'other': {'count': 0, 'description': 'Other Errors'}
            }
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Set up logging"""
        self.logger = logging.getLogger('async_founders_processor')
        self.logger.setLevel(logging.INFO)
        
        log_dir = os.path.join(Config.OUTPUT_DIR, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'async_founders_processor_{datetime.now().strftime("%Y%m%d_%H%M")}.log')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    async def get_founders(self, session: aiohttp.ClientSession, company_id: str) -> Optional[Dict]:
        """Enhanced get_founders with detailed tracking"""
        # Track concurrent requests
        self.failure_analysis['current_concurrent'] = len(self.active_requests)
        self.active_requests.add(company_id)
        
        try:
            await self.rate_limiter.acquire()
            
            url = f"https://data.api.aviato.co/company/{company_id}/founders"
            params = {"perPage": 100, "page": 0}
            
            async with session.get(url, params=params, headers=self.headers, timeout=10) as response:
                # Track response for concurrency analysis
                concurrent_count = len(self.active_requests)
                self.failure_analysis['concurrent_stats'][concurrent_count]['total'] += 1
                
                if response.status == 200:
                    data = await response.json()
                    self.successful_calls += 1
                    self.successful_companies.add(company_id)
                    if data and 'founders' in data:
                        self.total_founders += len(data['founders'])
                    
                    # Reset sequential failures on success
                    self.failure_analysis['sequential_failures'] = 0
                    return {company_id: data}
                else:
                    self.failed_calls += 1
                    self.failure_analysis['failed_companies'][company_id] += 1
                    self.failure_analysis['failure_timestamps'].append(datetime.now())
                    self.failure_analysis['concurrent_stats'][concurrent_count]['failed'] += 1
                    
                    # Track specific error
                    if response.status in self.error_counts:
                        self.error_counts[response.status]['count'] += 1
                    else:
                        self.error_counts['other']['count'] += 1
                    
                    # Track sequential failures
                    self.failure_analysis['sequential_failures'] += 1
                    self.failure_analysis['max_sequential_failures'] = max(
                        self.failure_analysis['max_sequential_failures'],
                        self.failure_analysis['sequential_failures']
                    )
                    
                    error_body = await response.text()
                    self.logger.error(f"Error {response.status} for company {company_id}. Response: {error_body[:200]}")
                    return None
                    
        except asyncio.TimeoutError:
            self.failed_calls += 1
            self.error_counts['timeout']['count'] += 1
            self.failure_analysis['failed_companies'][company_id] += 1
            self.logger.error(f"Timeout for company {company_id}")
            return None
        except Exception as e:
            self.failed_calls += 1
            self.error_counts['other']['count'] += 1
            self.failure_analysis['failed_companies'][company_id] += 1
            self.logger.error(f"Error processing company {company_id}: {str(e)}")
            return None
        finally:
            self.active_requests.remove(company_id)

    async def process_batch(self, session: aiohttp.ClientSession, company_ids: List[str]):
        """Process batch with enhanced analysis"""
        batch_start = time.time()
        initial_success_count = self.successful_calls
        
        tasks = [self.get_founders(session, company_id) for company_id in company_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze batch performance
        batch_success = sum(1 for r in results if r is not None)
        self.failure_analysis['success_by_batch'].append({
            'batch_size': len(company_ids),
            'success_rate': batch_success / len(company_ids),
            'concurrent_requests': len(self.active_requests),
            'duration': time.time() - batch_start
        })
        
        for result in results:
            if isinstance(result, dict):
                self.buffer.update(result)

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

    def print_error_summary(self):
        """Print summary of errors encountered"""
        if hasattr(self, 'error_counts'):
            print("\nError Summary:")
            print("-------------")
            for status, data in self.error_counts.items():
                if data['count'] > 0:
                    print(f"{data['description']} ({status}): {data['count']} occurrences")
            print("-------------")
    
    def analyze_failures(self) -> Dict:
        """Analyze failure patterns"""
        analysis = {
            'pattern_analysis': self._analyze_patterns(),
            'concurrency_analysis': self._analyze_concurrency(),
            'company_analysis': self._analyze_company_failures()
        }
        
        return analysis

    def _analyze_patterns(self) -> Dict:
        """Analyze temporal patterns in failures"""
        if not self.failure_analysis['failure_timestamps']:
            return {"message": "No failures recorded"}
            
        timestamps = self.failure_analysis['failure_timestamps']
        
        # Analyze time-based patterns
        time_patterns = defaultdict(int)
        for ts in timestamps:
            hour = ts.hour
            if 5 <= hour < 12:
                time_patterns['morning'] += 1
            elif 12 <= hour < 17:
                time_patterns['afternoon'] += 1
            else:
                time_patterns['evening'] += 1
        
        return {
            "time_distribution": dict(time_patterns),
            "max_sequential_failures": self.failure_analysis['max_sequential_failures'],
            "total_failures": len(timestamps),
            "failure_rate": len(timestamps) / (self.successful_calls + self.failed_calls)
        }

    def _analyze_concurrency(self) -> Dict:
        """Analyze impact of concurrent requests"""
        concurrency_stats = {}
        
        for concurrent_count, stats in self.failure_analysis['concurrent_stats'].items():
            if stats['total'] > 0:
                failure_rate = stats['failed'] / stats['total']
                concurrency_stats[concurrent_count] = {
                    'total_requests': stats['total'],
                    'failed_requests': stats['failed'],
                    'failure_rate': failure_rate
                }
        
        return {
            "concurrency_impact": concurrency_stats,
            "optimal_concurrency": self._find_optimal_concurrency(concurrency_stats)
        }

    def _analyze_company_failures(self) -> Dict:
        """Analyze patterns in failed company IDs"""
        multiple_failures = {
            company_id: count 
            for company_id, count in self.failure_analysis['failed_companies'].items() 
            if count > 1
        }
        
        return {
            "companies_with_multiple_failures": len(multiple_failures),
            "max_failures_per_company": max(multiple_failures.values()) if multiple_failures else 0,
            "total_unique_failed_companies": len(self.failure_analysis['failed_companies']),
            "companies_with_most_failures": dict(sorted(
                multiple_failures.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])  # Top 10 failing companies
        }

    def _find_optimal_concurrency(self, concurrency_stats: Dict) -> int:
        """Find optimal concurrency level based on failure rates"""
        if not concurrency_stats:
            return self.batch_size
            
        # Find the concurrency level with lowest failure rate
        optimal = min(
            concurrency_stats.items(),
            key=lambda x: (x[1]['failure_rate'], -x[0])  # Prefer higher concurrency if rates are equal
        )
        
        return optimal[0]

    async def process_companies(self, company_ids: List[str]):
        """Main processing function with enhanced analysis"""
        total_companies = len(company_ids)
        self.start_time = time.time()
        
        print(f"\nProcessing {total_companies} companies...")
        
        try:
            async with aiohttp.ClientSession() as session:
                for i in range(0, total_companies, self.batch_size):
                    batch = company_ids[i:i + self.batch_size]
                    await self.process_batch(session, batch)
                    
                    if (i + len(batch)) % self.progress_interval == 0:
                        elapsed = time.time() - self.start_time
                        rate = (i + len(batch)) / elapsed
                        print(f"\nProgress: {i + len(batch)}/{total_companies} "
                              f"({((i + len(batch))/total_companies)*100:.1f}%)")
                        print(f"Processing rate: {rate:.1f} companies/second")
                        print(f"Total founders found: {self.total_founders}")
                        print(f"Success rate: {self.successful_calls/(self.successful_calls + self.failed_calls)*100:.1f}%")
                        self.print_error_summary()
                
                await self.save_results()
            
            elapsed = time.time() - self.start_time
            print(f"\nProcessing completed in {elapsed/60:.1f} minutes")
            print(f"Total companies processed: {self.successful_calls}")
            print(f"Total founders found: {self.total_founders}")
            print(f"Average rate: {total_companies/elapsed:.1f} companies/second")
            
            # Final analysis
            print("\nAnalyzing failure patterns...")
            analysis = self.analyze_failures()
            
            print("\nFailure Pattern Analysis:")
            print("-------------------------")
            pattern_analysis = analysis['pattern_analysis']
            print(f"Time-based distribution: {pattern_analysis['time_distribution']}")
            print(f"Maximum sequential failures: {pattern_analysis['max_sequential_failures']}")
            print(f"Overall failure rate: {pattern_analysis['failure_rate']:.2%}")
            
            print("\nConcurrency Analysis:")
            print("--------------------")
            concurrency = analysis['concurrency_analysis']
            print(f"Optimal concurrency level: {concurrency['optimal_concurrency']}")
            
            print("\nCompany Failure Analysis:")
            print("-----------------------")
            company_analysis = analysis['company_analysis']
            print(f"Companies with multiple failures: {company_analysis['companies_with_multiple_failures']}")
            print(f"Maximum failures for a single company: {company_analysis['max_failures_per_company']}")
            
            # Save detailed analysis to file
            analysis_file = os.path.join(Config.OUTPUT_DIR, "founders_analysis.json")
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"\nDetailed analysis saved to: {analysis_file}")
            
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
            await self.save_results()
            elapsed = time.time() - self.start_time
            print(f"Processed {self.successful_calls} companies in {elapsed/60:.1f} minutes")
            self.print_error_summary()
        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}")
            print(f"\nError during processing: {str(e)}")
            await self.save_results()
            self.print_error_summary()