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
        
        # Enhanced error tracking
        self.error_tracking = {
            'rate_limits': {
                'count': 0,
                'timestamps': [],
                'retry_after_values': []
            },
            'timeouts': {
                'count': 0,
                'timestamps': [],
                'response_times': []
            },
            'api_errors': defaultdict(lambda: {
                'count': 0,
                'timestamps': [],
                'messages': []
            })
        }
        
        # Analysis tracking
        self.failure_analysis = {
            'failed_companies': defaultdict(int),
            'failure_timestamps': [],
            'concurrent_stats': defaultdict(lambda: {'total': 0, 'failed': 0}),
            'sequential_failures': 0,
            'max_sequential_failures': 0,
            'current_concurrent': 0,
            'success_by_batch': [],
            'rate_limit_intervals': [],  # Track time between rate limits
            'response_times': [],        # Track successful response times
        }
        
        # Keep track of in-flight requests
        self.active_requests = set()
        
        # Track successful companies for comparison
        self.successful_companies = set()
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Set up logging with enhanced error tracking"""
        self.logger = logging.getLogger('async_founders_processor')
        self.logger.setLevel(logging.INFO)
        
        log_dir = os.path.join(Config.OUTPUT_DIR, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Add separate error log file
        error_log = os.path.join(log_dir, f'async_founders_errors_{datetime.now().strftime("%Y%m%d_%H%M")}.log')
        error_handler = logging.FileHandler(error_log)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(error_handler)
        
        # Regular log file
        log_file = os.path.join(log_dir, f'async_founders_processor_{datetime.now().strftime("%Y%m%d_%H%M")}.log')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def track_rate_limit(self, retry_after: int, company_id: str):
        """Track rate limit occurrence"""
        now = datetime.now()
        self.error_tracking['rate_limits']['count'] += 1
        self.error_tracking['rate_limits']['timestamps'].append(now)
        self.error_tracking['rate_limits']['retry_after_values'].append(retry_after)
        
        # Track intervals between rate limits
        if len(self.error_tracking['rate_limits']['timestamps']) > 1:
            last_two = self.error_tracking['rate_limits']['timestamps'][-2:]
            interval = (last_two[1] - last_two[0]).total_seconds()
            self.failure_analysis['rate_limit_intervals'].append(interval)
        
        self.logger.warning(
            f"Rate limit hit for company {company_id}. "
            f"Retry-After: {retry_after}s. "
            f"Total rate limits: {self.error_tracking['rate_limits']['count']}"
        )

    def track_timeout(self, company_id: str, response_time: float):
        """Track timeout occurrence"""
        now = datetime.now()
        self.error_tracking['timeouts']['count'] += 1
        self.error_tracking['timeouts']['timestamps'].append(now)
        self.error_tracking['timeouts']['response_times'].append(response_time)
        
        self.logger.error(
            f"Timeout for company {company_id}. "
            f"Response time: {response_time:.2f}s. "
            f"Total timeouts: {self.error_tracking['timeouts']['count']}"
        )
    
    async def get_founders(self, session: aiohttp.ClientSession, company_id: str) -> Optional[Dict]:
        """Enhanced get_founders with detailed error tracking"""
        # Track concurrent requests
        self.failure_analysis['current_concurrent'] = len(self.active_requests)
        self.active_requests.add(company_id)
        request_start = time.time()
        
        # Configure timeouts
        timeout = aiohttp.ClientTimeout(
            total=30,     # Total timeout
            connect=10,   # Connection timeout
            sock_read=20  # Socket read timeout
        )
        
        try:
            await self.rate_limiter.acquire()
            
            url = f"https://data.api.aviato.co/company/{company_id}/founders"
            params = {"perPage": 100, "page": 0}
            
            async with session.get(url, 
                                 params=params, 
                                 headers=self.headers, 
                                 timeout=timeout) as response:
                
                response_time = time.time() - request_start
                self.failure_analysis['response_times'].append(response_time)
                
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
                    
                    self.logger.info(
                        f"Successfully processed company {company_id}. "
                        f"Response time: {response_time:.2f}s. "
                        f"Founders found: {len(data.get('founders', []))}"
                    )
                    
                    return {company_id: data}
                    
                elif response.status == 429:  # Rate limit
                    retry_after = int(response.headers.get('Retry-After', 60))
                    self.track_rate_limit(retry_after, company_id)
                    
                    # If we get a rate limit, we'll wait and retry once
                    await asyncio.sleep(retry_after)
                    return await self.get_founders(session, company_id)
                    
                else:
                    self.failed_calls += 1
                    error_body = await response.text()
                    
                    # Track specific API error
                    error_key = str(response.status)
                    self.error_tracking['api_errors'][error_key]['count'] += 1
                    self.error_tracking['api_errors'][error_key]['timestamps'].append(datetime.now())
                    self.error_tracking['api_errors'][error_key]['messages'].append(error_body[:200])
                    
                    self.logger.error(
                        f"API Error for company {company_id}. "
                        f"Status: {response.status}. "
                        f"Response time: {response_time:.2f}s. "
                        f"Response: {error_body[:200]}"
                    )
                    
                    return None
                    
        except asyncio.TimeoutError:
            response_time = time.time() - request_start
            self.track_timeout(company_id, response_time)
            self.failed_calls += 1
            return None
            
        except Exception as e:
            response_time = time.time() - request_start
            self.failed_calls += 1
            
            self.logger.error(
                f"Unexpected error for company {company_id}. "
                f"Error: {str(e)}. "
                f"Response time: {response_time:.2f}s"
            )
            
            return None
            
        finally:
            self.active_requests.remove(company_id)

    async def process_batch(self, session: aiohttp.ClientSession, company_ids: List[str]):
        """Process batch with enhanced error tracking"""
        batch_start = time.time()
        initial_success_count = self.successful_calls
        
        tasks = [self.get_founders(session, company_id) for company_id in company_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze batch performance
        batch_success = sum(1 for r in results if r is not None)
        batch_duration = time.time() - batch_start
        
        self.failure_analysis['success_by_batch'].append({
            'batch_size': len(company_ids),
            'success_rate': batch_success / len(company_ids),
            'concurrent_requests': len(self.active_requests),
            'duration': batch_duration,
            'rate_per_second': batch_success / batch_duration if batch_duration > 0 else 0
        })
        
        for result in results:
            if isinstance(result, dict):
                self.buffer.update(result)

    def print_error_summary(self):
        """Print detailed error summary"""
        print("\nError Summary:")
        print("-------------")
        
        # Rate Limits
        if self.error_tracking['rate_limits']['count'] > 0:
            print(f"Rate Limits: {self.error_tracking['rate_limits']['count']}")
            if self.error_tracking['rate_limits']['retry_after_values']:
                avg_retry = sum(self.error_tracking['rate_limits']['retry_after_values']) / len(self.error_tracking['rate_limits']['retry_after_values'])
                print(f"Average Retry-After: {avg_retry:.1f}s")
        
        # Timeouts
        if self.error_tracking['timeouts']['count'] > 0:
            print(f"Timeouts: {self.error_tracking['timeouts']['count']}")
            if self.error_tracking['timeouts']['response_times']:
                avg_timeout = sum(self.error_tracking['timeouts']['response_times']) / len(self.error_tracking['timeouts']['response_times'])
                print(f"Average timeout response time: {avg_timeout:.1f}s")
        
        # API Errors
        for status, data in self.error_tracking['api_errors'].items():
            if data['count'] > 0:
                print(f"API Error {status}: {data['count']} occurrences")
                if data['messages']:
                    print(f"Latest message: {data['messages'][-1]}")
        
        print("-------------")

    def analyze_failures(self) -> Dict:
        """Enhanced failure pattern analysis"""
        analysis = {
            'pattern_analysis': self._analyze_patterns(),
            'concurrency_analysis': self._analyze_concurrency(),
            'company_analysis': self._analyze_company_failures(),
            'rate_limit_analysis': self._analyze_rate_limits(),
            'performance_analysis': self._analyze_performance()
        }
        
        return analysis

    def _analyze_patterns(self) -> Dict:
        """Analyze temporal patterns in failures"""
        if not self.failure_analysis['failure_timestamps']:
            return {
                "time_distribution": {},
                "max_sequential_failures": 0,
                "total_failures": 0,
                "failure_rate": 0.0
            }
            
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

    def _analyze_rate_limits(self) -> Dict:
        """Analyze rate limit patterns"""
        rate_limits = self.error_tracking['rate_limits']
        
        analysis = {
            "total_rate_limits": rate_limits['count'],
            "average_retry_after": 0,
            "rate_limit_frequency": 0,
            "rate_limits_per_minute": 0
        }
        
        if rate_limits['retry_after_values']:
            analysis['average_retry_after'] = (
                sum(rate_limits['retry_after_values']) / 
                len(rate_limits['retry_after_values'])
            )
        
        if len(rate_limits['timestamps']) > 1:
            # Calculate average time between rate limits
            intervals = self.failure_analysis['rate_limit_intervals']
            if intervals:
                analysis['rate_limit_frequency'] = sum(intervals) / len(intervals)
            
            # Calculate rate limits per minute
            total_time = (rate_limits['timestamps'][-1] - rate_limits['timestamps'][0]).total_seconds()
            if total_time > 0:
                analysis['rate_limits_per_minute'] = (rate_limits['count'] * 60) / total_time
        
        return analysis

    def _analyze_performance(self) -> Dict:
        """Analyze performance metrics"""
        response_times = self.failure_analysis['response_times']
        
        if not response_times:
            return {
                "average_response_time": 0,
                "min_response_time": 0,
                "max_response_time": 0,
                "success_rate": 0
            }
        
        return {
            "average_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "success_rate": self.successful_calls / (self.successful_calls + self.failed_calls)
        }

    async def process_companies(self, company_ids: List[str]):
        """Main processing function with enhanced analysis"""
        total_companies = len(company_ids)
        self.start_time = time.time()
        
        print(f"\nProcessing {total_companies} companies...")
        
        try:
            # Configure connection pooling
            connector = aiohttp.TCPConnector(
                limit=50,              # Maximum number of concurrent connections
                force_close=False,     # Keep connections alive
                enable_cleanup_closed=True
            )
            
            async with aiohttp.ClientSession(connector=connector) as session:
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
            
            print("\nRate Limit Analysis:")
            print("-------------------")
            rate_limit_analysis = analysis['rate_limit_analysis']
            print(f"Total rate limits: {rate_limit_analysis['total_rate_limits']}")
            print(f"Average retry after: {rate_limit_analysis['average_retry_after']:.1f}s")
            print(f"Rate limits per minute: {rate_limit_analysis['rate_limits_per_minute']:.1f}")
            
            print("\nPerformance Analysis:")
            print("--------------------")
            perf = analysis['performance_analysis']
            print(f"Average response time: {perf['average_response_time']:.3f}s")
            print(f"Min response time: {perf['min_response_time']:.3f}s")
            print(f"Max response time: {perf['max_response_time']:.3f}s")
            
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