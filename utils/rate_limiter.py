# /utils/rate_limiter.py

import time
import os
import shutil
from collections import deque
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging
from utils.config import Config

class EnhancedRateLimiter:
    def __init__(
        self,
        max_requests: int = 150,  # 150 requests per second
        time_window: int = 1,     # 1 second window
        output_dir: str = Config.OUTPUT_DIR
    ):
        self.max_requests = max_requests
        self.time_window = time_window
        self.output_dir = output_dir
        self.requests = deque(maxlen=max_requests)
        
        # Debug mode
        self.debug = True
        
        # Monitoring
        self.error_counts = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'timeouts': 0,
            'rate_limits': 0
        }
        
        # Rolling window for error rate calculation
        self.recent_results = deque(maxlen=1000)  # Last 1000 requests
        
        # Performance tracking
        self.start_time = time.time()
        self.last_check_time = self.start_time
        self.request_times = deque(maxlen=1000)  # Track last 1000 request times
        
        # Disk monitoring
        self.max_disk_usage = 500 * 1024 * 1024  # 500MB in bytes
        
        # Setup logging
        self.setup_logging()
        
        self.logger.info(f"Enhanced Rate Limiter initialized: {max_requests} requests per {time_window} second(s)")

    def setup_logging(self):
        """Set up logging for rate limiter"""
        self.logger = logging.getLogger('rate_limiter')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        log_file = os.path.join(log_dir, f'rate_limiter_{datetime.now().strftime("%Y%m%d_%H%M")}.log')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def wait_if_needed(self) -> float:
        """Enhanced version of wait_if_needed with monitoring"""
        now = time.time()
        
        if self.debug:
            print("\nRate Limiter Debug Info:")
            print(f"Current queue size: {len(self.requests)}")
            print(f"Time window: {self.time_window} seconds")
            print(f"Max requests: {self.max_requests}")
            if self.requests:
                print(f"Oldest request: {now - self.requests[0]:.3f} seconds ago")
                print(f"Current rate: {len(self.requests)/self.time_window:.1f} requests/second")
        
        # Clean old requests
        old_size = len(self.requests)
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()
        new_size = len(self.requests)
        
        if self.debug and old_size != new_size:
            print(f"Cleaned {old_size - new_size} old requests")
        
        # Calculate current rate
        current_rate = len(self.requests) / self.time_window if self.requests else 0
        
        # If at max requests, wait
        if len(self.requests) >= self.max_requests:
            wait_time = self.requests[0] + self.time_window - now
            if self.debug:
                print(f"At max requests. Need to wait: {wait_time:.3f} seconds")
            if wait_time > 0:
                time.sleep(wait_time)
                now = time.time()
        
        # Calculate minimum interval between requests
        min_interval = self.time_window / self.max_requests
        
        # Ensure minimum interval between requests
        if self.requests:
            elapsed = now - self.requests[-1]
            if elapsed < min_interval:
                wait_time = min_interval - elapsed
                if self.debug:
                    print(f"Enforcing minimum interval. Waiting: {wait_time:.3f} seconds")
                time.sleep(wait_time)
                now = time.time()
        
        self.requests.append(now)
        self.request_times.append(now)
        
        if self.debug:
            print(f"New rate: {len(self.requests)/self.time_window:.1f} requests/second")
        
        return now

    def check_disk_usage(self) -> Dict:
        """Monitor disk usage in output directory"""
        try:
            total_size = 0
            for dirpath, _, filenames in os.walk(self.output_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            
            is_critical = total_size > self.max_disk_usage
            if is_critical:
                self.logger.warning(f"Disk usage critical: {total_size / (1024*1024):.2f}MB")
            
            return {
                'current_usage': total_size,
                'max_allowed': self.max_disk_usage,
                'is_critical': is_critical
            }
        except Exception as e:
            self.logger.error(f"Error checking disk usage: {str(e)}")
            return {
                'error': str(e),
                'is_critical': False
            }

    def should_kill_process(self) -> Tuple[bool, str]:
        """Check if process should be killed based on monitoring"""
        # Check disk usage
        disk_status = self.check_disk_usage()
        if disk_status.get('is_critical'):
            return True, "Disk usage exceeded 500MB limit"
            
        # Check error rate
        if len(self.recent_results) >= 100:  # Wait for at least 100 requests
            error_rate = sum(1 for x in self.recent_results if not x) / len(self.recent_results)
            if error_rate > 0.10:  # 10% error rate threshold
                return True, f"Error rate too high: {error_rate:.2%}"
        
        # Check timeout rate
        timeout_rate = self.error_counts['timeouts'] / max(self.error_counts['total_requests'], 1)
        if timeout_rate > 0.05:  # 5% timeout threshold
            return True, f"Timeout rate too high: {timeout_rate:.2%}"
            
        return False, ""

    def record_result(self, success: bool, is_timeout: bool = False, is_rate_limit: bool = False):
        """Record the result of a request"""
        if success:
            self.error_counts['successful_requests'] += 1
        else:
            self.error_counts['failed_requests'] += 1
            if is_timeout:
                self.error_counts['timeouts'] += 1
            if is_rate_limit:
                self.error_counts['rate_limits'] += 1
        
        self.error_counts['total_requests'] += 1
        self.recent_results.append(success)

    def get_stats(self) -> Dict:
        """Get current statistics"""
        now = time.time()
        
        # Calculate current rate over the last second
        recent_requests = sum(1 for t in self.request_times if t > now - 1)
        
        return {
            'error_counts': self.error_counts.copy(),
            'current_rate': recent_requests,
            'error_rate': sum(1 for x in self.recent_results if not x) / len(self.recent_results) if self.recent_results else 0,
            'uptime': now - self.start_time,
            'requests_in_window': len(self.requests),
            'window_size': self.time_window
        }

    def set_debug(self, enabled: bool):
        """Enable or disable debug mode"""
        self.debug = enabled
        if enabled:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)