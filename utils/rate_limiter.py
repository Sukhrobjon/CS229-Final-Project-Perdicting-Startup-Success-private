# /utils/rate_limiter.py

import time
import os
import logging
from typing import Dict
from collections import deque
from datetime import datetime
from utils.config import Config

class RateLimiter:
    def __init__(
        self,
        max_requests: int = 15,    # Increased from 10 to 15 TODO: bring it back to 20 for funding
        time_window: int = 1,      # 1 second window
        output_dir: str = Config.OUTPUT_DIR
    ):
        self.max_requests = max_requests
        self.time_window = time_window
        self.output_dir = output_dir
        
        # Request tracking
        self.requests = deque(maxlen=max_requests)
        self.request_times = deque(maxlen=1000)  # Track last 1000 request times
        
        # Performance tracking
        self.start_time = time.time()
        self.last_request_time = self.start_time
        
        # Error tracking
        self.error_counts = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'timeouts': 0,
            'rate_limits': 0
        }
        
        # Dynamic rate adjustment
        self.avg_response_times = deque(maxlen=100)  # Track last 100 response times
        self.min_requests = 15      # Minimum requests per second
        self.max_allowed = 20      # Maximum allowed requests per second
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Set up logging"""
        self.logger = logging.getLogger('rate_limiter')
        self.logger.setLevel(logging.INFO)
        
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(
            log_dir, 
            f'rate_limiter_{datetime.now().strftime("%Y%m%d_%H%M")}.log'
        )
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)

    def wait_if_needed(self) -> float:
        """Optimized wait_if_needed with dynamic rate adjustment"""
        now = time.time()
        
        # Clean old requests
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()
        
        # Calculate current rate
        current_rate = len(self.requests)
        
        # Adjust max_requests based on average response time
        if self.avg_response_times:
            avg_response = sum(self.avg_response_times) / len(self.avg_response_times)
            if avg_response < 0.5:  # If responses are fast
                self.max_requests = min(self.max_requests + 1, self.max_allowed)
            elif avg_response > 2.0:  # If responses are slow
                self.max_requests = max(self.max_requests - 1, self.min_requests)
        
        # If we have too many requests in the window, wait
        if current_rate >= self.max_requests:
            wait_time = self.requests[0] + self.time_window - now
            if wait_time > 0:
                time.sleep(wait_time)
                now = time.time()
        
        # Add this request
        self.requests.append(now)
        self.request_times.append(now)
        
        return now

    def record_result(self, success: bool, response_time: float = None, 
                     is_timeout: bool = False, is_rate_limit: bool = False):
        """Record request result with response time"""
        self.error_counts['total_requests'] += 1
        
        if success:
            self.error_counts['successful_requests'] += 1
            if response_time is not None:
                self.avg_response_times.append(response_time)
        else:
            self.error_counts['failed_requests'] += 1
            if is_timeout:
                self.error_counts['timeouts'] += 1
            if is_rate_limit:
                self.error_counts['rate_limits'] += 1
                self.logger.warning("Rate limit encountered")

    def get_current_rate(self) -> float:
        """Calculate current request rate"""
        now = time.time()
        cutoff = now - self.time_window
        recent_requests = sum(1 for t in self.request_times if t > cutoff)
        return recent_requests / self.time_window

    def get_stats(self) -> Dict:
        """Get current statistics"""
        current_rate = self.get_current_rate()
        total_requests = self.error_counts['total_requests']
        
        stats = {
            'current_rate': current_rate,
            'max_requests': self.max_requests,
            'error_counts': self.error_counts.copy(),
            'success_rate': (self.error_counts['successful_requests'] / 
                           max(total_requests, 1)),
            'error_rate': (self.error_counts['failed_requests'] / 
                          max(total_requests, 1)),
            'uptime': time.time() - self.start_time,
            'requests_in_window': len(self.requests)
        }
        
        if self.avg_response_times:
            stats['avg_response_time'] = sum(self.avg_response_times) / len(self.avg_response_times)
        
        return stats

    def should_kill_process(self) -> tuple[bool, str]:
        """Check if process should be killed based on monitoring"""
        # Check error rate (if we have enough data)
        if self.error_counts['total_requests'] >= 100:
            error_rate = (self.error_counts['failed_requests'] / 
                         self.error_counts['total_requests'])
            if error_rate > 0.20:  # 20% error threshold
                return True, f"Error rate too high: {error_rate:.2%}"
        
        # Check timeout rate
        if self.error_counts['total_requests'] > 0:
            timeout_rate = (self.error_counts['timeouts'] / 
                          self.error_counts['total_requests'])
            if timeout_rate > 0.10:  # 10% timeout threshold (reduced from previous)
                return True, f"Timeout rate too high: {timeout_rate:.2%}"
        
        return False, ""