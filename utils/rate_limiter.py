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
        max_requests: int = 20,    # Optimal medium setting
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
        
        # Setup logging
        self.setup_logging()
        
        self.logger.info(
            f"Rate Limiter initialized with medium settings: "
            f"{max_requests} requests per {time_window} second(s)"
        )

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
        """Optimized wait_if_needed with medium settings"""
        now = time.time()
        
        # Clean old requests
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()
        
        # If we have too many requests in the window, wait
        if len(self.requests) >= self.max_requests:
            wait_time = self.requests[0] + self.time_window - now
            if wait_time > 0:
                time.sleep(wait_time)
                now = time.time()
        
        # Add this request
        self.requests.append(now)
        self.request_times.append(now)
        
        return now

    def record_result(self, success: bool, is_timeout: bool = False, 
                     is_rate_limit: bool = False):
        """Record request result"""
        self.error_counts['total_requests'] += 1
        
        if success:
            self.error_counts['successful_requests'] += 1
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
        
        return {
            'current_rate': current_rate,
            'error_counts': self.error_counts.copy(),
            'success_rate': (self.error_counts['successful_requests'] / 
                           max(total_requests, 1)),
            'error_rate': (self.error_counts['failed_requests'] / 
                          max(total_requests, 1)),
            'uptime': time.time() - self.start_time,
            'requests_in_window': len(self.requests)
        }

    def check_disk_usage(self) -> Dict:
        """Monitor disk usage in output directory"""
        try:
            total_size = 0
            for dirpath, _, filenames in os.walk(self.output_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            
            max_size = 900 * 1024 * 1024  # 500MB
            is_critical = total_size > max_size
            
            if is_critical:
                self.logger.warning(f"Disk usage critical: {total_size / (1024*1024):.2f}MB")
            
            return {
                'current_usage': total_size,
                'max_allowed': max_size,
                'is_critical': is_critical
            }
        except Exception as e:
            self.logger.error(f"Error checking disk usage: {str(e)}")
            return {
                'error': str(e),
                'is_critical': False
            }

    def should_kill_process(self) -> tuple[bool, str]:
        """Check if process should be killed based on monitoring"""
        # Check disk usage
        disk_status = self.check_disk_usage()
        if disk_status.get('is_critical'):
            return True, "Disk usage exceeded 500MB limit"
        
        # Check error rate (if we have enough data)
        if self.error_counts['total_requests'] >= 100:
            error_rate = (self.error_counts['failed_requests'] / 
                         self.error_counts['total_requests'])
            if error_rate > 0.10:  # 10% error threshold
                return True, f"Error rate too high: {error_rate:.2%}"
        
        # Check timeout rate
        if self.error_counts['total_requests'] > 0:
            timeout_rate = (self.error_counts['timeouts'] / 
                          self.error_counts['total_requests'])
            if timeout_rate > 0.05:  # 5% timeout threshold
                return True, f"Timeout rate too high: {timeout_rate:.2%}"
        
        return False, ""