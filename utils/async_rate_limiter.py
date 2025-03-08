# /utils/async_rate_limiter.py

import time
import asyncio
from typing import Dict, Optional, Tuple
import logging
import os
from datetime import datetime
from utils.config import Config

class AsyncRateLimiter:
    def __init__(
        self,
        max_requests: int = 150,
        time_window: int = 1,
        output_dir: str = Config.OUTPUT_DIR
    ):
        self.max_requests = max_requests
        self.time_window = time_window
        self.output_dir = output_dir
        
        # Token bucket parameters
        self.tokens = max_requests
        self.last_update = time.time()
        self.rate = max_requests / time_window
        
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
        
        # Performance tracking
        self.start_time = time.time()
        self.request_times = []
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Set up logging for rate limiter"""
        self.logger = logging.getLogger('async_rate_limiter')
        self.logger.setLevel(logging.INFO)
        
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'async_rate_limiter_{datetime.now().strftime("%Y%m%d_%H%M")}.log')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def _update_tokens(self):
        """Update the token count based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_update
        new_tokens = elapsed * self.rate
        self.tokens = min(self.max_requests, self.tokens + new_tokens)
        self.last_update = now

    async def acquire(self):
        """Acquire a token asynchronously"""
        while True:
            self._update_tokens()
            if self.tokens >= 1:
                self.tokens -= 1
                return
            wait_time = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)

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

    def get_stats(self) -> Dict:
        """Get current statistics"""
        now = time.time()
        return {
            'error_counts': self.error_counts.copy(),
            'uptime': now - self.start_time,
            'total_requests': self.error_counts['total_requests'],
            'success_rate': self.error_counts['successful_requests'] / max(self.error_counts['total_requests'], 1),
            'error_rate': self.error_counts['failed_requests'] / max(self.error_counts['total_requests'], 1)
        }