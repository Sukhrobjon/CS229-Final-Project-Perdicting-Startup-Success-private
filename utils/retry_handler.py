# /utils/retry_handler.py

import time
from typing import Dict, Optional, Callable
import logging
import requests
from datetime import datetime

class RetryHandler:
    def __init__(self, max_retries: int = 3, base_wait: int = 5):
        self.max_retries = max_retries
        self.base_wait = base_wait
        self.logger = logging.getLogger('retry_handler')
        
        # Retry statistics
        self.retry_counts = {
            'rate_limit': 0,
            'server_error': 0,
            'network_error': 0,
            'timeout': 0
        }
        
        # Error tracking
        self.error_history = []

    def handle_rate_limit(self, response: requests.Response, retry_func: Callable) -> Optional[Dict]:
        """Handle rate limit errors with retry"""
        retry_after = int(response.headers.get('Retry-After', 60))
        self.retry_counts['rate_limit'] += 1
        self.logger.warning(f"Rate limit hit, waiting {retry_after} seconds")
        time.sleep(retry_after)
        return retry_func()

    def handle_server_error(self, status_code: int, retry_func: Callable) -> Optional[Dict]:
        """Handle server errors with progressive backoff"""
        self.retry_counts['server_error'] += 1
        
        for attempt in range(self.max_retries):
            wait_time = (attempt + 1) * self.base_wait
            self.logger.warning(
                f"Server error {status_code}, attempt {attempt + 1}/{self.max_retries}, "
                f"waiting {wait_time}s"
            )
            time.sleep(wait_time)
            
            try:
                result = retry_func()
                if result:
                    return result
            except Exception as e:
                self.logger.error(f"Retry attempt {attempt + 1} failed: {str(e)}")
                
        return None

    def handle_network_error(self, error: Exception, retry_func: Callable) -> Optional[Dict]:
        """Handle network errors with retries"""
        self.retry_counts['network_error'] += 1
        
        for attempt in range(self.max_retries):
            wait_time = (attempt + 1) * self.base_wait
            self.logger.warning(
                f"Network error, attempt {attempt + 1}/{self.max_retries}, "
                f"waiting {wait_time}s: {str(error)}"
            )
            time.sleep(wait_time)
            
            try:
                return retry_func()
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Retry attempt {attempt + 1} failed: {str(e)}")
                continue
            
        return None

    def get_stats(self) -> Dict:
        """Get retry statistics"""
        return {
            'retry_counts': self.retry_counts.copy(),
            'total_retries': sum(self.retry_counts.values())
        }

    def log_error(self, error_type: str, details: str):
        """Log error details"""
        self.error_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'details': details
        })