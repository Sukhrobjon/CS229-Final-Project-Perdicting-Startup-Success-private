import time
from typing import Dict
from collections import deque

class RateLimiter:
    def __init__(self, max_requests: int = 10000, time_window: int = 60):
        """
        Initialize rate limiter
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds (default 60)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque(maxlen=max_requests)  # More efficient than list
        self.last_request_time = 0
        self.min_interval = time_window / max_requests  # 0.006 seconds between requests

    def wait_if_needed(self) -> float:
        """
        Wait if needed and return current timestamp
        Returns:
            float: Current timestamp after waiting if necessary
        """
        now = time.time()
        
        # Clean old requests more efficiently
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()
        
        # If at max requests, wait until oldest expires
        if len(self.requests) >= self.max_requests:
            wait_time = self.requests[0] + self.time_window - now
            if wait_time > 0:
                time.sleep(wait_time)
                now = time.time()
        
        # Ensure minimum interval between requests
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
            now = time.time()
        
        self.requests.append(now)
        self.last_request_time = now
        return now

    def get_current_rate(self) -> float:
        """Get current request rate per minute"""
        now = time.time()
        cutoff = now - 60
        # More efficient counting with deque
        return sum(1 for t in self.requests if t > cutoff)

    def get_status(self) -> Dict:
        """Get current rate limiter status"""
        return {
            "max_requests": self.max_requests,
            "time_window": self.time_window,
            "current_requests": len(self.requests),
            "current_rate": self.get_current_rate(),
            "min_interval": self.min_interval
        }