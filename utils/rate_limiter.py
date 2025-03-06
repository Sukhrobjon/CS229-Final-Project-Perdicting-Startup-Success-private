# utils/rate_limiter.py

import time

class RateLimiter:
    def __init__(self, max_requests: int, time_window: int = 60):
        """
        Initialize rate limiter
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds (default 60)
        """
        self.max_requests = max_requests  # 1000 requests
        self.time_window = time_window    # per 60 seconds
        self.requests = []
        self.min_interval = time_window / max_requests  # 0.06 seconds between requests

    def wait_if_needed(self):
        """Wait if rate limit is about to be exceeded"""
        now = time.time()
        
        # Remove requests older than time window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        # If at max requests, wait until oldest request expires
        if len(self.requests) >= self.max_requests:
            sleep_time = self.requests[0] + self.time_window - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.requests = self.requests[1:]
        
        # Add current request
        self.requests.append(now)
        
        # Ensure minimum interval between requests
        if len(self.requests) > 1:
            elapsed = now - self.requests[-2]
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

    def get_current_rate(self) -> float:
        """Get current request rate per minute"""
        now = time.time()
        # Count requests in last minute
        recent_requests = len([req for req in self.requests 
                             if now - req < 60])
        return recent_requests

    def get_status(self) -> dict:
        """Get current rate limiter status"""
        return {
            "max_requests": self.max_requests,
            "time_window": self.time_window,
            "current_requests": len(self.requests),
            "current_rate": self.get_current_rate()
        }