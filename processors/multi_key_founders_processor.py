# /processors/multi_key_founders_processor.py

import asyncio
import aiohttp
from typing import Dict, List, Optional, Set
import time
import json
import os
from datetime import datetime
import logging
from collections import defaultdict
from utils.config import Config
from utils.async_rate_limiter import AsyncRateLimiter

class MultiKeyFoundersProcessor:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.num_keys = len(api_keys)
        
        # Create output directories for each key
        self.output_dirs = self._setup_output_dirs()
        
        # Initialize tracking
        self.tracking = {
            'start_time': None,
            'per_key_stats': defaultdict(dict),
            'error_tracking': defaultdict(lambda: defaultdict(int)),
            'response_times': defaultdict(list)
        }
        
        # Setup logging
        self.setup_logging()
        
    def _setup_output_dirs(self) -> Dict[str, str]:
        """Create separate output directories for each API key"""
        dirs = {}
        for i, key in enumerate(self.api_keys):
            key_dir = os.path.join(Config.MULTI_KEY_OUTPUT_DIR, f'api_key_{i+1}')
            os.makedirs(key_dir, exist_ok=True)
            dirs[key] = key_dir
        return dirs
        
    def setup_logging(self):
        """Set up logging with separate files for each key"""
        self.loggers = {}
        for i, key in enumerate(self.api_keys):
            logger = logging.getLogger(f'multi_key_processor_{i+1}')
            logger.setLevel(logging.INFO)
            
            log_dir = os.path.join(Config.OUTPUT_DIR, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, f'multi_key_{i+1}_{datetime.now().strftime("%Y%m%d_%H%M")}.log')
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
            
            self.loggers[key] = logger
            
    def split_companies(self, company_ids: List[str]) -> List[List[str]]:
        """Split companies evenly among API keys"""
        total_companies = len(company_ids)
        chunk_size = total_companies // self.num_keys
        
        chunks = []
        for i in range(self.num_keys):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.num_keys - 1 else total_companies
            chunks.append(company_ids[start_idx:end_idx])
            
        return chunks
        
    async def process_company(self, session: aiohttp.ClientSession, 
                            company_id: str, api_key: str, key_index: int) -> Optional[Dict]:
        """Process a single company with specific API key"""
        start_time = time.time()
        
        try:
            url = f"https://data.api.aviato.co/company/{company_id}/founders"
            headers = {"Authorization": f"Bearer {api_key}"}
            params = {"perPage": 20, "page": 0}
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with session.get(url, params=params, headers=headers, 
                                 timeout=timeout) as response:
                response_time = time.time() - start_time
                self.tracking['response_times'][key_index].append(response_time)
                
                if response.status == 200:
                    data = await response.json()
                    self.track_success(key_index, data)
                    return {company_id: data}
                elif response.status == 429:
                    self.track_error(key_index, 'rate_limit')
                    retry_after = int(response.headers.get('Retry-After', 60))
                    self.loggers[api_key].warning(
                        f"Rate limit hit for company {company_id}. "
                        f"Retry-After: {retry_after}s"
                    )
                    return None
                else:
                    self.track_error(key_index, 'other')
                    error_body = await response.text()
                    self.loggers[api_key].error(
                        f"API Error for company {company_id}. "
                        f"Status: {response.status}. Response: {error_body[:200]}"
                    )
                    return None
                    
        except asyncio.TimeoutError:
            self.track_error(key_index, 'timeout')
            self.loggers[api_key].error(f"Timeout for company {company_id}")
            return None
        except Exception as e:
            self.track_error(key_index, 'other')
            self.loggers[api_key].error(
                f"Unexpected error for company {company_id}: {str(e)}"
            )
            return None
            
    def track_success(self, key_index: int, data: Dict):
        """Track successful request"""
        stats = self.tracking['per_key_stats'][key_index]
        stats['successful_requests'] = stats.get('successful_requests', 0) + 1
        stats['founders_found'] = stats.get('founders_found', 0) + len(data.get('founders', []))
        
    def track_error(self, key_index: int, error_type: str):
        """Track error occurrence"""
        self.tracking['error_tracking'][key_index][error_type] += 1
        
    async def process_chunk(self, company_ids: List[str], api_key: str, key_index: int) -> Dict:
        """Process a chunk of companies with specific API key"""
        connector = aiohttp.TCPConnector(limit=20)  # Limit concurrent connections
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            results = {}
            
            for company_id in company_ids:
                tasks.append(self.process_company(session, company_id, api_key, key_index))
                
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in chunk_results:
                if isinstance(result, dict):
                    results.update(result)
                    
            return results
            
    async def save_results(self, results: Dict, api_key: str):
        """Save results for specific API key"""
        output_file = os.path.join(self.output_dirs[api_key], Config.OUTPUT_FILES['founders'])
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.loggers[api_key].info(f"Saved {len(results)} results to {output_file}")
        except Exception as e:
            self.loggers[api_key].error(f"Error saving results: {str(e)}")
            
    def get_tracking_summary(self) -> Dict:
        """Get comprehensive tracking summary"""
        summary = {
            'total_runtime': (time.time() - self.tracking['start_time']) / 60,
            'key_stats': {},
            'error_stats': {},
            'file_stats': {}
        }
        
        total_processed = 0
        total_successful = 0
        
        for key_idx in range(self.num_keys):
            stats = self.tracking['per_key_stats'][key_idx]
            response_times = self.tracking['response_times'][key_idx]
            
            processed = stats.get('total_requests', 0)
            successful = stats.get('successful_requests', 0)
            total_processed += processed
            total_successful += successful
            
            summary['key_stats'][key_idx] = {
                'processed': processed,
                'successful': successful,
                'success_rate': (successful / processed * 100) if processed > 0 else 0,
                'founders_found': stats.get('founders_found', 0),
                'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                'error_rate': ((processed - successful) / processed * 100) if processed > 0 else 0
            }
            
            summary['error_stats'][key_idx] = {
                'total_errors': sum(self.tracking['error_tracking'][key_idx].values()),
                'timeouts': self.tracking['error_tracking'][key_idx]['timeout'],
                'rate_limits': self.tracking['error_tracking'][key_idx]['rate_limit'],
                'other': self.tracking['error_tracking'][key_idx]['other']
            }
            
            # Add file statistics
            output_file = os.path.join(
                self.output_dirs[self.api_keys[key_idx]], 
                Config.OUTPUT_FILES['founders']
            )
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert to MB
                with open(output_file, 'r') as f:
                    companies_saved = len(json.load(f))
                
                summary['file_stats'][key_idx] = {
                    'output_file': output_file,
                    'file_size': file_size,
                    'companies_saved': companies_saved
                }
        
        summary['total_processed'] = total_processed
        summary['total_successful'] = total_successful
        summary['success_rate'] = (total_successful / total_processed * 100) if total_processed > 0 else 0
        
        return summary
        
    async def process_companies(self, company_ids: List[str], 
                              tracking_info: Optional[Dict] = None) -> Dict:
        """Main processing function"""
        self.tracking['start_time'] = time.time()
        
        try:
            # Split companies among available keys
            chunks = self.split_companies(company_ids)
            
            # Process chunks in parallel
            tasks = []
            for i, (chunk, api_key) in enumerate(zip(chunks, self.api_keys)):
                self.loggers[api_key].info(
                    f"Starting processing of {len(chunk)} companies with API key {i+1}"
                )
                tasks.append(self.process_chunk(chunk, api_key, i))
            
            # Wait for all chunks to complete
            results = await asyncio.gather(*tasks)
            
            # Save results for each key
            for i, (result, api_key) in enumerate(zip(results, self.api_keys)):
                await self.save_results(result, api_key)
            
            # Return summary
            return self.get_tracking_summary()
            
        except Exception as e:
            print(f"Error during multi-key processing: {str(e)}")
            return self.get_tracking_summary()