# /processors/process_founders_multi_key.py

import asyncio
import time
from typing import Dict, List
from utils.config import Config
from processors.multi_key_founders_processor import MultiKeyFoundersProcessor

async def process_founders_multi_key(company_ids: List[str], test_mode: bool = True) -> Dict:
    """Process founders data using multiple API keys"""
    print("\nStarting Multi-Key Founders Processing...")
    
    try:
        # Validate API keys
        api_keys = Config.get_valid_api_keys()
        if not api_keys:
            print("Error: No valid API keys found")
            return
            
        # Handle test mode
        if test_mode:
            print("\nRunning in TEST MODE")
            company_ids = company_ids[:1000]  # Use only first 1000 companies
            
        # Print validation info
        print("\nAPI Key Validation:")
        print(f"Number of valid API keys: {len(api_keys)}")
        print(f"Total companies to process: {len(company_ids)}")
        print(f"Companies per key: ~{len(company_ids) // len(api_keys)}")
        
        # Validate output directories
        if not Config.validate_multi_key_config():
            print("Error: Multi-key configuration validation failed")
            return

        # Initialize tracking metrics
        tracking_info = {
            'start_time': time.time(),
            'total_companies': len(company_ids),
            'api_keys': len(api_keys),
            'companies_per_key': len(company_ids) // len(api_keys)
        }
        
        # Initialize and run processor
        processor = MultiKeyFoundersProcessor(api_keys)
        result = await processor.process_companies(company_ids, tracking_info)
        
        # Print detailed summary
        print_multi_key_summary(result)
        
        return result
        
    except Exception as e:
        print(f"\nError during multi-key founders processing: {str(e)}")
        return None

def print_multi_key_summary(result: Dict):
    """Print detailed summary of multi-key processing"""
    print("\nMulti-Key Processing Summary:")
    print("============================")
    
    # Overall statistics
    print("\nOverall Performance:")
    print(f"Total Runtime: {result['total_runtime']:.2f} minutes")
    print(f"Total Companies Processed: {result['total_processed']}")
    print(f"Total Successful Calls: {result['total_successful']}")
    print(f"Overall Success Rate: {result['success_rate']:.2f}%")
    
    # Per-key statistics
    print("\nPer-Key Performance:")
    for key_num, stats in result['key_stats'].items():
        print(f"\nAPI Key {key_num + 1}:")
        print(f"Companies Processed: {stats['processed']}")
        print(f"Successful Calls: {stats['successful']}")
        print(f"Success Rate: {stats['success_rate']:.2f}%")
        print(f"Founders Found: {stats['founders_found']}")
        print(f"Average Response Time: {stats['avg_response_time']:.2f}s")
        print(f"Error Rate: {stats['error_rate']:.2f}%")
        
    # Error summary
    print("\nError Summary:")
    for key_num, errors in result['error_stats'].items():
        if errors['total_errors'] > 0:
            print(f"\nAPI Key {key_num + 1} Errors:")
            print(f"Total Errors: {errors['total_errors']}")
            print(f"Timeouts: {errors['timeouts']}")
            print(f"Rate Limits: {errors['rate_limits']}")
            print(f"Other Errors: {errors['other']}")
    
    # File information
    print("\nOutput Files:")
    for key_num, file_info in result['file_stats'].items():
        print(f"\nAPI Key {key_num + 1}:")
        print(f"Output File: {file_info['output_file']}")
        print(f"File Size: {file_info['file_size']:.2f} MB")
        print(f"Companies Saved: {file_info['companies_saved']}")