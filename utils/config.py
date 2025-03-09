# /utils/config.py

import os
import glob
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables from .env file
load_dotenv()

class Config:
    # Primary API Configuration (maintain existing for backward compatibility)
    API_KEY = os.getenv('AVIATO_API_KEY', 'default_key')
    
    # Multi-key Configuration
    API_KEYS = [
        os.getenv('AVIATO_API_KEY_1', API_KEY),  # Use primary key as first key if not specified
        os.getenv('AVIATO_API_KEY_2', None),
        os.getenv('AVIATO_API_KEY_3', None),
        os.getenv('AVIATO_API_KEY_4', None)
    ]
    
    BASE_URL = "https://data.api.aviato.co"
    
    # API Endpoints
    ENDPOINTS = {
        'company_enrich': f"{BASE_URL}/company/enrich",
        'funding_rounds': f"{BASE_URL}/company/{{company_id}}/funding-rounds",
        'founders': f"{BASE_URL}/company/{{company_id}}/founders",
        'person_enrich': f"{BASE_URL}/person/bulk/enrich"
    }

    # Processing Configuration
    BATCH_SIZE = 1000
    PERSON_BATCH_SIZE = 20  # Maximum IDs per person enrichment request
    RATE_LIMIT_DELAY = 1  # Delay between API calls in seconds

    # Directory Configuration
    OUTPUT_DIR = "output"
    MULTI_KEY_OUTPUT_DIR = "output/multi_key"
    # NEW: Additional directory configurations for person enrichment
    FOUNDERS_DIR = os.path.join(OUTPUT_DIR, "founders")
    PERSON_DIR = os.path.join(OUTPUT_DIR, "person")
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")  # NEW: Explicit logs directory

    # File Names
    COMPANIES_FILE = "companies.json"
    # NEW: Founders data file for person enrichment
    FOUNDERS_READY_FILE = "founders_data_ready_for_person_enrichment.json"

    # NEW: Full paths for specific files
    FOUNDERS_DATA_PATH = os.path.join(FOUNDERS_DIR, FOUNDERS_READY_FILE)
    
    # Output File Names
    OUTPUT_FILES = {
        'company_enrichment': "company_enrichment.json",
        'funding_rounds': "funding_rounds.json",
        'founders': "founders.json",
        'person_enrichment': "person_enrichment.json"
    }

    @classmethod
    def get_valid_api_keys(cls) -> List[str]:
        """Get list of valid API keys"""
        return [key for key in cls.API_KEYS if key and key != 'default_key']

    @classmethod
    def validate_multi_key_config(cls) -> bool:
        """Validate configuration for multi-key processing"""
        valid_keys = cls.get_valid_api_keys()
        
        if not valid_keys:
            print("Error: No valid API keys found for multi-key processing")
            return False
            
        print(f"\nFound {len(valid_keys)} valid API keys for processing")
        
        # Ensure multi-key output directory exists
        os.makedirs(cls.MULTI_KEY_OUTPUT_DIR, exist_ok=True)
        
        return True

    @classmethod
    def validate_config(cls, mode: str = 'full') -> bool:
        """Validate essential configuration settings"""
        if cls.API_KEY == 'default_key':
            print("Error: API key not found in environment variables")
            return False
            
        # Ensure output directory exists
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
            
        # Only check for input file in full mode
        if mode == 'full':
            # Look for merged files first
            merged_files = glob.glob(os.path.join(cls.OUTPUT_DIR, "companies_merged_*.json"))
            if merged_files:
                # Use the latest merged file
                latest_file = max(merged_files, key=os.path.getctime)
                cls.COMPANIES_FILE = os.path.basename(latest_file)
                print(f"Using latest merged file: {cls.COMPANIES_FILE}")
                return True
                
            # Fall back to default companies.json if no merged files found
            companies_file = os.path.join(cls.OUTPUT_DIR, cls.COMPANIES_FILE)
            if not os.path.exists(companies_file) and not os.path.exists(cls.COMPANIES_FILE):
                print(f"Error: No company data found. Please run search first.")
                return False
        
        return True

    # NEW: Validation method for person enrichment
    @classmethod
    def validate_person_enrichment_config(cls) -> bool:
        """Validate configuration for person enrichment processing"""
        # Ensure all required directories exist
        for directory in [cls.OUTPUT_DIR, cls.FOUNDERS_DIR, cls.PERSON_DIR, cls.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)

        # Check for founders data file
        if not os.path.exists(cls.FOUNDERS_DATA_PATH):
            print(f"Error: Founders data file not found at {cls.FOUNDERS_DATA_PATH}")
            return False

        # Validate API key
        if cls.API_KEY == 'default_key':
            print("Error: API key not found in environment variables")
            return False

        # Validate person enrichment endpoint
        if 'person_enrich' not in cls.ENDPOINTS:
            print("Error: Person enrichment endpoint not configured")
            return False

        print(f"\nConfiguration validated successfully")
        print(f"Using founders data from: {cls.FOUNDERS_DATA_PATH}")
        return True

    # NEW: Helper method to get full output path
    @classmethod
    def get_output_path(cls, file_key: str) -> str:
        """Get full path for output file"""
        return os.path.join(cls.OUTPUT_DIR, cls.OUTPUT_FILES[file_key])