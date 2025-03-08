import os
import glob
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Configuration
    API_KEY = os.getenv('AVIATO_API_KEY', 'default_key')
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

    # File Names
    COMPANIES_FILE = "companies.json"  # Search results file
    
    # Output File Names
    OUTPUT_FILES = {
        'company_enrichment': "company_enrichment.json",
        'funding_rounds': "funding_rounds.json",
        'founders': "founders.json",
        'person_enrichment': "person_enrichment.json"
    }

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