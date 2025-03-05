import requests
import json
import time
import os
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    """Base class for all processors"""
    def __init__(self, api_key: str, batch_size: int = 1000):
        self.api_key = api_key
        self.batch_size = batch_size
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.data = {}
        self.companies_processed = 0
        
    def append_to_json(self, filename: str):
        """Append current batch data to JSON file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {}
            
            existing_data.update(self.data)
            
            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            print(f"Data appended to {filename}")
            
        except Exception as e:
            print(f"Error appending to JSON file {filename}: {str(e)}")
    
    def check_and_save_batch(self):
        """Check if batch size reached and save if necessary"""
        self.companies_processed += 1
        
        if self.companies_processed % self.batch_size == 0:
            print(f"\nProcessed {self.companies_processed} companies. Saving batch...")
            self.save_batch()
            self.data = {}  # Reset data after saving
            print("Batch saved successfully!")

    @abstractmethod
    def save_batch(self):
        """Save current batch - to be implemented by child classes"""
        pass

    @abstractmethod
    def process_companies(self, company_ids: List[str]):
        """Process list of companies - to be implemented by child classes"""
        pass