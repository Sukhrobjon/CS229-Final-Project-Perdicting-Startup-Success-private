# /main.py

import glob
from typing import List, Dict, Callable, Optional
import os
import time
import asyncio
from processors.search_processor import SearchProcessor
from processors.company_processor import CompanyEnrichmentProcessor
from processors.funding_processor import FundingRoundsProcessor
from processors.founders_processor import FoundersProcessor
from processors.async_founders_processor import AsyncFoundersProcessor
from processors.person_processor import PersonEnrichmentProcessor
from utils.file_handler import FileHandler
from utils.config import Config
from processors.multi_key_founders_processor import MultiKeyFoundersProcessor 

class ProcessorManager:
    def __init__(self):
        self.processors = {
            '1': {
                'name': 'Company Enrichment',
                'function': self.process_enrichment,
                'description': 'Process company enrichment data'
            },
            '2': {
                'name': 'Funding Rounds',
                'function': self.process_funding,
                'description': 'Process funding rounds data'
            },
            '3': {
                'name': 'Founders',
                'function': self.process_founders,
                'description': 'Process founders data'
            },
            '3a': {
                'name': 'Founders (Async)',
                'function': self.process_founders_async,
                'description': 'Process founders data (Async version)'
            },
            '3m': {                                                           
                'name': 'Founders (Multi-Key)',
                'function': self.process_founders_multi_key,
                'description': 'Process founders data using multiple API keys'
            },
            '4': {
                'name': 'Person Enrichment',
                'function': self.process_person_enrichment,
                'description': 'Process person enrichment data'
            }
        }

    def list_processors(self):
        """Display available processors"""
        print("\nAvailable processors:")
        for key, processor in self.processors.items():
            print(f"{key}. {processor['name']} - {processor['description']}")

    def process_search(self) -> bool:
        """Process company search and save results"""
        print("\nStarting Company Search...")
        try:
            processor = SearchProcessor(Config.API_KEY)
            processor.process_companies(target_companies=1000)
            return True
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return False
    
    def process_enrichment(self, company_ids: List[str]) -> None:
        """Process company enrichment data"""
        print("\nStarting Company Enrichment Processing...")
        try:
            processor = CompanyEnrichmentProcessor(Config.API_KEY, Config.BATCH_SIZE)
            processor.process_companies(company_ids)
        except SystemExit as e:
            print(f"\nProcessor stopped by kill switch: {str(e)}")
        except Exception as e:
            print(f"\nError during enrichment processing: {str(e)}")

    def process_funding(self, company_ids: List[str]) -> None:
        """Process funding rounds data"""
        print("\nStarting Funding Rounds Processing...")
        try:
            processor = FundingRoundsProcessor(Config.API_KEY, Config.BATCH_SIZE)
            processor.process_companies(company_ids)
        except SystemExit as e:
            print(f"\nProcessor stopped by kill switch: {str(e)}")
        except Exception as e:
            print(f"\nError during funding processing: {str(e)}")

    def process_founders(self, company_ids: List[str]) -> None:
        """Process founders data using sync version"""
        print("\nStarting Founders Processing...")
        try:
            processor = FoundersProcessor(Config.API_KEY, Config.BATCH_SIZE)
            processor.process_companies(company_ids)
        except SystemExit as e:
            print(f"\nProcessor stopped by kill switch: {str(e)}")
        except Exception as e:
            print(f"\nError during founders processing: {str(e)}")


    def process_founders_multi_key(self, company_ids: List[str]) -> None:
        """Process founders data using multiple API keys"""
        print("\nStarting Multi-Key Founders Processing...")
        try:
            # Get valid API keys
            api_keys = Config.get_valid_api_keys()
            if not api_keys:
                print("Error: No valid API keys found")
                return

            print(f"Found {len(api_keys)} valid API keys")
            processor = MultiKeyFoundersProcessor(api_keys)
            asyncio.run(processor.process_companies(company_ids))
        except Exception as e:
            print(f"\nError during multi-key founders processing: {str(e)}")

    async def process_founders_async(self, company_ids: List[str]) -> None:
        """Process founders data using async version"""
        print("\nStarting Async Founders Processing...")
        try:
            processor = AsyncFoundersProcessor(Config.API_KEY, Config.BATCH_SIZE)
            await processor.process_companies(company_ids)
        except Exception as e:
            print(f"\nError during async founders processing: {str(e)}")

    def process_person_enrichment(self, company_ids: List[str] = None) -> None:
        """Process person enrichment data"""
        founders_file = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['founders'])
        
        if not os.path.exists(founders_file):
            print("Founders data not found. Skipping person enrichment.")
            return

        print("\nStarting Person Enrichment Processing...")
        try:
            person_ids = FileHandler.extract_person_ids_from_founders(founders_file)
            
            if person_ids:
                processor = PersonEnrichmentProcessor(Config.API_KEY, Config.PERSON_BATCH_SIZE)
                processor.process_person_ids(person_ids)
            else:
                print("No person IDs found to process.")
        except SystemExit as e:
            print(f"\nProcessor stopped by kill switch: {str(e)}")
        except Exception as e:
            print(f"\nError during person enrichment processing: {str(e)}")

    def run_specific_processor(self, processor_key: str, company_ids: List[str], test_mode: bool = False):
        """Run a specific processor with test mode support"""
        if processor_key in self.processors:
            processor = self.processors[processor_key]
            if test_mode:
                print(f"\nRunning {processor['name']} in TEST MODE (first 1000 companies)...")
                test_ids = company_ids[:1000]
                
                if processor_key == '3a':  # Async Founders
                    asyncio.run(processor['function'](test_ids))
                else:
                    processor['function'](test_ids)
            else:
                print(f"\nRunning {processor['name']} in FULL MODE ({len(company_ids)} companies)...")
                if processor_key == '3a':  # Async Founders
                    asyncio.run(processor['function'](company_ids))
                else:
                    processor['function'](company_ids)
        else:
            print(f"Invalid processor key: {processor_key}")

    def run_search_only(self):
        """Run only the search process"""
        if not Config.validate_config(mode='search'):
            print("Configuration validation failed")
            return

        print("\nHow many companies would you like to process?")
        print("Default: 1000, Maximum: 500000")
        try:
            target = input("Enter number of companies (or press Enter for default): ").strip()
            target_companies = int(target) if target else 1000
            if target_companies > 500000:
                print("Maximum limit is 500000 companies. Setting to maximum.")
                target_companies = 500000
        except ValueError:
            print("Invalid input. Using default value of 1000.")
            target_companies = 1000

        processor = SearchProcessor(Config.API_KEY)
        processor.process_companies(target_companies)

    def run_full_process(self):
        """Run the complete data collection process"""
        if not Config.validate_config():
            print("Configuration validation failed")
            return

        FileHandler.ensure_output_directory(Config.OUTPUT_DIR)

        # Get the company data file path
        companies_file = os.path.join(Config.OUTPUT_DIR, Config.COMPANIES_FILE)
        
        print(f"\nReading company data from: {companies_file}")
        company_ids = FileHandler.read_company_ids_from_json(companies_file)
        
        if not company_ids:
            print("No company IDs found to process. Exiting.")
            return

        print(f"\nFound {len(company_ids)} companies to process")

        try:
            while True:
                self.list_processors()
                print("\nOptions:")
                print("1-4: Run test mode (1000 companies)")
                print("3a: Run async founders processor")
                print("3m: Run multi-key founders processor (recommended)")
                print("f1-f4: Run full process (all companies)")
                print("r: Run new search")
                print("b: Go back to main menu")
                
                choice = input("\nEnter your choice: ").lower()
                
                if choice == 'b':
                    break
                elif choice == 'r':
                    if self.process_search():
                        Config.validate_config()
                        companies_file = os.path.join(Config.OUTPUT_DIR, Config.COMPANIES_FILE)
                        company_ids = FileHandler.read_company_ids_from_json(companies_file)
                        print(f"\nUpdated: Found {len(company_ids)} companies to process")
                    continue
                elif choice in ['1', '2', '3', '3a', '3m', '4']:
                    # Test mode (1000 companies)
                    self.run_specific_processor(choice, company_ids, test_mode=True)
                    print("\nTest mode completed successfully!")
                elif choice.startswith('f') and choice[1:] in ['1', '2', '3', '3a', '3m', '4']:
                    # Full mode (all companies)
                    processor_key = choice[1:]
                    self.run_specific_processor(processor_key, company_ids, test_mode=False)
                    print("\nFull process completed successfully!")
                else:
                    print("Invalid selection. Please try again.")
                
                continue_choice = input("\nWould you like to run another processor? (y/n): ").lower()
                if continue_choice != 'y':
                    break
                
        except Exception as e:
            print(f"\nAn error occurred during processing: {str(e)}")
        finally:
            print("\nProcessing finished.")


def main():
    manager = ProcessorManager()
    
    while True:
        print("\nData Processing Options:")
        print("------------------------")
        print("search: Run company search only")
        print("full: Access all processors")
        print("exit: Quit the program")
        print("------------------------")
        
        run_mode = input("Enter your choice: ").lower()
        
        if run_mode == 'exit':
            print("\nExiting program. Goodbye!")
            break
        elif run_mode == 'search':
            manager.run_search_only()
        elif run_mode == 'full':
            manager.run_full_process()
        else:
            print("Invalid option. Please enter 'search', 'full', or 'exit'")


if __name__ == "__main__":
    main()