# /main.py

import glob
from typing import List, Dict, Callable, Optional
import os
import time
from processors.search_processor import SearchProcessor
from processors.company_processor import CompanyEnrichmentProcessor
from processors.funding_processor import FundingRoundsProcessor
from processors.founders_processor import FoundersProcessor
from processors.person_processor import PersonEnrichmentProcessor
from utils.file_handler import FileHandler
from utils.config import Config

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
                'description': 'Process founders data (Optimized)'
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
        """Process founders data with optimized settings"""
        print("\nStarting Founders Processing...")
        try:
            processor = FoundersProcessor(Config.API_KEY, Config.BATCH_SIZE)
            processor.process_companies(company_ids)
        except SystemExit as e:
            print(f"\nProcessor stopped by kill switch: {str(e)}")
        except Exception as e:
            print(f"\nError during founders processing: {str(e)}")

    def process_person_enrichment(self, company_ids: List[str] = None) -> None:
        """Process person enrichment data"""
        # Check for founders data in the correct location
        if not os.path.exists(Config.FOUNDERS_DATA_PATH):
            print(f"Founders data not found at: {Config.FOUNDERS_DATA_PATH}")
            print("Please ensure founders data is available at this location before running person enrichment.")
            return

        print("\nStarting Person Enrichment Processing...")
        try:
            processor = PersonEnrichmentProcessor(Config.API_KEY, Config.PERSON_BATCH_SIZE)
            # Use the new process_founders_data method
            processor.process_founders_data()
        except FileNotFoundError as e:
            print(f"\nError: {str(e)}")
            print(f"Expected founders data at: {Config.FOUNDERS_DATA_PATH}")
        except Exception as e:
            print(f"\nError during person enrichment processing: {str(e)}")

    def run_specific_processor(self, processor_key: str, company_ids: List[str], test_mode: bool = False):
        """Run a specific processor with test mode support"""
        if processor_key in self.processors:
            processor = self.processors[processor_key]
            if test_mode:
                print(f"\nRunning {processor['name']} in TEST MODE (first 500 companies)...")
                test_ids = company_ids[:500]
                processor['function'](test_ids)
            else:
                print(f"\nRunning {processor['name']} in FULL MODE ({len(company_ids)} companies)...")
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
                print("f1-f4: Run full process (all companies)")
                print("t: Run specific processor in test mode")
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
                elif choice == 't':
                    # Test mode with specific processor
                    print("\nSelect processor for test mode (1-4):")
                    test_choice = input("Enter processor number: ")
                    if test_choice in self.processors:
                        self.run_specific_processor(test_choice, company_ids, test_mode=True)
                        print("\nTest mode completed successfully!")
                    else:
                        print("Invalid processor selection.")
                elif choice in ['1', '2', '3', '4']:
                    # Default to test mode for direct number selection
                    self.run_specific_processor(choice, company_ids, test_mode=True)
                    print("\nTest mode completed successfully!")
                elif choice.startswith('f') and choice[1:] in ['1', '2', '3', '4']:
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
    