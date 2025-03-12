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

    def validate_output_directories(self):
        """Ensure all required output directories exist"""
        directories = [
            Config.OUTPUT_DIR,
            os.path.join(Config.OUTPUT_DIR, 'logs'),
            os.path.join(Config.OUTPUT_DIR, 'founders'),
            os.path.join(Config.OUTPUT_DIR, 'person')
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
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

    def process_person_enrichment(self, company_ids: List[str] = None, test_mode: bool = False) -> None:
        """Process person enrichment data"""
        # Ensure required directories exist
        self.validate_output_directories()
        
        # Check for founders data in the correct location
        if not os.path.exists(Config.FOUNDERS_DATA_PATH):
            print(f"Founders data not found at: {Config.FOUNDERS_DATA_PATH}")
            print("Please ensure founders data is available at this location before running person enrichment.")
            return

        print("\nStarting Person Enrichment Processing...")
        try:
            processor = PersonEnrichmentProcessor(Config.API_KEY, Config.PERSON_BATCH_SIZE)
            
            # Extract all person IDs first
            person_ids = processor.extract_person_ids_from_founders_data(Config.FOUNDERS_DATA_PATH)
            
            if not person_ids:
                print("No person IDs found to process.")
                return
                
            # Handle test mode
            if test_mode:
                total_ids = len(person_ids)
                test_size = min(500, total_ids)
                print(f"Test mode: Processing first {test_size} person IDs out of {total_ids}")
                person_ids = person_ids[:test_size]
            else:
                print(f"Full mode: Processing all {len(person_ids)} person IDs")
            
            # Process the IDs
            processor.process_person_ids(person_ids)
            
        except FileNotFoundError as e:
            print(f"\nError: {str(e)}")
            print(f"Expected founders data at: {Config.FOUNDERS_DATA_PATH}")
        except Exception as e:
            print(f"\nError during person enrichment processing: {str(e)}")

    def run_specific_processor(self, processor_key: str, company_ids: List[str], test_mode: bool = False):
        """Run a specific processor with test mode support"""
        if processor_key in self.processors:
            processor = self.processors[processor_key]
            
            # Special handling for person enrichment
            if processor_key == '4':
                if test_mode:
                    print(f"\nRunning {processor['name']} in TEST MODE (first 500 person IDs)...")
                else:
                    print(f"\nRunning {processor['name']} in FULL MODE...")
                processor['function'](company_ids, test_mode=test_mode)
                
            # Standard handling for other processors
            else:
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

        # Ensure all required directories exist
        self.validate_output_directories()

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
                print("1-4: Run test mode (500 companies/persons)")
                print("f1-f4: Run full process (all data)")
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
                    # Full mode (all companies/persons)
                    processor_key = choice[1:]
                    self.run_specific_processor(processor_key, company_ids, test_mode=False)
                    print("\nFull process completed successfully!")
                else:
                    print("Invalid selection. Please try again.")
                
                # Ask to continue only if the previous process completed
                try:
                    continue_choice = input("\nWould you like to run another processor? (y/n): ").lower()
                    if continue_choice != 'y':
                        break
                except KeyboardInterrupt:
                    print("\nProcess interrupted by user")
                    break
                except Exception as e:
                    print(f"\nError getting user input: {str(e)}")
                    break
                
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
        except Exception as e:
            print(f"\nAn error occurred during processing: {str(e)}")
        finally:
            print("\nProcessing finished.")


def main():
    manager = ProcessorManager()
    
    while True:
        try:
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
                
        except KeyboardInterrupt:
            print("\nProgram interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again or type 'exit' to quit.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        print("Program terminated.")