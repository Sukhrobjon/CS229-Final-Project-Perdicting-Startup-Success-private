import time
from typing import List, Dict, Callable
import os

from processors.search_processor import SearchProcessor
from processors.company_processor import CompanyEnrichmentProcessor
from processors.funding_processor import FundingRoundsProcessor
from processors.founders_processor import FoundersProcessor
from processors.person_processor import PersonEnrichmentProcessor
from utils.file_handlers import FileHandler
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
                'description': 'Process founders data'
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
            results = processor.search_companies()
            processor.save_results(results)
            return True
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return False

    def process_enrichment(self, company_ids: List[str]) -> None:
        """Process company enrichment data"""
        print("\nStarting Company Enrichment Processing...")
        processor = CompanyEnrichmentProcessor(Config.API_KEY, Config.BATCH_SIZE)
        processor.process_companies(company_ids)

    def process_funding(self, company_ids: List[str]) -> None:
        """Process funding rounds data"""
        print("\nStarting Funding Rounds Processing...")
        processor = FundingRoundsProcessor(Config.API_KEY, Config.BATCH_SIZE)
        processor.process_companies(company_ids)

    def process_founders(self, company_ids: List[str]) -> None:
        """Process founders data"""
        print("\nStarting Founders Processing...")
        processor = FoundersProcessor(Config.API_KEY, Config.BATCH_SIZE)
        processor.process_companies(company_ids)

    def process_person_enrichment(self, company_ids: List[str] = None) -> None:
        """Process person enrichment data"""
        founders_file = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['founders'])
        
        if not os.path.exists(founders_file):
            print("Founders data not found. Skipping person enrichment.")
            return

        print("\nStarting Person Enrichment Processing...")
        person_ids = FileHandler.extract_person_ids_from_founders(founders_file)
        
        if person_ids:
            processor = PersonEnrichmentProcessor(Config.API_KEY, Config.PERSON_BATCH_SIZE)
            processor.process_person_ids(person_ids)
        else:
            print("No person IDs found to process.")

    def run_specific_processor(self, processor_key: str, company_ids: List[str]):
        """Run a specific processor"""
        if processor_key in self.processors:
            processor = self.processors[processor_key]
            print(f"\nRunning {processor['name']}...")
            processor['function'](company_ids)
        else:
            print(f"Invalid processor key: {processor_key}")

    def run_search_only(self):
        """Run only the search process"""
        if not Config.validate_config(mode='search'):
            print("Configuration validation failed")
            return

        self.process_search()

    def run_full_process(self):
        """Run the complete data collection process"""
        if not Config.validate_config():
            print("Configuration validation failed")
            return

        FileHandler.ensure_output_directory(Config.OUTPUT_DIR)

        # Check if we already have company data
        if not os.path.exists(Config.COMPANIES_FILE):
            print("\nNo existing company data found. Running search first...")
            if not self.process_search():
                print("Failed to get company IDs. Exiting.")
                return
        else:
            print(f"\nUsing existing company data from {Config.COMPANIES_FILE}")

        company_ids = FileHandler.read_company_ids_from_json(Config.COMPANIES_FILE)
        
        if not company_ids:
            print("No company IDs found to process. Exiting.")
            return

        print(f"\nFound {len(company_ids)} companies to process")

        try:
            while True:
                self.list_processors()
                print("\nOptions:")
                print("1-4: Run specific processor")
                print("r: Run new search")
                print("b: Go back to main menu")
                
                choice = input("\nEnter your choice: ").lower()
                
                if choice == 'b':
                    break
                elif choice == 'r':
                    if self.process_search():
                        company_ids = FileHandler.read_company_ids_from_json(Config.COMPANIES_FILE)
                        print(f"\nUpdated: Found {len(company_ids)} companies to process")
                    continue
                elif choice in self.processors:
                    self.run_specific_processor(choice, company_ids)
                    print("\nProcessor completed successfully!")
                    
                    continue_choice = input("\nWould you like to run another processor? (y/n): ").lower()
                    if continue_choice != 'y':
                        break
                else:
                    print("Invalid selection. Please try again.")
                
        except Exception as e:
            print(f"\nAn error occurred during processing: {str(e)}")
        finally:
            print("\nProcessing finished.")

def main():
    manager = ProcessorManager()
    
    while True:
        run_mode = input("\nEnter 'search' for search only, 'full' for processor selection, or 'exit' to quit: ").lower()
        
        if run_mode == 'exit':
            break
        elif run_mode == 'search':
            manager.run_search_only()
        elif run_mode == 'full':
            manager.run_full_process()
        else:
            print("Invalid option. Please enter 'search', 'full', or 'exit'")

if __name__ == "__main__":
    main()