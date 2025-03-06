import requests
import json
import os
from typing import Dict, Optional, List
import time
from datetime import datetime
from processors.base_processor import BaseProcessor
from utils.config import Config

class CompanyEnrichmentProcessor(BaseProcessor):
    def __init__(self, api_key: str, batch_size: int = 1000):
        super().__init__(api_key, batch_size)
        self.progress_interval = 500
        self.api_calls_made = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.progress_file = os.path.join(Config.OUTPUT_DIR, "enrichment_progress.json")
        self.start_time = time.time()
        self.had_errors = False

    def filter_company_data(self, data: Dict) -> Dict:
        """Filter company data to keep only desired fields"""
        desired_fields = {
            "id", "name", "country", "region", "locality", "URLs", 
            "industryList", "alternateNames", "description", "tagline",
            "founded", "headcount", "financingStatus", "ownershipStatus",
            "status", "latestDealType", "latestDealDate", "latestDealAmount",
            "investorCount", "legalName", "leadInvestorCount", "totalFunding",
            "fundingRoundCount", "lastRoundValuation", "targetMarketList",
            "isAcquired", "isGovernment", "isNonProfit", "isExited",
            "isShutDown", "customerTypes"
        }
        return {k: v for k, v in data.items() if k in desired_fields}

    def get_company_enrichment(self, company_id: str) -> Optional[Dict]:
        """Get enriched company data"""
        url = "https://data.api.aviato.co/company/enrich"
        params = {"id": company_id}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            self.api_calls_made += 1
            
            if response.status_code == 200:
                self.successful_calls += 1
                data = response.json()
                return self.filter_company_data(data)
            else:
                self.failed_calls += 1
                self.had_errors = True
                print(f"Error fetching enrichment for {company_id}: {response.status_code}")
                return None
        except Exception as e:
            self.failed_calls += 1
            self.had_errors = True
            print(f"Exception fetching enrichment for {company_id}: {str(e)}")
            return None

    def save_progress(self, company_ids: List[str], current_index: int):
        """Save current progress"""
        progress = {
            "total_companies": len(company_ids),
            "companies_processed": current_index,
            "api_calls_made": self.api_calls_made,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "last_processed_id": company_ids[current_index-1] if current_index > 0 else None,
            "last_updated": datetime.now().isoformat(),
            "had_errors": self.had_errors
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def load_progress(self) -> Dict:
        """Load previous progress"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.api_calls_made = progress.get('api_calls_made', 0)
                    self.successful_calls = progress.get('successful_calls', 0)
                    self.failed_calls = progress.get('failed_calls', 0)
                    self.had_errors = progress.get('had_errors', False)
                    return progress
            except Exception as e:
                print(f"Error loading progress: {e}")
        return {}

    def save_batch(self, is_final: bool = False):
        """Save current batch of enrichment data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        status = "with_errors" if self.had_errors else "clean"
        stage = "final" if is_final else "partial"
        
        filename = os.path.join(
            Config.OUTPUT_DIR, 
            f"company_enrichment_{stage}_{status}_{timestamp}.json"
        )
        
        self.append_to_json(filename)
        return filename

    def process_companies(self, company_ids: List[str]):
        """Process list of companies for enrichment data"""
        total_companies = len(company_ids)
        print(f"\nStarting Company Enrichment processing for {total_companies} companies")
        print(f"Expected API credits needed: {total_companies}")
        
        # Load any previous progress
        progress = self.load_progress()
        start_index = progress.get('companies_processed', 0)
        
        if start_index > 0:
            print(f"Resuming from company {start_index + 1}")

        last_save_file = None
        try:
            for i, company_id in enumerate(company_ids[start_index:], start_index + 1):
                try:
                    enrichment_data = self.get_company_enrichment(company_id)
                    if enrichment_data:
                        self.data[company_id] = enrichment_data
                    
                    # Show progress every 500 companies
                    if i % self.progress_interval == 0:
                        elapsed_time = time.time() - self.start_time
                        avg_time_per_company = elapsed_time / i
                        remaining_companies = total_companies - i
                        estimated_remaining_time = remaining_companies * avg_time_per_company
                        
                        print(f"\nProgress: {i}/{total_companies} ({(i/total_companies)*100:.1f}%)")
                        print(f"API Calls: {self.api_calls_made} (Success: {self.successful_calls}, Failed: {self.failed_calls})")
                        print(f"Est. time remaining: {estimated_remaining_time/60:.1f} minutes")
                        
                        # Save progress and data
                        self.save_progress(company_ids, i)
                        last_save_file = self.save_batch()
                    
                    time.sleep(1)  # Rate limiting
                    
                except KeyboardInterrupt:
                    print("\nProcess interrupted by user. Saving progress...")
                    self.save_progress(company_ids, i)
                    last_save_file = self.save_batch()
                    raise
                except Exception as e:
                    self.had_errors = True
                    print(f"Error processing company {company_id}: {str(e)}")
                    continue

            # Final save
            if self.data:
                last_save_file = self.save_batch(is_final=True)
                print(f"\nFinal data saved to: {last_save_file}")

        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
        except Exception as e:
            self.had_errors = True
            print(f"\nError during processing: {str(e)}")
        finally:
            # Final summary
            elapsed_time = time.time() - self.start_time
            print(f"\nCompany Enrichment Processing Complete:")
            print("------------------------")
            print(f"Companies Processed: {self.successful_calls}/{total_companies}")
            print(f"API Calls: {self.api_calls_made} (Success: {self.successful_calls}, Failed: {self.failed_calls})")
            print(f"Total time: {elapsed_time/60:.1f} minutes")
            print(f"Status: {'Completed with errors' if self.had_errors else 'Clean'}")
            if last_save_file:
                print(f"Final data file: {last_save_file}")
            print("------------------------")