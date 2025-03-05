import requests
import json
import os
from typing import Dict, Optional, List
import time
from processors.base_processor import BaseProcessor
from utils.config import Config

class CompanyEnrichmentProcessor(BaseProcessor):
    def __init__(self, api_key: str, batch_size: int = 1000):
        super().__init__(api_key, batch_size)
        self.progress_interval = 100  # Show progress every 100 companies

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
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            return self.filter_company_data(data)
        else:
            print(f"Error fetching enrichment for {company_id}: {response.status_code}")
            return None

    def save_batch(self):
        """Save current batch of enrichment data"""
        filename = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['company_enrichment'])
        print(f"\nSaving batch to {filename}...")
        self.append_to_json(filename)

    def process_companies(self, company_ids: List[str]):
        """Process list of companies for enrichment data"""
        total_companies = len(company_ids)
        start_time = time.time()
        successful_processes = 0
        
        print(f"\nStarting Company Enrichment processing for {total_companies} companies...")

        for i, company_id in enumerate(company_ids, 1):
            try:
                enrichment_data = self.get_company_enrichment(company_id)
                if enrichment_data:
                    self.data[company_id] = enrichment_data
                    successful_processes += 1
                    
                    # Show periodic progress for large datasets
                    if i % self.progress_interval == 0:
                        elapsed_time = time.time() - start_time
                        avg_time_per_company = elapsed_time / i
                        remaining_companies = total_companies - i
                        estimated_remaining_time = remaining_companies * avg_time_per_company
                        
                        print(f"\nProgress: {i}/{total_companies} companies processed "
                              f"({(i/total_companies)*100:.1f}%)")
                        print(f"Estimated time remaining: {estimated_remaining_time/60:.1f} minutes")

                    # Don't check batch size for small datasets
                    if total_companies > self.batch_size:
                        self.check_and_save_batch()
                
                if i < total_companies:
                    time.sleep(1)  # Rate limiting
                    
            except Exception as e:
                print(f"Error processing company {company_id}: {str(e)}")
                continue
        
        # Final save if we have data
        if self.data:
            print("\nSaving final batch...")
            self.save_batch()

        # Final summary
        elapsed_time = time.time() - start_time
        print(f"\nCompany Enrichment processing completed:")
        print(f"Total companies processed: {total_companies}")
        print(f"Successful processes: {successful_processes}")
        print(f"Total time taken: {elapsed_time/60:.1f} minutes")
        print(f"Data saved to: {os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_FILES['company_enrichment'])}")