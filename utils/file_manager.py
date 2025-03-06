import os
import json
import glob
from typing import List, Dict, Set, Tuple
from datetime import datetime

class FileManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.batch_size = 10000
        self.company_ids_seen: Set[str] = set()
        self.company_names_seen: Set[str] = set()

    def get_batch_filename(self, start_offset: int, end_offset: int) -> str:
        """Generate filename for batch"""
        return os.path.join(self.output_dir, f"companies_{start_offset}_{end_offset}.json")

    def save_batch(self, companies: List[Dict], start_offset: int) -> Tuple[str, int]:
        """
        Save batch of companies
        Returns: (filename, actual_end_offset)
        """
        end_offset = start_offset + len(companies)
        filename = self.get_batch_filename(start_offset, end_offset)

        # Validate for duplicates
        duplicates = self.check_duplicates(companies)
        if duplicates:
            print(f"\nWarning: Found {len(duplicates)} duplicates in batch:")
            for dup in duplicates[:5]:  # Show first 5 duplicates
                print(f"Duplicate: {dup}")

        data = {
            "metadata": {
                "start_offset": start_offset,
                "end_offset": end_offset,
                "companies_count": len(companies),
                "timestamp": datetime.now().isoformat(),
                "duplicates_found": len(duplicates)
            },
            "companies": companies
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved batch to: {filename}")
        print(f"Companies in batch: {len(companies)}")
        
        return filename, end_offset

    def check_duplicates(self, companies: List[Dict]) -> List[Dict]:
        """Check for duplicates in current batch"""
        duplicates = []
        
        for company in companies:
            company_id = company.get('id')
            company_name = company.get('name')
            
            if company_id in self.company_ids_seen:
                duplicates.append({
                    'id': company_id,
                    'name': company_name,
                    'type': 'id_duplicate'
                })
            if company_name in self.company_names_seen:
                duplicates.append({
                    'id': company_id,
                    'name': company_name,
                    'type': 'name_duplicate'
                })
                
            self.company_ids_seen.add(company_id)
            self.company_names_seen.add(company_name)
            
        return duplicates

    def merge_files(self, target_companies: int) -> str:
        """
        Merge all files into one
        Returns: path to merged file
        """
        print("\nStarting file merge process...")
        
        # Get all batch files
        pattern = os.path.join(self.output_dir, "companies_*.json")
        files = glob.glob(pattern)
        files.sort(key=lambda x: int(x.split('_')[-2]))  # Sort by start offset

        if not files:
            print("No files to merge")
            return ""

        merged_companies = []
        total_companies = 0
        duplicates_found = 0

        # Reset duplicate tracking sets
        self.company_ids_seen.clear()
        self.company_names_seen.clear()

        for file in files:
            print(f"\nProcessing file: {file}")
            with open(file, 'r') as f:
                data = json.load(f)
                companies = data['companies']
                
                # Check for duplicates
                duplicates = self.check_duplicates(companies)
                duplicates_found += len(duplicates)
                
                merged_companies.extend(companies)
                total_companies += len(companies)
                
                print(f"Companies processed: {total_companies}")
                if duplicates:
                    print(f"Duplicates found in file: {len(duplicates)}")

        # Save merged file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        merged_filename = os.path.join(
            self.output_dir, 
            f"companies_merged_{total_companies}_{timestamp}.json"
        )

        merged_data = {
            "metadata": {
                "total_companies": total_companies,
                "target_companies": target_companies,
                "files_merged": len(files),
                "duplicates_found": duplicates_found,
                "timestamp": datetime.now().isoformat()
            },
            "companies": merged_companies
        }

        with open(merged_filename, 'w') as f:
            json.dump(merged_data, f, indent=2)

        print(f"\nMerge completed:")
        print(f"Total companies: {total_companies}")
        print(f"Files merged: {len(files)}")
        print(f"Duplicates found: {duplicates_found}")
        print(f"Merged file: {merged_filename}")

        return merged_filename