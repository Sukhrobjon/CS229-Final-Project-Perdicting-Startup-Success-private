import json
import pandas as pd
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
from utils.config import Config

class DataBuilder:
    def __init__(self):
        self.output_dir = Config.OUTPUT_DIR
        self.setup_logging()
        self.error_count = 0
        self.data_quality_metrics = {
            'companies_processed': 0,
            'companies_with_errors': 0,
            'missing_founder_data': 0,
            'missing_funding_data': 0,
            'missing_enrichment_data': 0
        }

    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime('%Y_%m_%d')
        log_file = os.path.join(self.output_dir, f'error_report_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_json_file(self, filepath: str, required: bool = True) -> Tuple[Dict, bool]:
        """Load JSON file and return data with success flag"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data, True
        except Exception as e:
            self.error_count += 1
            error_msg = f"Error loading {filepath}: {str(e)}"
            self.logger.error(error_msg)
            if required:
                raise Exception(error_msg)
            return {}, False

    def validate_person_data(self, person_data: Dict) -> Dict:
        """Validate and clean person enrichment data"""
        if not person_data:
            return {}
            
        person = person_data.get('person', {})
        if not person:
            return {}

        # Filter degree list to keep only fieldOfStudy and name
        filtered_degrees = []
        for degree in person.get('degreeList', []):
            filtered_degree = {
                'fieldOfStudy': degree.get('fieldOfStudy'),
                'name': degree.get('name')
            }
            filtered_degrees.append(filtered_degree)

        # Filter education list to keep only specific fields
        filtered_education = []
        for education in person.get('educationList', []):
            filtered_edu = {
                'endDate': education.get('endDate'),
                'startDate': education.get('startDate'),
                'name': education.get('name'),
                'subject': education.get('subject')
            }
            filtered_education.append(filtered_edu)

        return {
            'gender': person.get('gender'),
            'experienceList': person.get('experienceList', []),
            'degreeList': filtered_degrees,
            'educationList': filtered_education
        }

    def process_founder_details(self, person_data: Dict) -> str:
        """Process and validate founder details"""
        try:
            validated_data = self.validate_person_data(person_data)
            return json.dumps(validated_data)
        except Exception as e:
            self.logger.warning(f"Error processing founder details: {str(e)}")
            return json.dumps({})

    def process_funding_rounds(self, funding_data: Dict, company_id: str) -> Dict:
        """Process funding rounds data for a company"""
        try:
            rounds = funding_data.get(company_id, {}).get('fundingRounds', [])
            
            funding_metrics = {
                'total_rounds': 0,
                'total_raised': 0,
                'avg_round_size': 0,
                'has_vc_investors': False,
                'has_individual_investors': False,
                'has_lead_investor': False,
                'total_investors': 0,
                'latest_valuation': None,
                'seed_amount': float('nan'),
                'series_a_amount': float('nan'),
                'series_b_amount': float('nan'),
                'series_c_amount': float('nan'),
                'series_d_amount': float('nan'),
                'series_e_amount': float('nan'),
                'series_f_amount': float('nan'),
                'series_g_amount': float('nan'),
                'angel_amount': float('nan'),
                'pre_seed_amount': float('nan'),
                'private_equity_amount': float('nan')
            }
            
            if not rounds:
                self.data_quality_metrics['missing_funding_data'] += 1
                return funding_metrics

            stage_to_column = {
                'Seed': 'seed_amount',
                'Series A': 'series_a_amount',
                'Series B': 'series_b_amount',
                'Series C': 'series_c_amount',
                'Series D': 'series_d_amount',
                'Series E': 'series_e_amount',
                'Series F': 'series_f_amount',
                'Series G': 'series_g_amount',
                'Angel': 'angel_amount',
                'Pre-Seed': 'pre_seed_amount',
                'Private Equity': 'private_equity_amount'
            }

            total_raised = 0
            for round in rounds:
                try:
                    money_raised = round.get('moneyRaised', 0)
                    stage = round.get('stage', '')
                    
                    if stage in stage_to_column:
                        column_name = stage_to_column[stage]
                        current_amount = funding_metrics[column_name]
                        funding_metrics[column_name] = money_raised if pd.isna(current_amount) else current_amount + money_raised
                        total_raised += money_raised

                    person_investors = len(round.get('personInvestorList', []))
                    company_investors = len(round.get('companyInvestorList', []))
                    funding_metrics['total_investors'] += person_investors + company_investors
                    
                    if person_investors > 0:
                        funding_metrics['has_individual_investors'] = True
                    if company_investors > 0:
                        funding_metrics['has_vc_investors'] = True
                    if round.get('leadCompanyInvestorList'):
                        funding_metrics['has_lead_investor'] = True
                except Exception as e:
                    self.logger.warning(f"Error processing round for {company_id}: {str(e)}")
                    continue

            funding_metrics['total_rounds'] = len(rounds)
            funding_metrics['total_raised'] = total_raised
            funding_metrics['avg_round_size'] = total_raised / len(rounds) if rounds else 0
            
            latest_round = max(rounds, key=lambda x: x.get('announcedOn', ''), default={})
            funding_metrics['latest_valuation'] = latest_round.get('valuation', {}).get('exact')

            return funding_metrics
            
        except Exception as e:
            self.logger.error(f"Error processing funding rounds for {company_id}: {str(e)}")
            self.data_quality_metrics['companies_with_errors'] += 1
            return funding_metrics

    def process_founders(self, founders_data: Dict, person_enrichment_data: Dict, company_id: str) -> Dict:
        """Process founders data for a company"""
        try:
            founders = founders_data.get(company_id, {}).get('founders', [])
            
            founder_metrics = {
                'number_of_founders': len(founders)
            }
            
            if not founders:
                self.data_quality_metrics['missing_founder_data'] += 1

            # Process up to 5 founders
            for i in range(5):
                founder_column = f'founder_{i+1}'
                if i < len(founders):
                    founder = founders[i]
                    founder_id = founder.get('id')
                    
                    person_data = person_enrichment_data.get(founder_id, {})
                    if person_data:
                        founder_metrics[founder_column] = self.process_founder_details(person_data)
                    else:
                        founder_metrics[founder_column] = None
                        self.logger.warning(f"Missing person enrichment data for founder {founder_id}")
                else:
                    founder_metrics[founder_column] = None
            
            return founder_metrics
            
        except Exception as e:
            self.logger.error(f"Error processing founders for {company_id}: {str(e)}")
            self.data_quality_metrics['companies_with_errors'] += 1
            return {'number_of_founders': 0, **{f'founder_{i+1}': None for i in range(5)}}

    def save_data_quality_report(self):
        """Save data quality metrics"""
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
        report_file = os.path.join(self.output_dir, f'data_quality_report_{timestamp}.json')
        
        with open(report_file, 'w') as f:
            json.dump(self.data_quality_metrics, f, indent=2)
        
        self.logger.info("Data quality report saved to: " + report_file)

    def build_initial_dataset(self):
        """Build initial dataset from all data sources"""
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
        self.logger.info("Starting initial dataset build...")

        try:
            # Load all data sources
            enrichment_data, _ = self.load_json_file(
                os.path.join(self.output_dir, Config.OUTPUT_FILES['company_enrichment']),
                required=True
            )
            funding_data, _ = self.load_json_file(
                os.path.join(self.output_dir, Config.OUTPUT_FILES['funding_rounds']),
                required=False
            )
            founders_data, _ = self.load_json_file(
                os.path.join(self.output_dir, Config.OUTPUT_FILES['founders']),
                required=False
            )
            person_data, _ = self.load_json_file(
                os.path.join(self.output_dir, Config.OUTPUT_FILES['person_enrichment']),
                required=False
            )

            # Process records
            records = []
            for company_id, data in enrichment_data.items():
                try:
                    self.data_quality_metrics['companies_processed'] += 1
                    
                    # Process base company data
                    data_copy = data.copy()
                    
                    # Convert complex types to JSON strings
                    for field in ['URLs', 'industryList', 'alternateNames', 'targetMarketList', 'customerTypes']:
                        if field in data_copy:
                            data_copy[field] = json.dumps(data_copy[field])

                    # Add funding rounds metrics
                    funding_metrics = self.process_funding_rounds(funding_data, company_id)
                    data_copy.update(funding_metrics)
                    
                    # Add founders metrics
                    founders_metrics = self.process_founders(founders_data, person_data, company_id)
                    data_copy.update(founders_metrics)
                    
                    records.append(data_copy)
                    
                except Exception as e:
                    self.logger.error(f"Error processing company {company_id}: {str(e)}")
                    self.data_quality_metrics['companies_with_errors'] += 1
                    continue

            # Create DataFrame
            df = pd.DataFrame(records)

            # Determine output filename based on errors
            filename_prefix = 'consolidated_data_with_errors' if self.error_count > 0 else 'consolidated_data_clean'
            output_file = os.path.join(self.output_dir, f'{filename_prefix}_{timestamp}.csv')
            
            # Save to CSV
            df.to_csv(output_file, index=False)

            # Save data quality report
            self.save_data_quality_report()

            # Final summary
            self.logger.info(f"\nDataset build completed:")
            self.logger.info(f"Total companies processed: {len(df)}")
            self.logger.info(f"Total columns: {len(df.columns)}")
            self.logger.info(f"Total errors encountered: {self.error_count}")
            self.logger.info(f"Output saved to: {output_file}")
            
            # Display column overview
            self.logger.info("\nColumn Overview:")
            self.logger.info(df.dtypes)

        except Exception as e:
            self.logger.error(f"Critical error in dataset build: {str(e)}")
            raise

def main():
    builder = DataBuilder()
    builder.build_initial_dataset()

if __name__ == "__main__":
    main()