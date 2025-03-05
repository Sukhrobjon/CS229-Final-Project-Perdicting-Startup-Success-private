Here's a concise README.md:

```markdown
# Aviato Data Pipeline

A data processing pipeline for collecting and consolidating company and founder information from Aviato API.

## Project Structure
```
aviato-pipeline/
├── processors/          # API processors for different endpoints
├── utils/              # Utility functions and configurations
├── output/             # Generated data files and reports
├── main.py            # Main API collection script
└── build_data_from_json.py  # Data consolidation script
```

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with:
```
AVIATO_API_KEY=your_api_key_here
```

## Usage

### Data Collection
```bash
python main.py
```
Choose:
- `search`: Collect company IDs
- `full`: Run specific data processor (enrichment, funding, founders, person)

### Data Consolidation
```bash
python build_data_from_json.py
```
Generates:
- Consolidated CSV file
- Data quality report
- Error logs

## Output Files
- `consolidated_data_*.csv`: Main dataset for analysis
- `data_quality_report_*.json`: Processing statistics
- `error_report_*.log`: Detailed error logs

## Requirements
- Python 3.8+
- See requirements.txt for dependencies
```
