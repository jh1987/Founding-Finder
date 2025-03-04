# Funding Finder Quiz App

A Streamlit application that helps startups discover potential funding opportunities through an interactive quiz. The app matches startups with funding opportunities based on their profile and requirements using data from a comprehensive database.

## Features

- Interactive quiz interface for gathering startup information
- Multi-step form with targeted questions
- Smart matching algorithm considering multiple factors:
  - Industry alignment
  - Funding type preferences
  - Amount requirements
  - Geographic location
  - Startup stage
- Detailed funding opportunity profiles
- Match percentage scoring
- Expandable results with comprehensive information

## Data Structure

The app uses a CSV file (`data.csv`) containing funding opportunities with the following information:
- Funding Program Name
- Funding Type (Grant, Loan, Investment, etc.)
- Funding Amount/Range
- Eligibility Criteria
- Application Process
- Application Deadline
- Industry Focus
- Funding Source
- Geographical Restrictions
- Website/Link to Apply
- Contact Information
- Required Documents
- Duration of Funding/Support
- Additional Benefits

## Setup

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

To run the Funding Finder Quiz app, use the following command:
```bash
streamlit run app.py
```

The app will open in your default web browser. If it doesn't open automatically, you can access it at http://localhost:8501

## Usage

1. Start the quiz and provide your startup's basic information:
   - Startup Name
   - Industry
   - Current Stage

2. Specify your funding requirements:
   - Type of funding needed
   - Amount required
   - Location

3. Review your matches:
   - See your startup profile
   - Browse matching opportunities sorted by match percentage
   - Expand each opportunity to view detailed information
   - Use the "Start Over" button to begin a new search

## Adding New Funding Opportunities

To add new funding opportunities, simply add new rows to the `data.csv` file following the existing format. The app will automatically include new opportunities in its matching algorithm. 