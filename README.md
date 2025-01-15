# TechInAsia AI News Scraper

A robust web scraper built with Python to collect artificial intelligence news articles from TechInAsia. The scraper uses Selenium WebDriver for dynamic content loading and BeautifulSoup for HTML parsing.

## Features

- Scrapes AI-related news articles from TechInAsia
- Extracts comprehensive article metadata including titles, sources, timestamps, categories, and tags
- Handles dynamic content loading through automated scrolling
- Implements retry mechanisms and error handling
- Saves data in batches as CSV files
- Provides detailed logging
- Configurable scraping parameters

## Requirements

- Python 3.6+
- Chrome browser installed
- Required Python packages (install via pip):
  ```
  selenium
  beautifulsoup4
  pandas
  webdriver_manager
  undetected-chromedriver
  tqdm
  python-dateutil
  ```

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install selenium beautifulsoup4 pandas webdriver_manager undetected-chromedriver tqdm python-dateutil
   ```

## Usage

Run the scraper with default settings:
```bash
python techinasia_scraper_v1_5.py
```

### Command Line Arguments

- `--num_articles`: Number of articles to scrape (default: 100)
- `--max_scrolls`: Maximum number of page scrolls (default: 15)
- `--output_dir`: Output directory for CSV files (default: 'techinasia_output')
- `--min_delay`: Minimum delay between scrolls (default: 1 second)
- `--max_delay`: Maximum delay between scrolls (default: 3 seconds)

Example with custom parameters:
```bash
python techinasia_scraper_v1_5.py --num_articles 200 --max_scrolls 20 --output_dir custom_output
```

## Output

The scraper generates:
- CSV files containing scraped articles in batches
- Detailed log files in the `logs` directory
- Each article contains:
  - Article ID
  - Title
  - URL
  - Source and source URL
  - Image URL
  - Posted time (both original and ISO format)
  - Categories
  - Tags
  - Scraping timestamp

## Project Structure

```
project-ai-news-aggregator/
├── techinasia_scraper_v1_5.py  # Main scraper script
├── logs/                       # Log files directory
├── techinasia_output/          # Default output directory for CSV files
└── README.md                   # This file
```

## Error Handling

The scraper includes:
- Retry mechanisms for failed requests
- Stale element handling
- Duplicate article detection
- Incomplete article filtering
- Detailed logging of errors and warnings

## Limitations

- Requires Chrome browser installed
- Subject to website's rate limiting and robot policies
- Performance depends on internet connection speed
- May require adjustments if website structure changes

## License

This project is open-source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 