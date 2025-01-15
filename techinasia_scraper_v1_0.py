# techinasia_scraper_v1.py

"""
File: techinasia_scraper_v1.py
Overview: This script scrapes articles from the TechInAsia website, specifically from the artificial intelligence category.
It uses Selenium WebDriver for navigating the website and BeautifulSoup for parsing the HTML content. The scraped data is
stored in a pandas DataFrame and saved in batches as CSV files.
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from random import uniform
import os

# Set up logging
log_filename = f"logs/logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class Article:
    """Data class for storing article information"""
    article_id: str
    title: str
    article_url: str
    source: str
    source_url: str
    image_url: Optional[str]
    posted_time: str
    relative_time: str
    categories: List[str]
    tags: List[str]
    scraped_at: str

    def to_dict(self) -> Dict:
        """Convert the Article instance to a dictionary"""
        return {
            'article_id': self.article_id,
            'title': self.title,
            'article_url': self.article_url,
            'source': self.source,
            'source_url': self.source_url,
            'image_url': self.image_url,
            'posted_time': self.posted_time,
            'posted_time_iso': self.get_iso_time(),
            'relative_time': self.relative_time,
            'categories': ','.join(self.categories),
            'tags': ','.join(self.tags),
            'scraped_at': self.scraped_at
        }

    def get_iso_time(self) -> Optional[str]:
        """Convert posted_time to ISO format"""
        try:
            dt = datetime.strptime(self.posted_time, '%I:%M %p at %b %d, %Y')
            return dt.isoformat()
        except ValueError:
            return None

class ScraperConfig:
    """Configuration management for the scraper"""
    DEFAULT_CONFIG = {
        'num_articles': 50,
        'max_scrolls': 10,
        'timeout': 20,
        'retry_count': 3,
        'scroll_pause_time': 1.5,
        'batch_size': 100,
        'base_url': 'https://www.techinasia.com/news',
        'output_dir': 'output'
    }

    def __init__(self, **kwargs):
        """Initialize the configuration with default and custom values"""
        self.config = {**self.DEFAULT_CONFIG, **kwargs}
        self.__dict__.update(self.config)

class TechInAsiaScraper:
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the scraper with configuration"""
        self.config = ScraperConfig(**(config or {}))
        self.driver = None
        self._setup_output_directory()

    def _setup_output_directory(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

    def setup_driver(self):
        """Initialize Selenium WebDriver with retry mechanism"""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

        try:
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            logger.info("WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise

    def scroll_page(self) -> bool:
        """Scroll the page and return True if new content was loaded"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(uniform(1.0, self.config.scroll_pause_time))
        new_height = self.driver.execute_script("return document.body.scrollHeight")
        return new_height > last_height

    def parse_article(self, article_element) -> Optional[Article]:
        """Parse a single article element"""
        try:
            content_div = article_element.find('div', class_='post-content')

            # Extract article information
            title_element = content_div.find('h3', class_='post-title')
            title = title_element.text.strip() if title_element else 'N/A'

            # Get article URL and ID
            article_links = [a for a in content_div.find_all('a') if not 'post-source' in a.get('class', [])]
            article_url = None
            article_id = None
            if article_links:
                href = article_links[0]['href']
                article_url = f"https://www.techinasia.com{href}" if not href.startswith('http') else href
                article_id = href.split('/')[-1] if href else None

            # Get source information
            source_element = content_div.find('span', class_='post-source-name')
            source = source_element.text.strip() if source_element else 'N/A'

            source_link = content_div.find('a', class_='post-source')
            source_url = source_link['href'] if source_link else None

            # Get image information
            image_div = article_element.find('div', class_='post-image')
            image_url = None
            if image_div:
                img_tag = image_div.find('img')
                if img_tag:
                    image_url = img_tag.get('src')

            # Get time and categories/tags
            footer = article_element.find('div', class_='post-footer')
            time_element = footer.find('time') if footer else None
            posted_time = time_element['datetime'] if time_element else 'N/A'
            relative_time = time_element.text.strip() if time_element else 'N/A'

            # Parse categories and tags
            categories = []
            tags = []
            if footer:
                tag_elements = footer.find_all('a', class_='post-taxonomy-link')
                for tag in tag_elements:
                    tag_text = tag.text.strip('· ')
                    if tag_text:
                        if tag.get('href', '').startswith('/category/'):
                            categories.append(tag_text)
                        elif tag.get('href', '').startswith('/tag/'):
                            tags.append(tag_text)

            return Article(
                article_id=article_id,
                title=title,
                article_url=article_url,
                source=source,
                source_url=source_url,
                image_url=image_url,
                posted_time=posted_time,
                relative_time=relative_time,
                categories=categories,
                tags=tags,
                scraped_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )

        except Exception as e:
            logger.error(f"Error parsing article: {e}")
            return None

    def scrape(self) -> pd.DataFrame:
        """Main scraping method"""
        articles = []
        try:
            self.setup_driver()
            articles = self._scrape_with_retry()
            return self._process_articles(articles)
        finally:
            if self.driver:
                self.driver.quit()

    def _scrape_with_retry(self) -> List[Article]:
        """Implement retry logic for scraping"""
        for attempt in range(self.config.retry_count):
            try:
                return self._perform_scraping()
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.config.retry_count - 1:
                    raise
                time.sleep(2 * (attempt + 1))  # Exponential backoff

    def _perform_scraping(self) -> List[Article]:
        """Perform the actual scraping"""
        articles = []
        url = f"{self.config.base_url}?category=artificial-intelligence"

        self.driver.get(url)
        wait = WebDriverWait(self.driver, self.config.timeout)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'post-card')))

        scroll_count = 0
        while len(articles) < self.config.num_articles and scroll_count < self.config.max_scrolls:
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            article_elements = soup.find_all('article', class_='post-card')

            for article_element in article_elements:
                if len(articles) >= self.config.num_articles:
                    break

                article = self.parse_article(article_element)
                if article:
                    articles.append(article)

            if not self.scroll_page():
                break
            scroll_count += 1
            logger.info(f"Scrolled {scroll_count} times, found {len(articles)} articles")

        return articles

    def _process_articles(self, articles: List[Article]) -> pd.DataFrame:
        """Process and clean scraped articles"""
        if not articles:
            return pd.DataFrame()

        # Convert articles to DataFrame
        df = pd.DataFrame([article.to_dict() for article in articles])

        # Remove duplicates
        df = df.drop_duplicates(subset=['article_url'], keep='first')

        # Save in batches
        self._save_batches(df)

        return df

    def _save_batches(self, df: pd.DataFrame):
        """Save DataFrame in batches"""
        batch_size = self.config.batch_size
        for i in range(0, len(df), batch_size):
            batch = df[i:i + batch_size]
            filename = f"{self.config.output_dir}/techinasia_ai_news_v1_batch_{i//batch_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            batch.to_csv(filename, index=False)
            logger.info(f"Saved batch {i//batch_size + 1} to {filename}")

def main():
    """Main function to run the scraper"""
    config = {
        'num_articles': 100,
        'max_scrolls': 15,
        'output_dir': 'techinasia_output'
    }

    logger.info("Starting TechInAsia scraper v1")
    scraper = TechInAsiaScraper(config)

    try:
        df = scraper.scrape()
        logger.info(f"Successfully scraped {len(df)} articles")

        # Display sample of results
        if not df.empty:
            print("\nSample of scraped articles:")
            display_columns = ['title', 'source', 'posted_time_iso', 'categories', 'tags']
            print(df[display_columns].head())

    except Exception as e:
        logger.error(f"Scraping failed: {e}")

# instruction to run the main function: python techinasia_scraper_v1.py
# run with parameter: python techinasia_scraper_v1.py --num_articles 100 --max_scrolls 15 --output_dir techinasia_output
if __name__ == "__main__":
    main()