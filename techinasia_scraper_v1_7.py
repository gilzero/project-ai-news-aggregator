# techinasia_scraper_v1_7.py

"""
File: techinasia_scraper_v1_7.py
Overview: This script scrapes articles from the TechInAsia website, specifically from a configurable category.
It uses Selenium WebDriver for navigating the website and BeautifulSoup for parsing the HTML content. The scraped data is
stored in a pandas DataFrame and saved in batches as CSV files.
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Optional
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
import re
from dateutil import parser
import argparse
import undetected_chromedriver as uc
from tqdm import tqdm
from pydantic import BaseModel, Field, field_validator, ConfigDict, HttpUrl

# Set up logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = os.path.join(log_dir, f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class Article(BaseModel):
    """Data class for storing article information"""
    article_id: Optional[str] = None
    title: str = 'N/A'
    article_url: Optional[str] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    image_url: Optional[str] = None
    posted_time: Optional[str] = None
    relative_time: str = 'N/A'
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    scraped_at: str = Field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    posted_time_iso: Optional[str] = None

    @field_validator('article_url', 'source_url', 'image_url', mode='before')
    def validate_urls(cls, v):
        """Validate and clean URLs"""
        if v == 'N/A':
            return None
        return v

    @field_validator('posted_time', mode='before')
    def convert_posted_time(cls, v):
        """Convert posted_time to ISO format"""
        if v:
            try:
                dt = parser.parse(v)
                return dt.isoformat()
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse posted_time: {v}, error: {e}")
                return None
        return v

    def model_dump_json(self) -> str:
        """Convert the Article instance to a JSON string"""
        return self.json()

class ScraperConfig(BaseModel):
    """Configuration management for the scraper"""
    num_articles: int = 50
    max_scrolls: int = 10
    timeout: int = 20
    retry_count: int = 3
    scroll_pause_time: float = 1.5
    batch_size: int = 100
    base_url: str = 'https://www.techinasia.com/news'
    output_dir: str = 'output'
    user_agent: str = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    min_delay: float = 1
    max_delay: float = 3
    category: str = 'artificial-intelligence'

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True
    )

    @field_validator('num_articles', 'max_scrolls', 'timeout', 'retry_count', 'batch_size')
    def validate_positive_int(cls, v: int) -> int:
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"{cls.__name__} must be a positive integer")
        return v

    @field_validator('scroll_pause_time', 'min_delay', 'max_delay')
    def validate_positive_number(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"{cls.__name__} must be a positive number")
        return v

    @field_validator('max_delay')
    def validate_max_delay(cls, v: float, info) -> float:
        if 'min_delay' in info.data and v < info.data['min_delay']:
            raise ValueError("max_delay must be greater than or equal to min_delay")
        return v

    @field_validator('category')
    def validate_non_empty_string(cls, v: str) -> str:
        if not isinstance(v, str) or not v:
            raise ValueError("category must be a non-empty string")
        return v

class ScrollManager:
    """Manages page scrolling"""
    def __init__(self, driver, config: ScraperConfig):
        self.driver = driver
        self.config = config

    def scroll_page(self) -> bool:
        """Scroll the page and return True if new content was loaded"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(uniform(self.config.min_delay, self.config.max_delay))
        new_height = self.driver.execute_script("return document.body.scrollHeight")
        return new_height > last_height

class TechInAsiaScraper:
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the scraper with configuration"""
        self.config = ScraperConfig(**(config or {}))
        self.driver = None
        self.scroll_manager = None
        self._setup_output_directory()
        self.processed_article_ids = set()
        self.incomplete_articles = 0
        self.total_articles = 0

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
        options.add_argument(f'user-agent={self.config.user_agent}')

        try:
            driver_manager = ChromeDriverManager()
            driver_path = driver_manager.install()
            self.driver = uc.Chrome(
                service=Service(driver_path),
                options=options,
                use_subprocess=True
            )
            self.scroll_manager = ScrollManager(self.driver, self.config)
            logger.info("🚀 WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"😞 Failed to initialize WebDriver: {e}")
            raise

    def _is_valid_article(self, article: Article) -> bool:
        """Validate if the article has all required fields"""
        if not article.article_url:
            logger.warning(f"⚠️ Skipping article with missing article_url: {article.article_id}")
            return False
        return True

    def _clean_article_data(self, article: Article) -> Article:
        """Clean and normalize article data"""
        return article

    def parse_article(self, article_element) -> Optional[Article]:
        """Parse a single article element"""
        article_id = None
        title = None
        article_url = None
        source = None
        source_url = None
        image_url = None
        posted_time = None
        relative_time = None
        categories = []
        tags = []

        try:
            content_div = article_element.find('div', class_='post-content')

            # Extract article information
            title_element = content_div.find('h3', class_='post-title')
            title = title_element.text.strip() if title_element else None

            # Get article URL and ID first
            article_links = [a for a in content_div.find_all('a') if not 'post-source' in a.get('class', [])]
            if article_links:
                href = article_links[0]['href']
                article_url = f"https://www.techinasia.com{href}" if not href.startswith('http') else href
                match = re.search(r'/([^/]+)$', href)
                article_id = match.group(1) if match else None
                logger.info(f"Parsing article: {article_id} - {title}")
            else:
                logger.warning("No article links found.")
                return None

            # Get source information
            logger.info(f"  - 📰 Extracting source...")
            source_element = content_div.find('span', class_='post-source-name')
            source = source_element.text.strip() if source_element else None
            logger.info(f"  - ✅ Source extracted: {source}")

            source_link = content_div.find('a', class_='post-source')
            source_url = source_link.get('href') if source_link else None
            logger.info(f"  - 🌐 Source URL extracted: {source_url}")

            # Get image information
            logger.info(f"  - 🖼️ Extracting image URL...")
            image_div = article_element.find('div', class_='post-image')
            if image_div:
                img_tag = image_div.find('img')
                image_url = img_tag.get('src') if img_tag else None
                logger.info(f"  - 🖼️ Image URL extracted: {image_url}")
            else:
                logger.warning(f"  - 🖼️ No image div found.")

            # Get time and categories/tags
            logger.info(f"  - ⏰ Extracting time and categories/tags...")
            footer = article_element.find('div', class_='post-footer')
            time_element = footer.find('time') if footer else None
            posted_time = time_element.get('datetime') if time_element else None
            relative_time = time_element.text.strip() if time_element else None
            logger.info(f"  - ⏰ Time extracted: {posted_time}, Relative Time: {relative_time}")

            # Parse categories and tags
            if footer:
                tag_elements = footer.find_all('a', class_='post-taxonomy-link')
                for tag in tag_elements:
                    tag_text = tag.text.strip('· ')
                    if tag_text:
                        if tag.get('href', '').startswith('/category/'):
                            categories.append(tag_text)
                        elif tag.get('href', '').startswith('/tag/'):
                            tags.append(tag_text)
                logger.info(f"  - 🏷️ Categories: {categories}, Tags: {tags}")

            article = Article(
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
            )
            logger.info(f"🎉 Article parsing complete: {article_id}")
            return article

        except AttributeError as e:
            logger.warning(f"⚠️ AttributeError parsing article: {article_id}, error: {e}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Error parsing article: {article_id}, error: {e}")
            return None

    def scrape(self) -> pd.DataFrame:
        """Main scraping method"""
        articles = []
        start_time = datetime.now()
        try:
            self.setup_driver()
            articles = self._scrape_with_retry()
            df = self._process_articles(articles)
            return df
        finally:
            if self.driver:
                self.driver.quit()
            end_time = datetime.now()
            self._log_summary(start_time, end_time)

    def _scrape_with_retry(self) -> List[Article]:
        """Implement retry logic for scraping"""
        for attempt in range(self.config.retry_count):
            try:
                return self._perform_scraping()
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                if attempt == self.config.retry_count - 1:
                    raise
                time.sleep(2 * (attempt + 1))  # Exponential backoff

    def _perform_scraping(self) -> List[Article]:
        """Perform the actual scraping"""
        articles = []
        url = f"{self.config.base_url}?category={self.config.category}"

        logger.info(f"Navigating to URL: {url}")
        self.driver.get(url)
        wait = WebDriverWait(self.driver, self.config.timeout)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'post-card')))
        logger.info(f"Page loaded successfully.")

        progress_bar = tqdm(total=self.config.num_articles, desc="Scraping Articles", unit="article")
        scroll_count = 0
        last_article_count = 0
        while len(articles) < self.config.num_articles and scroll_count < self.config.max_scrolls:
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            article_elements = soup.find_all('article', class_='post-card')

            for article_element in article_elements:
                try:
                    article = self.parse_article(article_element)
                    if article and article.article_id not in self.processed_article_ids:
                        if self._is_valid_article(article):
                            cleaned_article = self._clean_article_data(article)
                            articles.append(cleaned_article)
                            self.processed_article_ids.add(article.article_id)
                            logger.info(f"Processed article {article.article_id} - {article.title}")
                            progress_bar.update(1)
                            progress_bar.set_postfix({"incomplete": self.incomplete_articles})
                        else:
                            self.incomplete_articles += 1
                        self.total_articles += 1
                        if len(articles) >= self.config.num_articles:
                            break
                except StaleElementReferenceException as e:
                    logger.warning(f"StaleElementReferenceException: {e}, retrying parsing")
                    continue
                except Exception as e:
                    logger.error(f"Error during article processing: {e}")
                    continue

            if not self.scroll_manager.scroll_page() or len(articles) == last_article_count:
                break
            last_article_count = len(articles)
            scroll_count += 1
            logger.info(f"Scrolled {scroll_count} times, found {len(articles)} articles")

            # Implementing rate limiting
            time.sleep(uniform(self.config.min_delay, self.config.max_delay))

        progress_bar.close()
        return articles

    def _process_articles(self, articles: List[Article]) -> pd.DataFrame:
        """Process and clean scraped articles"""
        if not articles:
            return pd.DataFrame()

        # Convert articles to DataFrame
        df = pd.DataFrame([article.dict() for article in articles])

        # Remove duplicates
        df = df.drop_duplicates(subset=['article_url'], keep='first')

        # Save in batches
        self._save_batches(df)

        return df

    def _save_batches(self, df: pd.DataFrame):
        """Save DataFrame in batches"""
        batch_size = self.config.batch_size
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for i in range(0, len(df), batch_size):
            batch = df[i:i + batch_size]
            filename = f"{self.config.output_dir}/techinasia_ai_news_v1_batch_{i//batch_size}_{timestamp}.csv"
            try:
                batch.to_csv(filename, index=False)
                logger.info(f"Saved batch {i//batch_size + 1} to {filename}")
            except Exception as e:
                logger.error(f"Error saving batch {i//batch_size + 1} to {filename}: {e}")

    def _log_summary(self, start_time, end_time):
        """Log summary statistics of the scraping process"""
        duration = end_time - start_time
        logger.info("📊 --- Scraping Summary ---")
        logger.info(f"📰 Total articles scraped: {self.total_articles}")
        logger.info(f"✅ Valid articles scraped: {len(self.processed_article_ids)}")
        logger.info(f"⚠️ Incomplete articles skipped: {self.incomplete_articles}")
        logger.info(f"🔄 Duplicate articles skipped: {self.total_articles - len(self.processed_article_ids) - self.incomplete_articles}")
        logger.info(f"⏱️ Scraping duration: {duration}")
        logger.info("📊 ------------------------")

def main():
    """Main function to run the scraper"""
    parser = argparse.ArgumentParser(description="TechInAsia Scraper")
    parser.add_argument('--num_articles', type=int, help='Number of articles to scrape', default=100)
    parser.add_argument('--max_scrolls', type=int, help='Maximum number of scrolls', default=15)
    parser.add_argument('--output_dir', type=str, help='Output directory', default='techinasia_output')
    parser.add_argument('--min_delay', type=float, help='Minimum delay between scrolls', default=1)
    parser.add_argument('--max_delay', type=float, help='Maximum delay between scrolls', default=3)
    parser.add_argument('--category', type=str, help='Category to scrape', default='artificial-intelligence')
    args = parser.parse_args()

    config = {
        'num_articles': args.num_articles,
        'max_scrolls': args.max_scrolls,
        'output_dir': args.output_dir,
        'min_delay': args.min_delay,
        'max_delay': args.max_delay,
        'category': args.category
    }

    logger.info("Starting TechInAsia scraper v1.7")
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

if __name__ == "__main__":
    main()