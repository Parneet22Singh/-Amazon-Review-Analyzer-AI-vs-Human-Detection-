# Scraper.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pandas as pd
import os
import time

def scrape_amazon_reviews(product_url, max_pages=10):
    dataset_dir = "dataset"
    output_csv = os.path.join(dataset_dir, "reviews.csv")
    chromedriver_path = os.path.join(os.path.dirname(__file__), "chromedriver.exe")

    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--headless=new")  # Modern headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Use user-data-dir to persist login and session cookies
    user_data_dir = os.path.abspath("chrome_user_data")
    options.add_argument(f"--user-data-dir={user_data_dir}")
    options.add_argument("--profile-directory=Default")

    print("🚀 Launching Chrome with user profile...")
    driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)
    driver.get(product_url)
    time.sleep(5)

    print("🔍 Scraping reviews...")
    reviews = []

    for page in range(max_pages):
        review_blocks = driver.find_elements(By.XPATH, '//span[@data-hook="review-body"]')
        for block in review_blocks:
            reviews.append(block.text.strip())

        try:
            next_button = driver.find_element(By.XPATH, '//li[@class="a-last"]/a')
            next_button.click()
            time.sleep(3)
        except:
            print("🔚 No more pages.")
            break

    driver.quit()

    if reviews:
        os.makedirs(dataset_dir, exist_ok=True)
        df = pd.DataFrame({'Review': reviews})
        df.to_csv(output_csv, index=False)
        print(f"✅ {len(reviews)} reviews scraped.")
        return df
    else:
        print("⚠️ No reviews found.")
        return pd.DataFrame()
