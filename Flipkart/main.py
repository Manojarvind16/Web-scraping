import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException, ElementNotInteractableException
import time
import nltk
import os
import sys
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger()

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Define the product name
product_name = "POCO_X6_Pro 5G"

# Initialize the WebDriver
browser = webdriver.Chrome()
browser.get("https://www.flipkart.com/poco-x6-pro-5g-racing-grey-512-gb/product-reviews/itm7134f12949dc7?pid=MOBGWMGBF7F5KBV2&lid=LSTMOBGWMGBF7F5KBV2CAM1M4&marketplace=FLIPKART")

# Lists to hold the collected data
reviews_data = []

while True:
    try:
        # Wait for the review section to be visible
        WebDriverWait(browser, 10).until(
            EC.visibility_of_element_located((By.XPATH, "//div[@class='DOjaWF gdgoEp col-9-12']"))
        )

        # Find and click all "Read More" buttons to expand the reviews
        read_more_buttons = browser.find_elements(By.XPATH, "//span[@class='b4x-fr']")
        for button in read_more_buttons:
            try:
                # Scroll the button into view
                browser.execute_script("arguments[0].scrollIntoView(true);", button)
                time.sleep(1)  # Slight pause to ensure smooth scrolling
                
                # Attempt to click the button
                try:
                    WebDriverWait(browser, 5).until(EC.element_to_be_clickable(button))
                    button.click()
                    time.sleep(2)  # Allow some time for the expanded content to load
                except (ElementClickInterceptedException, ElementNotInteractableException):
                    logger.warning(f"Direct click failed, trying JavaScript click for button at location ({button.location['x']}, {button.location['y']}).")
                    browser.execute_script("arguments[0].click();", button)
                    time.sleep(2)  # Allow some time for the expanded content to load

            except Exception as e:
                logger.error(f"Could not click 'Read More' button: {e}")

        # Extract individual ratings and reviews
        ratings = browser.find_elements(By.XPATH, "//div[@class='XQDdHH Ga3i8K']")
        reviews = browser.find_elements(By.XPATH, "//div[@class='ZmyHeo']")

        # Append each review and rating to the respective lists with the product name
        for review, rating in zip(reviews, ratings):
            reviews_data.append({
                "Product": product_name,
                "Rating": rating.text.strip(),
                "Review": review.text.strip()
            })

        # Find and click the "Next" button
        next_buttons = browser.find_elements(By.XPATH, "//span[contains(text(),'Next')]")

        if len(next_buttons) > 0:
            next_button = next_buttons[0]
            # Check if the button's parent is clickable (the <a> tag containing the "Next" span)
            next_button_parent = next_button.find_element(By.XPATH, "..")
            if next_button_parent.tag_name == 'a' and next_button_parent.get_attribute('href'):
                next_button_parent.click()
                logger.info("Navigating to the next page...")
                time.sleep(3)  # Allow some time for the next page to load
            else:
                logger.info("Next button is not clickable or no more pages.")
                break
        else:
            logger.info("No 'Next' button found. Reached the last page.")
            break

    except NoSuchElementException:
        logger.error("No more pages or error navigating to the next page.")
        break

    except TimeoutException:
        logger.error("Loading timeout or no 'Next' button found, exiting.")
        break

# Convert the collected data into a DataFrame
df = pd.DataFrame(reviews_data)

# Define the CSV file name
csv_file_name = "POCO_X6_Pro 5G.csv"

# Save the DataFrame to a CSV file
df.to_csv(csv_file_name, index=False)

logger.info(f"All reviews and ratings have been saved to {csv_file_name}.")

# Print the first few rows to verify
print(df.head())

# Close the browser
browser.quit()

# List of CSV file paths
csv_files = [
    r"C:\Users\manoj\OneDrive\Desktop\Flipkart\Motorola Edge 50 Pro 5G.csv",
    r"C:\Users\manoj\OneDrive\Desktop\Flipkart\OnePlus 11R 5G.csv",
    r"C:\Users\manoj\OneDrive\Desktop\Flipkart\POCO_X6_Pro 5G.csv",  # Ensure this path is correct
    r"C:\Users\manoj\OneDrive\Desktop\Flipkart\realme 12 Pro+ 5G.csv",
    r"C:\Users\manoj\OneDrive\Desktop\Flipkart\SAMSUNG Galaxy S22 5G.csv"
]

# Combine all CSV files into one DataFrame
combined_df_list = []
missing_files = []

for file in csv_files:
    if os.path.exists(file):
        try:
            combined_df_list.append(pd.read_csv(file))
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
    else:
        missing_files.append(file)
        logger.error(f"File not found: {file}")

if missing_files:
    logger.info("The following files were not found and were skipped:")
    for missing in missing_files:
        logger.info(missing)

# Combine the valid DataFrames into one
if combined_df_list:
    combined_df = pd.concat(combined_df_list, ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_csv_file_name = r"C:\Users\manoj\OneDrive\Desktop\Flipkart\Combined_Reviews.csv"
    combined_df.to_csv(combined_csv_file_name, index=False)

    logger.info(f"All reviews and ratings have been combined and saved to {combined_csv_file_name}.")

    # Display the first few rows of the combined DataFrame to verify
    print(combined_df.head())
else:
    logger.warning("No files were combined. Please check the file paths.")

# Perform sentiment analysis
sia = SentimentIntensityAnalyzer()
combined_df['Sentiment'] = combined_df['Review'].apply(lambda x: 'Positive' if sia.polarity_scores(x)['compound'] > 0.05 else ('Negative' if sia.polarity_scores(x)['compound'] < -0.05 else 'Neutral'))

# Save the DataFrame with sentiment analysis to a new CSV file
sentiment_csv_file_name = r"C:\Users\manoj\OneDrive\Desktop\Flipkart\Combined_Reviews_with_Sentiment.csv"
combined_df.to_csv(sentiment_csv_file_name, index=False)

logger.info(f"Sentiment analysis added and saved to {sentiment_csv_file_name}.")

# Display the first few rows with sentiment analysis to verify
print(combined_df.head())

# Group by Product and Sentiment to count occurrences
sentiment_counts = combined_df.groupby(['Product', 'Sentiment']).size().unstack(fill_value=0)

# Display sentiment counts
print(sentiment_counts)

# Plot the sentiment analysis results
sentiment_counts.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
plt.title('Sentiment Analysis of Product Reviews')
plt.xlabel('Product')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
plot_file_name = r"C:\Users\manoj\OneDrive\Desktop\Flipkart\Sentiment_Analysis_Plot.png"
plt.savefig(plot_file_name)
plt.show()

logger.info(f"Sentiment analysis plot saved to {plot_file_name}.")

# PandasAI Integration Example (Requires valid API key)
try:
    os.environ['PANDASAI_API_KEY'] = "your-api-key"  # Replace with your actual API key
    llm = OpenAI(api_token=os.environ['PANDASAI_API_KEY'])

    if llm.is_authenticated():
        sdf = SmartDataframe(combined_df, llm=llm)
        prompt_result = sdf.chat("Show me the best mobile for camera quality.")
        print(prompt_result)
    else:
        logger.warning("Invalid API key. PandasAI functionality will not be used.")

except Exception as e:
    logger.error(f"An error occurred with PandasAI: {e}")
