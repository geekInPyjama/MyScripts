import praw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Reddit API credentials (replace with your own)
client_id = 'yxjBi16rhRpXXke6FW20VQ'
client_secret = 'wx5uHEI38I5bnJXhTIvu1cSUbw_A5w'
user_agent = 'MyDashboard'

# Initialize the Reddit API client
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# Define the subreddit(s) you want to collect data from
subreddit_names = ['EUROSKINCARE', 'AUSSKINCARE']

# Initialize the Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Create a function to collect data from a subreddit
def collect_subreddit_data(subreddit_name, limit):
    subreddit = reddit.subreddit(subreddit_name)
    subreddit_data = []

    # Collect posts from the subreddit
    for submission in subreddit.top(limit=limit):
        # Analyze sentiment of the post title
        title = submission.title
        sentiment_scores = sia.polarity_scores(title)

        # Determine sentiment based on the compound score
        if sentiment_scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif sentiment_scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        # Append data to the subreddit_data list
        subreddit_data.append({
            'Title': title,
            'Sentiment': sentiment
        })

    return subreddit_data

# Define the data collection parameters
limit = 100  # Number of posts to retrieve (adjust as needed)

# Collect data for each subreddit
data_collection = {}
for subreddit_name in subreddit_names:
    subreddit_data = collect_subreddit_data(subreddit_name, limit)
    data_collection[subreddit_name] = subreddit_data

# Create dataframes from the collected data
dataframes = {subreddit_name: pd.DataFrame(data) for subreddit_name, data in data_collection.items()}

# Save the dataframes to CSV files
for subreddit_name, dataframe in dataframes.items():
    csv_file_path = f'{subreddit_name.lower()}_sentiment_data.csv'
    dataframe.to_csv(csv_file_path, index=False)

    print(f'Sentiment data for r/{subreddit_name} saved to {csv_file_path}')
