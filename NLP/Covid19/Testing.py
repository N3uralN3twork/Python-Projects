"""
Sources:
https://stackoverflow.com/questions/22469713/managing-tweepy-api-search
https://stackoverflow.com/questions/47615211/using-tweepy-api-search-to-search-for-any-element-in-a-list
https://towardsdatascience.com/how-to-scrape-more-information-from-tweets-on-twitter-44fd540b8a1f#eaf6
https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/api-reference/get-search-tweets

API key: pxrtalYKquG8xj5w3jFMN6de4
API secret key: KeYHuOkoau7TgHRSAcsh0YoYlwszVwp0DFlteofWDTE88yJixJ
Bearer Token: AAAAAAAAAAAAAAAAAAAAAGgHHwEAAAAA1S68XROfKfcq0iuMHnPmltc8qZo%3D2950y8fE8lz91scZD0KIqdWLSOLY9WSLpuu7oQx4eU4Rq2wEGG
Access Token: 3091915332-q9el7nWgroM9HD0you5umW4uSKOpYMwVGTL1Z4r
Access Token Secret: M7sdMDlCARLWF3BJKLorg8NfXKqE6DDpmLc09kKsAa5h9
"""

import tweepy
import numpy as np
import pandas as pd
import re
import time

pd.set_option("max_colwidth", 400)

consumer_key = 'pxrtalYKquG8xj5w3jFMN6de4'
consumer_secret = 'KeYHuOkoau7TgHRSAcsh0YoYlwszVwp0DFlteofWDTE88yJixJ'
access_token_key = '3091915332-q9el7nWgroM9HD0you5umW4uSKOpYMwVGTL1Z4r'
access_token_secret = "M7sdMDlCARLWF3BJKLorg8NfXKqE6DDpmLc09kKsAa5h9"

auth = tweepy.OAuthHandler(consumer_key,
                           consumer_secret)

auth.set_access_token(access_token_key,
                      access_token_secret)
api = tweepy.API(auth,
                 wait_on_rate_limit=True,
                 wait_on_rate_limit_notify=True)

# Testing of authentication:
try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")

# Run a quick search:
result = [status for status in tweepy.Cursor(api.search, q='"#covid19" OR "#covid"',
                                              lang="en", since="2020-03-01", tweet_mode='extended').items(1)]
# Check what we can extract from each tweet:
dir(result[0])

# Get the full tweet:
result[0].full_text
result[0].source


# Can we standardize this process:
def ScrapeTweets(query, lang, num_tweets, dateuntil="2020-01-01", max_id=None):
    date = []
    text = []
    geography = []
    screen_name = []
    verify = []
    followers = []
    source = []
    results = [status for status in
               tweepy.Cursor(api.search, q=query, lang=lang, count=100,
                             until=dateuntil, tweet_mode="extended", max_id=max_id).items(num_tweets)]
    for tweet in results:
        date.append(tweet.created_at)
    for tweet in results:
        text.append(tweet.full_text)
    for tweet in results:
        geography.append(tweet.geo)
    for tweet in results:
        screen_name.append(tweet.user.screen_name)
    for tweet in results:
        verify.append(tweet.user.verified)
    for tweet in results:
        followers.append(tweet.user.followers_count)
    for tweet in results:
        source.append(tweet.source)
    scraped = pd.DataFrame(data=[date, text, geography, screen_name, verify, followers, source])
    finale = pd.DataFrame(np.transpose(scraped))
    finale.columns = ["Created_At", "Text", "Location", "ScreenName", "Verified", "NumFollowers", "Source"]
    return finale


# You can get pretty fancy with the query:
dataset = ScrapeTweets(query='"#covid19" OR "#covid" OR "coronavirus"', lang="en", num_tweets=20,
                       dateuntil="2020-09-20")
dataset


def isRetweet(tweet):
    if re.search("^RT", tweet) is not None:
        return True
    else:
        return False


dataset["isRetweet"] = dataset["Text"].apply(isRetweet)
dataset["isRetweet"].value_counts()

