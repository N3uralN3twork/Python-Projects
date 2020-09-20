"""
Sources:
https://stackoverflow.com/questions/22469713/managing-tweepy-api-search
https://stackoverflow.com/questions/47615211/using-tweepy-api-search-to-search-for-any-element-in-a-list
https://towardsdatascience.com/how-to-scrape-more-information-from-tweets-on-twitter-44fd540b8a1f#eaf6

API key: pxrtalYKquG8xj5w3jFMN6de4
API secret key: KeYHuOkoau7TgHRSAcsh0YoYlwszVwp0DFlteofWDTE88yJixJ
Bearer Token: AAAAAAAAAAAAAAAAAAAAAGgHHwEAAAAA1S68XROfKfcq0iuMHnPmltc8qZo%3D2950y8fE8lz91scZD0KIqdWLSOLY9WSLpuu7oQx4eU4Rq2wEGG
Access Token: 3091915332-q9el7nWgroM9HD0you5umW4uSKOpYMwVGTL1Z4r
Access Token Secret: M7sdMDlCARLWF3BJKLorg8NfXKqE6DDpmLc09kKsAa5h9
"""

import tweepy
import numpy as np
import pandas as pd
pd.set_option("max_colwidth", 400)

consumer_key = 'pxrtalYKquG8xj5w3jFMN6de4'
consumer_secret = 'KeYHuOkoau7TgHRSAcsh0YoYlwszVwp0DFlteofWDTE88yJixJ'
access_token_key = '3091915332-q9el7nWgroM9HD0you5umW4uSKOpYMwVGTL1Z4r'
access_token_secret = "M7sdMDlCARLWF3BJKLorg8NfXKqE6DDpmLc09kKsAa5h9"

search_terms = ["Covid19", "COVID19", "covid", "covid19", "coronavirus"]

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
                                              lang="en", since="2020-03-01").items(2)]
# Check what we can extract from each tweet:
dir(result[0])

# Can we standardize this process:
def ScrapeTweets(query, lang, DateSince, num_tweets):
    date = []
    text = []
    geography = []
    screen_name = []
    verify = []
    followers = []
    results = [status for status in tweepy.Cursor(api.search, q=query, lang=lang, since=DateSince).items(num_tweets)]
    for tweet in results:
        date.append(tweet.created_at)
    for tweet in results:
        text.append(tweet.text)
    for tweet in results:
        geography.append(tweet.geo)
    for tweet in results:
        screen_name.append(tweet.user.screen_name)
    for tweet in results:
        verify.append(tweet.user.verified)
    for tweet in results:
        followers.append(tweet.user.followers_count)
    scraped = pd.DataFrame(data=[date, text, geography, screen_name, verify, followers])
    finale = pd.DataFrame(np.transpose(scraped))
    finale.columns = ["Created_At", "Text", "Location", "ScreenName", "Verified", "NumFollowers"]
    return finale


dataset = ScrapeTweets(query='"#covid19" OR "#covid"', lang="en", DateSince="2020-03-01", num_tweets=20)


dataset["Text"]

