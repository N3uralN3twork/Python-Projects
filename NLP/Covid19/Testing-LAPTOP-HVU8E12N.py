"""
API key: pxrtalYKquG8xj5w3jFMN6de4
API secret key: KeYHuOkoau7TgHRSAcsh0YoYlwszVwp0DFlteofWDTE88yJixJ
Bearer Token: AAAAAAAAAAAAAAAAAAAAAGgHHwEAAAAA1S68XROfKfcq0iuMHnPmltc8qZo%3D2950y8fE8lz91scZD0KIqdWLSOLY9WSLpuu7oQx4eU4Rq2wEGG
Access Token: 3091915332-q9el7nWgroM9HD0you5umW4uSKOpYMwVGTL1Z4r
Access Token Secret: M7sdMDlCARLWF3BJKLorg8NfXKqE6DDpmLc09kKsAa5h9
"""
import os
import tweepy
import pandas as pd


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


# Create a test API object:

user = api.get_user("MikezGarcia")
print("User details:")
print(user.name)
print(user.description)
print(user.location)

search_words = "#covid19"
date_since = "2020-03-01"
tweets = tweepy.Cursor(api.search,
                       q=search_words,
                       lang="en",
                       since=date_since).items(5)

for tweet in tweets:
    print(tweet.text)






