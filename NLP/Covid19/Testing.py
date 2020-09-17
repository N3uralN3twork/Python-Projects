"""
API key: pxrtalYKquG8xj5w3jFMN6de4
API secret key: KeYHuOkoau7TgHRSAcsh0YoYlwszVwp0DFlteofWDTE88yJixJ
Bearer Token: AAAAAAAAAAAAAAAAAAAAAGgHHwEAAAAAsxZnAwDk4QpV8s9fQsTEMXMew%2FM%3DCzZSQauNVBKgik5j5nULjkEyd2Ptuw3VP4rdMRHEpeMi1mlfr2
Access Token: 3091915332-sSi3ypvr9PE8eMWcG7SF54Dq69zfUP4PsyS9wcc
Access Token Secret: zSaJupqLcfLlvDd4NsglYEo3Vle3hiX0UCtZGpXQ1tP40
"""

import json
import tweepy

consumer_key = 'pxrtalYKquG8xj5w3jFMN6de4',
consumer_secret = 'KeYHuOkoau7TgHRSAcsh0YoYlwszVwp0DFlteofWDTE88yJixJ',
access_token_key = '3091915332-sSi3ypvr9PE8eMWcG7SF54Dq69zfUP4PsyS9wcc',
access_token_secret = "zSaJupqLcfLlvDd4NsglYEo3Vle3hiX0UCtZGpXQ1tP40"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)

api = tweepy.API(auth)

search_terms = ["Covid19", "COVID19", "covid", "covid19", "coronavirus"]


def stream_tweets(search_term):
    data = []  # empty list to which tweet_details obj will be added
    counter = 0  # counter to keep track of each iteration
    for tweet in tweepy.Cursor(api.search, q='\"{}\" -filter:retweets'.format(search_term), count=100, lang='en',
                               tweet_mode='extended').items():
        tweet_details = {}
        tweet_details['name'] = tweet.user.screen_name
        tweet_details['tweet'] = tweet.full_text
        tweet_details['retweets'] = tweet.retweet_count
        tweet_details['location'] = tweet.user.location
        tweet_details['created'] = tweet.created_at.strftime("%d-%b-%Y")
        tweet_details['followers'] = tweet.user.followers_count
        tweet_details['is_user_verified'] = tweet.user.verified
        data.append(tweet_details)

        counter += 1
        if counter == 1000:
            break
        else:
            pass
    with open('data/{}.json'.format(search_term), 'w') as f:
        json.dump(data, f)
    print('done!')

if __name__ == "__main__":
    print('Starting to stream...')
    for search_term in search_terms:
        stream_tweets(search_term)
    print('finished!')