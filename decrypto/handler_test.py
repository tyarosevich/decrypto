from decrypto.twitter_search import get_tweets, get_start_stop
import pandas as pd
import tweepy
import json
from decrypto.aws_resources import get_secret

twitter_api_info = get_secret()
dct_auth = json.loads(twitter_api_info['SecretString'])
bearer_token = dct_auth['twitter_bearer']
ret_max = 1000
query = "#bitcoin lang:en"
lst_tweet_fields = ['lang', 'public_metrics', 'text', 'created_at']
dct_params = {
    'start_time': None,
    'end_time': None,
    'expansions': None,
    'max_results': 10,
    'next_token': None,
    'tweet_fields': lst_tweet_fields

}
client = tweepy.Client(bearer_token, wait_on_rate_limit=True)

def test_handler(client, query, ret_max, dct_params):

    df_tweet_batch = get_tweets(client, query, ret_max, dct_params)

    return df_tweet_batch

# TODO: Create a test table in the AWS project DB and test a function to write results to the  db.

#%% test

df = test_handler(client, query, ret_max, dct_params)
