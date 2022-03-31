import pandas as pd
import numpy as np
import tweepy
import json
from decrypto.aws_resources import get_secret
from datetime import datetime
from datetime import timedelta
#%%
df = pd.DataFrame([{'foo': 'bar'}])
#%%
twitter_api_info = get_secret()
dct_auth = json.loads(twitter_api_info['SecretString'])
bearer_token = dct_auth['twitter_bearer']
query = "#bitcoin lang:en"
lst_tweet_fields = ['lang', 'public_metrics', 'text']
params = {
    'start_time': None,
    'end_time': None,
    'expansions': None,
    'max_results': 10,
    'next_token': None,
    'tweet_fields': lst_tweet_fields

}
client = tweepy.Client(bearer_token, wait_on_rate_limit=True)
#%%
result = client.search_recent_tweets(query, **params)
df = pd.DataFrame(result.data)
# lst_meta = df['public_metrics'].tolist()
# df_meta = pd.DataFrame(lst_meta)
# df_final = pd.concat([df, df_meta], axis=1).drop(['public_metrics'], axis=1)
# #%%
#### One liner for production to save memory.
df_final = pd.concat([df, pd.DataFrame(df['public_metrics'].tolist())], axis=1).drop(['public_metrics'], axis=1)

#%% Count tweets
count_result = client.get_recent_tweets_count("bitcoin")
#%%
next_token = result.meta['next_token']
result2 = client.search_recent_tweets(query, next_token=next_token)

# So grabbing tweets recursively could look something like:
# TODO: add counts and date handling
def get_tweets(client, query, dct, next_token=None):
    result = client.search_recent_tweets(query, next_token=next_token)
    # store results
    next_token = result.meta['next_token']
    return dct if not next_token else get_tweets(client, query, next_token)

#%%
query = "bitcoin lang:en"
yesterday = datetime.now() - timedelta(days = 1)
start_tie = datetime(yesterday.year, yesterday.month, yesterday.day).strftime("%Y-%m-%dT%H:%M:%SZ")
now = datetime.now()
end_time = datetime(now.year, now.month, now.day).strftime("%Y-%m-%dT%H:%M:%SZ")
max_resuls = 510

#%% v1 test
consumer_key = dct_auth['twitter_dev']
consumer_secret = dct_auth['twitter_secret']
access_token = dct_auth['twitter_access']
access_token_secret = dct_auth['twitter_access_secret']
bearer_token = dct_auth['twitter_bearer']
# auth = tweepy.OAuth1UserHandler(
#    consumer_key, consumer_secret, access_token, access_token_secret
# )

auth = tweepy.OAuth2BearerHandler(bearer_token)
api = tweepy.API(auth)
#%%
test_return = api.search_30_day(label='Developer', query="bitcoin", max_resuls=10)