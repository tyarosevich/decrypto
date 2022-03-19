import pandas as pd
import numpy as np
import tweepy
import json
from decrypto.aws_resources import get_secret
from datetime import datetime
from datetime import timedelta
#%%
twitter_api_info = get_secret()
dct_auth = json.loads(twitter_api_info['SecretString'])
bearer_token = dct_auth['twitter_bearer']
query = "#bitcoin lang:en"
lst_tweet_fields = ['lang', 'public_metrics', 'text', 'created_at']
params = {
    'start_time': None,
    'end_time': None,
    'expansions': None,
    'max_results': 100,
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
#%%
result_data = result.data
for tweet in result_data:
    print(tweet.data['text'])
#%% Count tweets
# count_result = client.get_recent_tweets_count("bitcoin")
# #%%
# next_token = result.meta['next_token']
# result2 = client.search_recent_tweets(query, next_token=next_token)

# So grabbing tweets recursively could look something like:
# TODO: add counts and date handling
def get_tweets(client, query, dct_params, ret_max=1000):
    lst_all_results = []
    dct_params['next_token'] = None
    while len(lst_all_results) < ret_max:
        try:
            result = client.search_recent_tweets(query, **params)
        except (tweepy.HTTPException, tweepy.BadRequest, tweepy.Unauthorized, tweepy.Forbidden, tweepy.NotFound,
                tweepy.TwitterServerError) as e:
            pass
        finally:
            pass

        lst_all_results += result.data
        dct_params['next_token'] = result.meta['next_token']

    return lst_all_results

#%%

test_two_pages = get_tweets(client, query, params, ret_max=200)
df = pd.DataFrame(test_two_pages)
df_final = pd.concat([df, pd.DataFrame(df['public_metrics'].tolist())], axis=1).drop(['public_metrics'], axis=1)

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