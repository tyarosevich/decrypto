import pandas as pd
import numpy as np
import tweepy
import json
from decrypto.aws_resources import get_secret
from datetime import datetime
from datetime import timedelta
import pytz
#%%
df = pd.DataFrame([{'foo': 'bar'}])
#%%
twitter_api_info = get_secret()
dct_auth = json.loads(twitter_api_info['SecretString'])
bearer_token = dct_auth['twitter_bearer']
query = "#bitcoin lang:en"
lst_tweet_fields = ['lang', 'public_metrics', 'text', 'created_at', 'geo']
params = {
    'start_time': None,
    'end_time': None,
    'expansions': ['geo.place_id'],
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
df_final = pd.concat([df, pd.DataFrame(df['public_metrics'].tolist())], axis=1).drop(['public_metrics'], axis=1).reset_index(drop=True)
#%%
result_data = result.data
for tweet in result_data:
    print(tweet.data['text'])

#%%
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
def get_start_stop():
    '''
    A simple method to return start/end times for the twitter API call. Encapsulates the full 24 hour period of the day
    before the current call, i.e. yesterday. Second granular, but starts and ends at 00:00:00.
    :return: tuple
    (start_time, end_time)
    '''
    yesterday = datetime.now() - timedelta(days=1)
    start_time = datetime(yesterday.year, yesterday.month, yesterday.day)
    now = datetime.now()
    end_time = datetime(now.year, now.month, now.day)

    return start_time, end_time
def get_24_hour_ranges():
    local_timezone = pytz.timezone("America/Los_Angeles")
    naive_start_time = datetime(2022, 3, 24, tzinfo=local_timezone)
    utc_current_time = naive_start_time.astimezone(pytz.utc)
    utc_current_plus1 = utc_current_time + timedelta(hours=1)
    lst_timeranges = []
    for i in range(24):
        lst_timeranges.append((utc_current_time, utc_current_plus1))
        utc_current_time += timedelta(hours=1)
        utc_current_plus1 = utc_current_time + timedelta(hours=1)

    return lst_timeranges


#%% Count tweets

local_timezone = pytz.timezone("America/Los_Angeles")
naive_start_time = datetime(2022, 3, 24, 9, tzinfo=local_timezone)
utc_start_time = naive_start_time.astimezone(pytz.utc)
utc_end_time = naive_start_time + timedelta(hours=10)
#%%
# start_time = start_time - timedelta(days=5)
count_result = client.get_recent_tweets_count(query, start_time=utc_start_time, end_time=utc_end_time)
print(count_result.meta['total_tweet_count'])
#%%
lst_time_ranges = get_24_hour_ranges()
lst_counts = []

for timestamps in lst_time_ranges:
    counts = client.get_recent_tweets_count(query, start_time=timestamps[0], end_time=timestamps[1])
    lst_counts.append(counts.meta['total_tweet_count'])
#%% Look at the tweet count for this query all day yesterday by hour.

import matplotlib.pyplot as plt
x = [x[0].astimezone(pytz.timezone("America/Los_Angeles")) for x in lst_time_ranges]
y = lst_counts
ax = plt.subplot(111)
ax.bar(x, y, width=0.03)
ax.xaxis_date()
plt.show()
#%%
plt.clf()