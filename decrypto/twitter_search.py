import pandas as pd
import tweepy
import json
from decrypto.aws_resources import get_secret
from datetime import datetime
from datetime import timedelta

# So grabbing tweets recursively could look something like:
# TODO: add counts and date handling
def api_request(client, query, ret_max, dct_params):
    lst_all_results = []
    while len(lst_all_results) < ret_max:
        try:
            result = client.search_recent_tweets(query, **dct_params)
        except (tweepy.HTTPException, tweepy.BadRequest, tweepy.Unauthorized, tweepy.Forbidden, tweepy.NotFound,
                tweepy.TwitterServerError) as e:
            pass
        finally:
            pass

        lst_all_results += result.data
        dct_params['next_token'] = result.meta['next_token']

    return lst_all_results

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

def get_tweets(client, query, ret_max, dct_params):

    lst_results = api_request(client, query, ret_max, dct_params)
    df = pd.DataFrame(lst_results)

    return pd.concat([df, pd.DataFrame(df['public_metrics'].tolist())], axis=1).drop(['public_metrics'], axis=1)