import pandas as pd
import tweepy
from datetime import datetime
from datetime import timedelta
import pytz
from decrypto.aws_resources import get_secret
import json

# So grabbing tweets recursively could look something like:
# TODO: add counts and date handling

def tweet_handler(query, ret_max):
    '''
    A top level wrapper to retrieve tweets. Many parameters are hard coded for the use-case, see docs for details.
    :param query: str
    :param ret_max: int
    :return: DataFrame
    '''

    twitter_api_info = get_secret()
    dct_auth = json.loads(twitter_api_info['SecretString'])
    bearer_token = dct_auth['twitter_bearer']
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

    df_tweet_batch = get_tweets(client, query, ret_max, dct_params)

    return df_tweet_batch

def api_request(client, query, ret_max, dct_params):
    """
    Iteratively makes API requests until the number of resulting records exceeds ret_max. Note this can always
    overshoot by up to dct_params['max_results'].
    :param client: Client
        A tweepy api client.
    :param query: str
    :param ret_max: int
    :param dct_params: dict
        Parameter keys are: start_time, end_time, expansions, max_results, next_token, tweet_fields
    :return: list
        List of tweet records from Twitter API get_recent_tweets.
    """
    # TODO: Finish exception handling.
    lst_all_results = []
    result = client.search_recent_tweets(query, **dct_params)
    dct_params['next_token'] = result.meta.get("next_token")
    lst_all_results += result.data
    while (len(lst_all_results) < ret_max) and (dct_params["next_token"] is not None):
        try:
            result = client.search_recent_tweets(query, **dct_params)

        except (tweepy.HTTPException, tweepy.BadRequest, tweepy.Unauthorized, tweepy.Forbidden, tweepy.NotFound,
                tweepy.TwitterServerError) as e:
            print(e)
        except Exception as e:
            print(e)
        if result.data:
            lst_all_results += result.data

        dct_params['next_token'] = result.meta.get('next_token')

    return lst_all_results

def get_start_stop():
    '''
    Returns a list of tuples containing start/stop times in 1 hour intervals from yesterday. Uses PST and converts to
    UTC.
    :return: list
    [(start_time, end_time), ...]
    '''
    naive_start_time = datetime.now(pytz.utc)
    naive_start_time = naive_start_time.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    lst_scalars = list(range(24))
    hour_delta = timedelta(hours=1)
    lst_timedeltas = [x * hour_delta for x in lst_scalars]
    lst_starts = [naive_start_time + x for x in lst_timedeltas]
    lst_ends = [x + hour_delta for x in lst_starts]
    lst_timeranges = list(zip(lst_starts, lst_ends))

    return lst_timeranges


def get_tweets(client, query, ret_max, dct_params):
    """
    Collects ret_max tweet records evenly spanning a 24 hour period from the previous day.
    :param client: Client
        A tweepy api client.
    :param query: str
    :param ret_max: int
    :param dct_params: dict
        Parameter keys are: start_time, end_time, expansions, max_results, next_token, tweet_fields
    :return: DataFrame
        Dataframe containing the tweets with the JSON style columns flattened into multiple columns.
    """
    lst_results = []
    per_hour_max = ret_max // 24
    for start_time, end_time in get_start_stop():
        dct_params['start_time'] = start_time
        dct_params['end_time'] = end_time
        dct_params['next_token'] = None
        lst_results += api_request(client, query, per_hour_max, dct_params)

    df = pd.DataFrame(lst_results)

    # Public metrics is a list of dicts with matching keys. Flattens this into columns and drops the original.
    df = pd.concat([df, pd.DataFrame(df['public_metrics'].tolist())], axis=1).drop(['public_metrics'], axis=1)
    df.rename(columns={'created_at': 'timestamp'}, inplace=True)

    return df