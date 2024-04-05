import json
import os
import sys

efs_path = "/mnt/efs"
package_path =  os.path.join(efs_path, "lambda_envs/python/")
sys.path.append(package_path)

from twitter_search import tweet_handler
import pandas as pd
import json
from db_resources import get_db_engine, db_write, db_read

def lambda_handler(event, context):

    query = "#bitcoin lang:en"
    ret_max = 60000
    max_request_results = 100
    
    try:
        df_tweet_batch = tweet_handler(query, ret_max, max_request_results)
    except Exception as e:
        raise
    # print(df_tweet_batch.iloc[0:5, :].to_string())
    
    print(df_tweet_batch.shape[0])
    engine = get_db_engine()
    table = 'raw_tweets'
    db_write(df_tweet_batch, table, engine)
    
    # Test the db connection
    # engine = get_db_engine()
    # query = 'SELECT * from raw_tweets limit 10;'
    # df = db_read(query, engine)


