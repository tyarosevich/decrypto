import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine
import pandas as pd
import mariadb
import pymysql
from decrypto.aws_resources import get_secret
import json
from decrypto.twitter_search import tweet_handler
import numpy as np
#%%
decrypto_secrets = get_secret()
dct_auth = json.loads(decrypto_secrets['SecretString'])
db_login = 'admin'
db_pword = dct_auth['decrypto_db_pword']

dct_conn_args = {
    "ssl": {
        'ssl_ca': '/home/tyarosevich/Documents/access/decrypto-db.pem'
    }
}
#%%
conn_string = "mariadb+pymysql://{}:{}@decrypto-db.cmspnvwujzak.us-west-2.rds.amazonaws.com/decrypto?charset=utf8mb4".format(db_login,
                                                                                                               db_pword)
engine = create_engine(conn_string, connect_args=dct_conn_args)

#%%

query = "SELECT * from raw_tweets;"
df = pd.read_sql(query, engine)

#%% Test writing to the raw_tweets table
table = 'raw_tweets'
# Note, create df in exploratory
df_final.to_sql(table, engine, index=False, if_exists='append')

#%% PUre python
dct_conn_args = {
    # "ssl": {
        'ssl_ca': '/home/tyarosevich/Documents/access/decrypto-db.pem'
    # }
}
host = "decrypto-db.cmspnvwujzak.us-west-2.rds.amazonaws.com:3306"
db_name = "decrypto"
conn_string = "mysql+mysqlconnector://{}:{}@{}/{}".format(db_login, db_pword, host, db_name)
engine = create_engine(conn_string, connect_args=dct_conn_args)
sql_query = "SELECT * from raw_tweets;"
df_before = pd.read_sql(sql_query, engine)

#%% Get some new tweets
query = "#bitcoin lang:en"
ret_max = 100
df_new_tweets = tweet_handler(query, ret_max)

#%% Pure pytho test writing

table = 'raw_tweets'
# Note, create df in exploratory
df_new_tweets.to_sql(table, engine, index=False, if_exists='append')

#%% Check that they were appended

df_after = pd.read_sql(sql_query, engine)

#%%
def get_db_engine():

    decrypto_secrets = get_secret()
    dct_auth = json.loads(decrypto_secrets['SecretString'])
    db_login = 'admin'
    db_pword = dct_auth['decrypto_db_pword']
    dct_conn_args = {
        # "ssl": {
        'ssl_ca': '/home/tyarosevich/Documents/access/decrypto-db.pem'
        # }
    }
    host = "decrypto-db.cmspnvwujzak.us-west-2.rds.amazonaws.com:3306"
    db_name = "decrypto"
    conn_string = "mysql+mysqlconnector://{}:{}@{}/{}".format(db_login, db_pword, host, db_name)

    return create_engine(conn_string, connect_args=dct_conn_args)

def db_read(sql_query, engine):
    '''
    Superfluous wrapper for pandas sql query.
    :param sql_query: str
    :param engine: Engine
    :return: DataFrame
    '''

    try:
        df = pd.read_sql(sql_query, engine)
    except:
        # TODO
        pass

    return df

def db_write(df, table, engine):
    '''
    Superfluous wrapper for pandas sql write statements.
    :param df: DataFrame
    :param table: str
    :param engine: Engine
    :return: None
    '''
    df.to_sql(table, engine, index=False, if_exists='append')

    return

#%% Test funs

engine = get_db_engine()

sql_query = 'select * from test;'
df_test = db_read(sql_query, engine)
df_test.loc[1] = [2, 'bar']
df_test = df_test.astype({'int_column': 'int32'})
df_test = df_test.loc[[1]]
db_write(df_test, 'test', engine)
df_after_test = db_read(sql_query, engine)