from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import insert
import pandas as pd
import pymysql
from aws_resources import get_secret
import json
# Note this assumes AWS Lambda, with its preconfigured logger.
import logging

logging.getLogger().setLevel(logging.INFO)


def get_db_engine():

    db_secrets = get_secret('deploy-db-secrets')
    dct_auth = json.loads(db_secrets['SecretString'])
    db_login = dct_auth['username']
    db_pword = dct_auth['password']
    # dct_conn_args = {
    #     # "ssl": {
    #     'ssl_ca': '/home/tyarosevich/Documents/access/deploy-db.pem'
    #     # }
    # }
    host = dct_auth['host']
    host = '{}{}'.format(host, ':3306')
    db_name = "deploy"
    conn_string = "mysql+pymysql://{}:{}@{}/{}".format(db_login, db_pword, host, db_name)

    # return create_engine(conn_string, connect_args=dct_conn_args)
    return create_engine(conn_string)


def db_read(sql_query, engine):
    '''
    Superfluous wrapper for pandas sql query.
    :param sql_query: str
    :param engine: Engine
    :return: DataFrame
    '''

    # try:
    #     df = pd.read_sql(sql_query, engine)
    # except Exception as e:
    #     # TODO
    #     print(e)

    df = pd.read_sql(sql_query, engine)

    return df


def db_write(df, table, engine):
    '''
    Superfluous wrapper for pandas sql write statements.
    :param df: DataFrame
    :param table: str
    :param engine: Engine
    :return: None
    '''

    try:
        df.to_sql(table, engine, index=False, if_exists='append', chunksize=1000, method='multi')
    except:
        raise

    return
