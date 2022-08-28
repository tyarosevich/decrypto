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

    db_secrets = get_secret('decrypto-db-secrets')
    dct_auth = json.loads(db_secrets['SecretString'])
    db_login = dct_auth['username']
    db_pword = dct_auth['password']
    # dct_conn_args = {
    #     # "ssl": {
    #     'ssl_ca': '/home/tyarosevich/Documents/access/decrypto-db.pem'
    #     # }
    # }
    host = dct_auth['host']
    host = '{}{}'.format(host, ':3306')
    db_name = "decrypto"
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
    # Insure no weird columns sneak through.
    lst_accepted_cols = lst_index_codes = ['created_at', '^TYX', '^FVX', '^TNX', '^HSCE', '^SPGSCI', '^STI', '^BVSP', '^MXX', '^GSPTSE', '^VIX', '^DJI', '^FCHI', '^N100', 'IMOEX.ME', '^AXJO', '^NZ50', '^N225', '^TWII', '^BFX', '^TA125.TA', '^KLSE', '^BSESN', '^STOXX50E', '^IXIC', '^KS11', '^JN0U.JO', '000001.SS', '^CASE30', '^GSPC', '^FTSE', '^BUK100P', '^AORD', '^RUT', '^GDAXI', '^JKSE', '^HSI', '^NSEI', '^NSEBANK', '^SP500TR', '^DJITR', '^RUTTR', '^XNDX', '^RUITR', '^RUATR', '^RMCCTR', '^AEX', '^IRX', '^DJT', '^NDX', '^RUA', '^RUI', '^RVX', '^VXN', '^OVX', '^GVZ', '^VVIX', '^VXSLV', '^IBEX', '^SSMI', 'DX-Y.NYB', '^XAX', '^MID', '^NYA', 'TX60.TS', 'XU100.IS']

    df = df[lst_accepted_cols]
    try:
        df.to_sql(table, engine, index=False, if_exists='append', chunksize=1000, method='multi')
    except:
        raise

    return
