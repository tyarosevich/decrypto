from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine
import pandas as pd
import pymysql
from decrypto.aws_resources import get_secret
import json
# Note this assumes AWS Lambda, with its preconfigured logger.
import logging
logging.getLogger().setLevel(logging.INFO)

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
    try:
        df.to_sql(table, engine, index=False, if_exists='append', chunksize=1000, method='multi')
    except:
        # TODO
        pass
    return