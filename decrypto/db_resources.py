import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine
import pandas as pd
import mariadb
import pymysql
from decrypto.aws_resources import get_secret
import json
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
conn_string = "mariadb+pymysql://{}:{}@decrypto-db.cmspnvwujzak.us-west-2.rds.amazonaws.com/decrypto".format(db_login,
                                                                                                               db_pword)
engine = create_engine(conn_string, connect_args=dct_conn_args)

#%%

query = "SELECT * from test;"
df = pd.read_sql(query, engine)