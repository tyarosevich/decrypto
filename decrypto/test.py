from db_resources import get_db_engine, db_read
import pandas as pd

engine = get_db_engine()
query = 'SELECT * FROM raw_tweets LIMIT 10'
df_test = db_read(query, engine)

print(df_test)