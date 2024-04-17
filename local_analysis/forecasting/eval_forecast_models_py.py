import numpy as np
from pathlib import Path
import pandas as pd

data_folder = Path('data/')

coin_names = ['BTC', 'ETH', 'SOL', 'LINK', 'USDC']
dct_coin_tables = {}
for suffix in coin_names:
    dct_coin_tables[suffix] = pd.read_csv(Path(data_folder / 'coin_index_vals_merged_{}.csv'.format(suffix)), parse_dates=['date'])

print(dct_coin_tables.keys())

df_bitcoin = dct_coin_tables['BTC'] # A view to a table for inspection because VScode Jupyter dict inspection is a catastrophe.
print(df_bitcoin.columns)

#%%
sent_file = data_folder / 'sentiment_historical.pkl'
token_file = data_folder / 'tokens_historical.pkl'
tweet_path_thresh_retweet = data_folder / 'tweets_filt_retweet_thresh.csv'

# Tweets need to be sorted appropriately, as I forgot to do it in the original processing.
# TODO: Reprocess as sorted, I don't like it.
df_tweets = pd.read_csv(tweet_path_thresh_retweet, parse_dates=['date'])
df_tweets.reset_index(drop=True, inplace=True)

vec_sent = pd.read_pickle(sent_file)
mat_token = pd.read_pickle(token_file)

print("Sentiment shape: {} \nToken Shape: {} \nTweet Shape: {}".format(vec_sent.shape, mat_token.shape, df_tweets.shape))
assert(vec_sent.shape[0] == mat_token.shape[0])
assert(df_tweets.shape[0] == vec_sent.shape[0])

df_tweets['sentiment'] = vec_sent
df_tweets['tokens'] = mat_token.tolist()

df_tweets.sort_values(by='date', inplace=True)
df_tweets.reset_index(drop=True, inplace=True)

del vec_sent, mat_token

#%%
assert(df_bitcoin['date'].dtype == df_tweets['date'].dtype)
date_start = df_tweets['date'].min()
date_end = df_tweets['date'].max()
mask_dates = (df_bitcoin['date'] >= date_start) & (df_bitcoin['date'] <= date_end)
mask_dates = np.where(mask_dates.to_numpy())[0]
print(len(mask_dates))
df_bitcoin = df_bitcoin.iloc[mask_dates, :]

