import pandas as pd
from pathlib import Path
import os

#%%

# tabular_folder = '/home/tyarosevich/Projects/decrypto/data/'
tabular_folder = './data/' # If using container (project root is automatically mounted).
lookup_path = Path(tabular_folder + 'crypto_lookup_202304251523.csv')
crypto_prices_path = Path(tabular_folder + 'raw_crypto_prices_202304251523.csv')
stock_index_path = Path(tabular_folder + 'raw_stock_indexes_202304251524.csv')
tweets_path = Path(tabular_folder + 'raw_tweets_202304251524.csv')
stock_index_lookup_path = Path(tabular_folder + 'stock_index_lookup_202304251536.csv')
hourly_bitcoin_path = Path(tabular_folder + 'Bittrex_BTCUSD_1h.csv')
hourly_ether_path = Path(tabular_folder + 'Bittrex_ETHUSD_1h.csv')

df_ether_hourly = pd.read_csv(hourly_ether_path, header=1)
df_ether_hourly['date'] = pd.to_datetime(df_ether_hourly['date'])
df_bitcoin_hourly = pd.read_csv(hourly_bitcoin_path, header=1)
df_bitcoin_hourly['date'] = pd.to_datetime(df_bitcoin_hourly['date'])
df_crypto_lookup = pd.read_csv(lookup_path)
df_crypto_prices = pd.read_csv(crypto_prices_path)
df_crypto_prices['created_at'] = pd.to_datetime(df_crypto_prices['created_at'])
df_stock_prices = pd.read_csv(stock_index_path)
df_stock_prices['created_at'] = pd.to_datetime(df_stock_prices['created_at'])
df_tweets = pd.read_csv(tweets_path)
df_tweets['created_at'] = pd.to_datetime(df_tweets['created_at'])
df_stock_index_lookup = pd.read_csv(stock_index_lookup_path)
for df in [df_tweets, df_crypto_prices, df_stock_prices]:
    df.rename(columns={'created_at': 'date'}, inplace=True)
df_tweets['date'] = pd.to_datetime(df_tweets['date']).dt.tz_localize('UTC')

# Hourly historical Indexes
hourly_index_col_names = ['date', 'hour', 'open', 'high', 'low', 'close', 'volume']
nasdaq_hourly_path = Path(tabular_folder + 'nasdaq_hourly.csv')
s_and_p_hourly_path = Path(tabular_folder + 's_and_p_hourly.csv')
dowjones_hourly_path = Path(tabular_folder + 'dowjones_hourly.csv')
df_nasdaq_hourly = pd.read_csv(nasdaq_hourly_path, low_memory=False, names=hourly_index_col_names, delimiter=';')
df_dowjones_hourly = pd.read_csv(dowjones_hourly_path, low_memory=False, names=hourly_index_col_names, delimiter=';')
df_s_and_p_hourly = pd.read_csv(s_and_p_hourly_path, low_memory=False, names=hourly_index_col_names, delimiter=';')

for df_curr in [df_nasdaq_hourly, df_dowjones_hourly, df_s_and_p_hourly]:
    df_curr['date'] = df_curr['date'] + '-' + df_curr['hour']
    df_curr['date'] = pd.to_datetime(df_curr['date'], dayfirst=True).dt.tz_localize('US/EASTERN').dt.tz_convert('UTC')
    df_curr.drop(columns=['hour'], inplace=True)
col_append_vals = ['_SP500', '_NSDQ', '_DJ']
append_cols = ['open', 'high', 'low', 'close', 'volume']
for df_curr, append_str in zip([df_s_and_p_hourly, df_nasdaq_hourly, df_dowjones_hourly], col_append_vals):
    curr_append_cols = [col + append_str for col in append_cols]
    df_curr.rename(columns=dict(zip(append_cols, curr_append_cols)), inplace=True)

#%% Additional Crypto resources.

link_path = Path(tabular_folder + 'Gemini_LINKUSD_1h.csv')
sol_path = Path(tabular_folder + 'Gemini_SOLUSD_1h.csv')
usdc_path = Path(tabular_folder + 'Gemini_USDCUSD_1h.csv')
# ust_path = Path(tabular_folder + 'Gemini_USTUSD_1h.csv') # Data ends in 2022 for some reason.
df_link = pd.read_csv(link_path, low_memory=False)
df_sol = pd.read_csv(sol_path, low_memory=False)
df_usdc = pd.read_csv(usdc_path, low_memory=False)
# df_ust = pd.read_csv(ust_path, low_memory=False)
for df_curr in [df_link, df_sol, df_usdc]:
    df_curr['date'] = pd.to_datetime(df_curr['unix'], unit='ms', utc=True)

#%% Load the Kaggle Tweet data

path_kaggle_tweets = Path(tabular_folder + 'kaggle_bitcoin_tweets_filtered.csv')
if path_kaggle_tweets.is_file():
    df_kaggle = pd.read_csv(path_kaggle_tweets)
else:
    df_kaggle = pd.read_csv('./data/Bitcoin_tweets.csv')
    df_kaggle = df_kaggle[df_kaggle['date'].notna()]
    df_kaggle = df_kaggle[df_kaggle['is_retweet'] == False]
    df_kaggle.drop(columns=['user_name', 'user_location', 'user_description', 'user_created', 'user_followers',
                            'user_friends', 'user_favourites', 'user_verified', 'is_retweet', 'hashtags', 'source'],
                   inplace=True)
    df_kaggle['date'] = pd.to_datetime(df_kaggle['date']).dt.tz_localize('UTC')
    date_start = df_tweets['date'].min()
    date_end = df_tweets['date'].max()
    mask_kaggle_dates = (df_kaggle['date'] < date_start) | (df_kaggle['date'] > date_end)
    df_kaggle = df_kaggle[mask_kaggle_dates]
    for col in ['retweet_count', 'reply_count', 'like_count', 'quote_count']:
        df_kaggle[col] = -1

    df_kaggle.to_csv('./data/kaggle_bitcoin_tweets_filtered.csv', index=False)

#%% Combine tweet sources
df_tweets_combined = pd.concat([df_tweets, df_kaggle], ignore_index=True)
df_tweets_combined.reset_index(drop=True, inplace=True)
out_path = Path(tabular_folder + 'tweets_combined.csv')
df_tweets_combined.to_csv(out_path, index=False)
#%% Test joining stock info onto bitcoin by nearest hourly value.

def merge_index_into_crypto(df_crypto, index_frames: list):
    df_crypto.sort_values(by='date', inplace=True)
    df_merged = pd.merge_asof(df_crypto, index_frames[0], on='date', tolerance=pd.Timedelta("1h"))
    df_merged = pd.merge_asof(df_merged, index_frames[1], on='date', tolerance=pd.Timedelta("1h"))
    df_merged = pd.merge_asof(df_merged, index_frames[1], on='date', tolerance=pd.Timedelta("1h"))

    return df_merged

# This works well, but there are duplicate column names (open etc.)
index_frames = [df_s_and_p_hourly, df_nasdaq_hourly, df_dowjones_hourly]
df_btc_market_indices_merged = merge_index_into_crypto(df_bitcoin_hourly, index_frames)
df_eth_market_indices_merged = merge_index_into_crypto(df_ether_hourly, index_frames)
df_sol_market_indices_merged = merge_index_into_crypto(df_sol, index_frames)
df_link_market_indices_merged = merge_index_into_crypto(df_link, index_frames)
df_usdc_market_indices_merged = merge_index_into_crypto(df_usdc, index_frames)

coin_suffixes = ['BTC', 'ETH', 'SOL', 'LINK', 'USDC']
for df_curr, suffix in zip([df_btc_market_indices_merged, df_eth_market_indices_merged, df_sol_market_indices_merged, df_link_market_indices_merged, df_usdc_market_indices_merged], coin_suffixes):
    filename = 'coin_index_vals_merged_{}.csv'.format(suffix)
    outpath = Path(tabular_folder + filename)
    df_curr.to_csv(outpath, index=False)


#%% Testing sentiment model
