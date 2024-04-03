import pandas as pd
from pathlib import Path

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

#%% Test joining stock info onto bitcoin by nearest hourly value.

# This works well, but there are duplicate column names (open etc.)
df_bitcoin_hourly.sort_values(by='date', inplace=True)
df_btc_sp_merged = pd.merge_asof(df_bitcoin_hourly, df_s_and_p_hourly, on='date', tolerance=pd.Timedelta("1h"))

#%% Testing sentiment model
