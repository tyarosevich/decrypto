import numpy as np
from pathlib import Path
import pandas as pd
from datetime import timedelta
from time import time
data_folder = Path('data/')

coin_names = ['BTC', 'ETH', 'SOL', 'LINK', 'USDC']
dct_coin_tables = {}
for suffix in coin_names:
    dct_coin_tables[suffix] = pd.read_csv(Path(data_folder / 'coin_index_vals_merged_{}.csv'.format(suffix)), parse_dates=['date'])

print(dct_coin_tables.keys())

df_bitcoin = dct_coin_tables['BTC'] # A view to a table for inspection because VScode Jupyter dict inspection is a catastrophe.
print(df_bitcoin.columns)

#%%
tweet_sent_file_csv = Path('data/tweets_and_sent.csv')

# Tweets need to be sorted appropriately, as I forgot to do it in the original processing.
# TODO: Reprocess as sorted, I don't like it.
df_tweets = pd.read_csv(tweet_sent_file_csv, parse_dates=['date'])
df_tweets.reset_index(drop=True, inplace=True) # Not sure if I did this originally.

print("Tweet and sentiment shape: {}".format(df_tweets.shape))


#%%
assert(df_bitcoin['date'].dtype == df_tweets['date'].dtype)
date_start = df_tweets['date'].min()
date_end = df_tweets['date'].max()

# Note the date mask needs to be kept around for some models and not others. In some cases, e.g. TFT, I believe the
# tweet data can be nan-filled and used with the full crypto dataset. In others, e.g. ARIMA, I believe all multivariate
# features need to be present.
mask_dates = (df_bitcoin['date'] >= date_start) & (df_bitcoin['date'] <= date_end)
mask_dates = np.where(mask_dates.to_numpy())[0]
print(len(mask_dates))

df_bitcoin_post_tweets = df_bitcoin.iloc[mask_dates, :]

#%% Produce a single sentiment column for analysis
# Note that neutral tweets are fairly unusual, and if we believe in the hypothesis that tweet sentiment reflects market
# confidence, then neutral tweets don't contain actionable information anyway. Furthermore, neutral is ambiguous: a false
# neutral's true sentiment can be either positive or negative.
positive_cnt = np.sum(df_tweets['label'] == 'positive')
negative_cnt = np.sum(df_tweets['label'] == 'negative')
neutral_cnt = np.sum(df_tweets['label'] == 'neutral')
print('Tweets %:\nPositive: {}\nNegative: {}\nNeutral: {}'.format(positive_cnt/df_tweets.shape[0], negative_cnt/df_tweets.shape[0], neutral_cnt/df_tweets.shape[0]))


sent_map = {'negative': -1, 'neutral': 0, 'positive': 1}
df_tweets['single_val_sent'] = 0
df_tweets['single_val_sent'] = df_tweets['label'].map(sent_map)
# Might be a better way to do this that captures neutrality, but for monotonicity, this should be good enough for
# cursory analysis.
df_tweets['single_val_sent'] = df_tweets['single_val_sent'] * df_tweets['score']

#%% Test getting masks
one_second = timedelta(seconds=1)
date_pairs = list(zip(df_bitcoin_post_tweets['date'].iloc[0:-2].to_numpy(), df_bitcoin_post_tweets['date'].iloc[1:].to_numpy() - one_second))
start = time()
date_pair_masks = [ np.where((df_tweets['date'] >= date_start) & (df_tweets['date'] <= date_end))[0] for date_start, date_end in date_pairs ]
end = time()
print('Time taken to create date_pairs single-thread: {}'.format(end - start))

# Let's multiprocess it
import multiprocessing as mp
import local_analysis.forecasting.utils_multiprocess as utils_multiprocess
num_cores = mp.cpu_count() - 2
tweet_dates = df_tweets['date'].to_numpy()

NP_SHARED_NAME = 'npshared'


# def get_date_mask(date_range: tuple) -> np.ndarray:
#
#     shm = shared_memory.SharedMemory(name=name)
#     np_array = np.ndarray(ARRAY_SHAPE, dtype=NP_DATA_TYPE, buffer=shm.buf)
#     return np.where( (tweet_dates >= date_range[0]) & (tweet_dates < date_range[1]))[0]

try:
    shm = utils_multiprocess.create_shared_memory_nparray(tweet_dates)
except FileExistsError:
    utils_multiprocess.release_shared(NP_SHARED_NAME)
    shm = utils_multiprocess.create_shared_memory_nparray(tweet_dates)

start = time()
with mp.Pool(processes=num_cores) as pool:
    processed = pool.map(utils_multiprocess.get_date_mask, date_pairs)
end = time()
print('Time taken to create date_pairs with multiprocess: {}'.format(end - start))