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
tweet_sent_path =  Path('data/tweets_and_sent.pickle')
tokens_file = Path('data/tweet_tokens.pickle')

df_tweets = pd.read_pickle(tweet_sent_path)
mat_tokens = pd.read_pickle(tokens_file)
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
# This is a view, be mindful of indices.
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

# Convert bitcoin timestamps to posix, and get pairs of one hour minus 1s timespans for each row.
btc_dates = df_bitcoin_post_tweets['date'].astype(int) / 10 ** 9
btc_dates = btc_dates.astype(int).to_numpy()
date_pairs = list(zip(btc_dates[0:-2], btc_dates[1:] - 1))

tweet_dates = df_tweets['date'].astype(int) / 10 ** 9
tweet_dates = tweet_dates.astype(int).to_numpy()

# Create index lists for span of time against the tweet timestamps.
start = time()
date_pair_masks = [ np.where((tweet_dates >= date_start) & (tweet_dates <= date_end))[0] for date_start, date_end in date_pairs ]
end = time()
print('Time taken to create date_pairs single-thread: {}'.format(end - start))

#%%
# Let's try with multiprocessing. Not particularly important for the date pair masks, but it probably will be for
# feature extraction, and the same limitatiosn (needing to share a numpy array via shared mem) applies.
import multiprocessing as mp
import functools
import local_analysis.forecasting.utils_multiprocess as utils_multiprocess
num_cores = mp.cpu_count() - 2

NP_SHARED_NAME = 'npshared'


# def get_date_mask(date_range: tuple) -> np.ndarray:
#
#     shm = shared_memory.SharedMemory(name=name)
#     np_array = np.ndarray(ARRAY_SHAPE, dtype=NP_DATA_TYPE, buffer=shm.buf)
#     return np.where( (tweet_dates >= date_range[0]) & (tweet_dates < date_range[1]))[0]

try:
    shm = utils_multiprocess.create_shared_memory_nparray(tweet_dates, NP_SHARED_NAME)
except FileExistsError:
    utils_multiprocess.release_shared(NP_SHARED_NAME)
    shm = utils_multiprocess.create_shared_memory_nparray(tweet_dates)

runner = functools.partial(utils_multiprocess.get_date_mask, array_shape = tweet_dates.shape)

# So this isn't working, some kind of typecasting is happening with the datetime objects. Either try doing the whole shebang with
# unix time or see if there's a way to do a list to shared memory (kinda doubt it, I think numpy arrys are special because of
# contiguous memory).
start = time()
with mp.Pool(processes=num_cores) as pool:
    processed = pool.map(runner, date_pairs)
end = time()
print('Time taken to create date_pairs with multiprocess: {}'.format(end - start))
utils_multiprocess.release_shared(NP_SHARED_NAME)

#%% So now we need to do feature extraction. This has to iterate over each mask, performing feature extraction on both
#   the sentiment array and token matrix, which will need to be in shared mem (particularly the matrix, it's big).

vec_sent = df_tweets['single_val_sent'].to_numpy()

#%% Test LSTM
from local_analysis.forecasting.models import forecast_lstm
from importlib import reload
reload(forecast_lstm)

btc_timeseries = df_bitcoin['close']
train_size = int(len(btc_timeseries) * 0.9)
test_size = int(len(btc_timeseries) * 0.1)
train, test = btc_timeseries[0:train_size], btc_timeseries[train_size:]

stride = 24
epochs = 1000
batch_size = 32

trained_model = forecast_lstm.run_lstm_training(train, test, stride, epochs, batch_size, 'cuda')
