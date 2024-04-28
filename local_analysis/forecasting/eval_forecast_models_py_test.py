import numpy as np
from pathlib import Path
import pandas as pd
from datetime import timedelta
from importlib import reload
from time import time


# data_folder = Path('../../data/')
data_folder = Path('./data/')

coin_names = ['BTC', 'ETH', 'SOL', 'LINK', 'USDC']
dct_coin_tables = {}
for suffix in coin_names:
    dct_coin_tables[suffix] = pd.read_csv(Path(data_folder / 'coin_index_vals_merged_{}.csv'.format(suffix)), parse_dates=['date'])

print(dct_coin_tables.keys())

df_bitcoin = dct_coin_tables['BTC'] # A view to a table for inspection because VScode Jupyter dict inspection is a catastrophe.
print(df_bitcoin.columns)

#%%
tweet_sent_path = Path(data_folder / 'tweets_and_sent.pickle')
tokens_file = Path(data_folder / 'tweet_tokens.pickle')

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
mask_dates = mask_dates.to_numpy()
# mask_dates = np.where(mask_dates.to_numpy())[0]
print(np.sum(mask_dates))
# This is a view, be mindful of indices.
df_bitcoin_post_tweets = df_bitcoin[mask_dates]

#%% Collect sentiment values. I originally thought I'd combine positive/negative and ignore neutral, for simplicity,
#   but over half the tweets are neutral, and that sentiment probably carries a lot of information.
positive_cnt = np.sum(df_tweets['label'] == 'positive')
negative_cnt = np.sum(df_tweets['label'] == 'negative')
neutral_cnt = np.sum(df_tweets['label'] == 'neutral')
print('Tweets %:\nPositive: {}\nNegative: {}\nNeutral: {}'.format(positive_cnt/df_tweets.shape[0], negative_cnt/df_tweets.shape[0], neutral_cnt/df_tweets.shape[0]))
dct_sent_label_masks = {}
sent_labels = ['negative', 'neutral', 'positive']
for label in sent_labels:
    mask_temp = df_tweets['label'] == label
    dct_sent_label_masks[label] = mask_temp.to_numpy()

# Neutral needs to be included in some way I think. Perhaps an entire second feature category with all the same moment type features
# but just or neutral.
# sent_map = {'negative': -1, 'neutral': 0, 'positive': 1}
# df_tweets['single_val_sent'] = 0
# df_tweets['single_val_sent'] = df_tweets['label'].map(sent_map)
# # Might be a better way to do this that captures neutrality, but for monotonicity, this should be good enough for
# # cursory analysis.
# df_tweets['single_val_sent'] = df_tweets['single_val_sent'] * df_tweets['score']

#%% Test getting masks

# # Convert bitcoin timestamps to posix, and get pairs of one hour minus 1s timespans for each row.
# btc_dates = df_bitcoin_post_tweets['date'].astype(int) / 10 ** 9
# btc_dates = btc_dates.astype(int).to_numpy()
# date_pairs = list(zip(btc_dates[0:-2], btc_dates[1:] - 1))
#
# tweet_dates = df_tweets['date'].astype(int) / 10 ** 9
# tweet_dates = tweet_dates.astype(int).to_numpy()
#
# # Create index lists for span of time against the tweet timestamps.
# start = time()
# date_pair_masks_single_core = [ np.where((tweet_dates >= date_start) & (tweet_dates <= date_end))[0] for date_start, date_end in date_pairs ]
# end = time()
# print('Time taken to create date_pairs single-thread: {}'.format(end - start))

#%%
# Let's try with multiprocessing. Not particularly important for the date pair masks, but it probably will be for
# feature extraction, and the same limitations (needing to share a numpy array via shared mem) applies.
import multiprocessing as mp
import functools
import local_analysis.forecasting.utils_multiprocess as utils_multiprocess
reload(utils_multiprocess)
num_cores = mp.cpu_count() - 2

# Convert bitcoin timestamps to posix, and get pairs of one hour minus 1s timespans for each row.
btc_dates = df_bitcoin_post_tweets['date'].astype(int) / 10 ** 9
btc_dates = btc_dates.astype(int).to_numpy()
date_pairs = list(zip(btc_dates[0:-1], btc_dates[1:] - 1))

tweet_dates = df_tweets['date'].astype(int) / 10 ** 9
tweet_dates = tweet_dates.astype(int).to_numpy()


# Critical shared mem parameters that must be consistent across processes.
NP_SHARED_NAME = 'npshared'
NP_SHARED_TYPE = np.int64
assert(tweet_dates.dtype == NP_SHARED_TYPE)

try:
    shm = utils_multiprocess.create_shared_memory_nparray(tweet_dates, NP_SHARED_NAME)
except FileExistsError:
    utils_multiprocess.release_shared(NP_SHARED_NAME)
    shm = utils_multiprocess.create_shared_memory_nparray(tweet_dates)

runner = functools.partial(utils_multiprocess.get_date_mask, args=(tweet_dates.shape, NP_SHARED_NAME, NP_SHARED_TYPE))

# WARNING: Running this twice, even after releasing the shared memory, causes a 7:SIGBUS error. Something weird is up.
# Could be Pycharm related.
try:
    start = time()
    with mp.Pool(processes=num_cores) as pool:
        date_pair_masks_multicore = pool.map(runner, date_pairs)
    end = time()
    print('Time taken to create date_pairs with multiprocess: {}'.format(end - start))
except Exception as e:
    raise e
finally:
    try:
        shm = mp.shared_memory.SharedMemory(name=NP_SHARED_NAME)
        shm.close()
        shm.unlink()  # Free and release the shared memory block
        print('Shared memory "{}" is hypothetically released'.format(NP_SHARED_NAME))
    except Exception as e:
        print("Failed to release the shared memory.")
        raise e

#%% The date pairs relate all tweets to the date range for each row in the bitcoin data. This need to be further
#   filtered into the three different sentiment types using the intersection of the sentiment label indices, and the
#   date pair indices.

# Intersections were sanity checked manually.
dct_date_pair_indices_by_sent_label = {}
start = time()
for label in sent_labels:
    mask_temp = []
    for date_pair_mask in date_pair_masks_multicore:
        mask_temp.append(np.where(np.logical_and(dct_sent_label_masks[label], date_pair_mask))[0])

    dct_date_pair_indices_by_sent_label[label] = mask_temp
end = time()
print('Time to process {} date pairs took {} seconds'.format(len(date_pair_masks_multicore), end - start))


#%% So now we need to do feature extraction. This has to iterate over each mask, performing feature extraction on both
#   Ran into an issue with shared mem where it doesn't work with large variables past a certain size in IDEs. What's
#   Bizarre is that it works fine in the interpreter, or run in a file from command line. I altered the logic in the
#   v2 functions to use the global + copy-on-write pattern instead.

# Need to pickle the sentiment data in order to load it at the module level for multiprocess write on copy. The tokens
# are already pickled. NOTE: These paths must correspond to the module-level variables loaded in utils_feature_extraction.py
path_sent_column = Path(data_folder / 'sent_val_column.pickle')
vec_sent = df_tweets['score'].to_numpy()
pd.to_pickle(vec_sent, path_sent_column)

from local_analysis.forecasting import utils_feature_extraction
reload(utils_feature_extraction)

settings = {'sent_settings': {'extract_types': ['mean', 'median', 'max', 'moment', 'std']},
            'token_settings': {'top_n': 20}}
subset = -1 # Slice size to subset data for performance testing. Set to -1 to use all data.

# Need to pass in date pair indices that haven't been divided by sentiment in order to process tokens.
date_pair_idx_all = [np.where(tmp_date_pair)[0] for tmp_date_pair in date_pair_masks_multicore]
# date_pair_data = [{'date_start': k, 'idx_date_pair': v} for k,v in zip(df_bitcoin_post_tweets['date'].to_list(), date_pair_masks)]
date_pair_datetime_starts = df_bitcoin_post_tweets['date'].iloc[0:-1].to_list()
flag_use_mp = False

start = time()
map_output = utils_feature_extraction.multiprocess_feature_extract_v2(dct_date_pair_indices_by_sent_label,
                                                                      date_pair_datetime_starts,
                                                                      date_pair_idx_all,
                                                                      settings, flag_use_mp,
                                                                      n=subset)
end = time()
print('Time taken to extract features: {} minutes'.format((end - start)/60))

# #%% Test LSTM
# from local_analysis.forecasting.models import forecast_lstm
# from importlib import reload
# reload(forecast_lstm)
#
# btc_timeseries = df_bitcoin['close']
# train_size = int(len(btc_timeseries) * 0.9)
# test_size = int(len(btc_timeseries) * 0.1)
# train, test = btc_timeseries[0:train_size], btc_timeseries[train_size:]
#
# stride = 24
# epochs = 1000
# batch_size = 32
#
# trained_model = forecast_lstm.run_lstm_training(train, test, stride, epochs, batch_size, 'cuda')
