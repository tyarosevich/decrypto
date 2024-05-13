# TODO: Add quartiles to feature extraction.

import numpy as np
from pathlib import Path
import pandas as pd
from datetime import timedelta
from importlib import reload
from time import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from string import punctuation
from bs4 import BeautifulSoup
import multiprocessing as mp
import re
from datetime import datetime
import pytz
from matplotlib import pyplot as plt
import seaborn as sns


# data_folder = Path('../../data/')
data_folder = Path('./data/')
#%%
coin_names = ['BTC', 'ETH', 'SOL', 'LINK', 'USDC']
dct_coin_tables = {}
for suffix in coin_names:
    df_temp = pd.read_csv(Path(data_folder / 'coin_index_vals_merged_{}.csv'.format(suffix)), parse_dates=['date'])
    curr_cols = list(df_temp.columns)
    bad_cols = [col for col in curr_cols if 'Volume' in col]
    good_cols = [col.lower().replace(' ', '_') for col in bad_cols]
    df_temp.rename(columns=dict(zip(bad_cols, good_cols)), inplace=True)
    dct_coin_tables[suffix] = df_temp

print(dct_coin_tables.keys())

df_bitcoin = dct_coin_tables['BTC'] # A view to a table for inspection because VScode Jupyter dict inspection is a catastrophe.
print(df_bitcoin.columns)

#%% The 1 hr lagged corr was a product of me misinterpreting statsmodels bad documentation. There is no lagged correlation.
df_btc_eth = df_bitcoin[['date', 'close']].merge(dct_coin_tables['ETH'][['date', 'close']], how='left', on='date', suffixes=['_btc', '_eth'])
df_btc_eth['close_eth'] = df_btc_eth['close_eth'].ffill()
btc_close_diff = np.diff(df_btc_eth['close_btc'].to_numpy())
eth_close_diff = np.diff(df_btc_eth['close_eth'].to_numpy())

slice_start = 0
ether_increase = (df_btc_eth['close_eth'].iloc[-1] - df_btc_eth['close_eth'].iloc[slice_start]) / df_btc_eth['close_eth'].iloc[slice_start]
df_btc_eth = df_btc_eth.iloc[slice_start:, :]

pot_start = 1000
cnt_ether = pot_start / df_btc_eth['close_eth'].iloc[0]
pot = 0
cashout_vals = []
for i in range(2, df_btc_eth.shape[0], 1):
    curr_eth_val = df_btc_eth['close_eth'].iloc[i]
    btc_delta = df_btc_eth['close_btc'].iloc[i] - df_btc_eth['close_btc'].iloc[i - 1]
    if btc_delta < -10 and pot == 0:
        pot = cnt_ether * curr_eth_val
        cnt_ether = 0
        cashout_vals.append(pot)
    elif btc_delta > 10 and pot > 0:
        cnt_ether = pot / curr_eth_val
        pot = 0

if pot == 0:
    pot = cnt_ether * df_btc_eth['close_eth'].iloc[-1]

print(pot)
time_delta = df_btc_eth['date'].iloc[-1] - df_btc_eth['date'].iloc[0]
print("Total return of {:.2f}% over the course of {}".format(pot/pot_start * 100 - 100, time_delta))
print("ETH return untraded: {}".format(ether_increase * 100))

import seaborn as sns
from matplotlib import pyplot as plt
sns.lineplot(x=range(len(cashout_vals)), y = cashout_vals)
plt.show()

#%% What about with my own, rock solid, daily data?
path_btc_eth_daily = Path(data_folder / 'raw_crypto_prices_202304251523.csv')
df_btc_eth_daily = pd.read_csv(path_btc_eth_daily)
df_btc_eth_daily['created_at'] = pd.to_datetime(df_btc_eth_daily['created_at']).dt.tz_localize('UTC')

pot_start = 1000
cnt_ether = pot_start / df_btc_eth_daily['ETHUSD'].iloc[0]
pot = 0
for i in range(1, df_btc_eth_daily.shape[0], 1):
    curr_eth_val = df_btc_eth_daily['ETHUSD'].iloc[i]
    flag_btc_increase = df_btc_eth_daily['BTCUSD'].iloc[i] - df_btc_eth_daily['BTCUSD'].iloc[i - 1] > 0
    if not flag_btc_increase and pot == 0:
        pot = cnt_ether * curr_eth_val
        cnt_ether = 0
    elif flag_btc_increase and pot > 0:
        cnt_ether = pot / curr_eth_val
        pot = 0

if pot == 0:
    pot = cnt_ether * df_btc_eth_daily['ETHUSD'].iloc[-2]

print(pot)
time_delta = df_btc_eth_daily['created_at'].iloc[-2] - df_btc_eth_daily['created_at'].iloc[0]
print("Total return of {:.2f} over the course of {}".format(pot/pot_start * 100 - 100, time_delta))


#%%
tweet_sent_path = Path(data_folder / 'tweets_and_sent.pickle')

df_tweets = pd.read_pickle(tweet_sent_path)
print("Tweet and sentiment shape: {}".format(df_tweets.shape))

#%% Tweet sentiment is done, but the tweet tokens to be used for features need to be generated differently, i.e.
#   cleaned, lemmatized etc.

path_clean_tweet_tokens = Path(data_folder / 'clean_tweet_tokens.pickle')
overwrite = False
if overwrite or not path_clean_tweet_tokens.exists():
    from transformers import pipeline
    sentiment_pipeline = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0)

    # Any weird words I want to include.
    additional_words = []

    remove_words = set(stopwords.words('english') + list(punctuation) + additional_words)
    cleaned_tweets = []


    def clean_tweet(text):
        # First do pre-split cleaning
        text = text.lower()
        text = BeautifulSoup(text, 'lxml').text

        # Any regex matches you want to drop. Opening HTML strikethroughs were leaking through for example.
        # Explanation of regexes: [remove <xxx> up to 3 letters enclosed, remove URLs,  ]
        regex_list = ['<[^>]+>', r'http\S+']
        for regex in regex_list:
            text = re.sub(regex, '', text)

        # Tokenized cleaning
        text_split = word_tokenize(text)
        stopwords_removed = [word for word in text_split if word not in remove_words]
        lemmatizer = WordNetLemmatizer()
        cleaned_text = [lemmatizer.lemmatize(word) for word in stopwords_removed]

        return " ".join(cleaned_text)

    all_tweets = df_tweets['text']

    # Do any vectorized str functions.
    all_tweets = all_tweets.str.replace(r"http\S+", "")

    # This is cpu bound.
    start = time()
    num_cores = mp.cpu_count() - 2
    with mp.Pool(processes=num_cores) as pool:
        cleaned_tweets = list(pool.map(clean_tweet, all_tweets))
    end = time()
    print("Time taken to clean tweets: {}".format(end - start))

    tweet_tokens = sentiment_pipeline.tokenizer(cleaned_tweets, max_length=64, truncation=True, padding=True)
    token_output = [np.resize(np.array(enc.ids), 64) for enc in tweet_tokens.encodings]
    assert (len(token_output) == df_tweets.shape[0])
    mat_tokens = np.array(token_output).astype(int)

    pd.to_pickle(mat_tokens, path_clean_tweet_tokens)

else:
    mat_tokens = pd.read_pickle(path_clean_tweet_tokens)
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
    idx_temp = np.where(df_tweets['label'] == label)[0]
    dct_sent_label_masks[label] = idx_temp

#%% Test getting masks with new generalized mp map wrapper.
from local_analysis.forecasting import utils_multiprocess
reload(utils_multiprocess)

def get_date_mask(date_range, dct_shared_data) -> np.ndarray:

    tweet_dates = dct_shared_data['tweet_dates']['ndarray']
    mask_output = np.where((tweet_dates >= date_range[0]) & (tweet_dates < date_range[1]))[0]

    return mask_output

# To calculate the date-pairs we add something like 59m to each hourly start time.
pair_starts = df_bitcoin_post_tweets['date'].astype('int64') / 10 ** 9
pair_starts = pair_starts.astype(int).to_numpy()
pair_ends = pair_starts + 59*60 + 59
date_pairs = list(zip(pair_starts[0:-1], pair_ends[0:-1]))

tweet_dates = df_tweets['date'].astype('int64') / 10 ** 9
tweet_dates = tweet_dates.astype(int).to_numpy()

shared_data = {'tweet_dates': tweet_dates}
input_data = (date_pairs,)
mp_settings = {'num_cores_kept_free': 2}

start = time()
date_pair_masks_multicore = utils_multiprocess.run_shared_mem_multiproc_map(input_data, get_date_mask, shared_data, mp_settings)
end = time()
print('Time taken to run multiprocessing map: {} seconds'.format(end - start))

#%% This was super slow. Let's try mapping with shared mem again. Was super CPU bound, almost linear speedup x cores.
    # Niiice.

def get_sent_date_intersect(idx_dates, dct_shared_data):
    idx_sent = dct_shared_data['sent_indices']['ndarray']
    return np.intersect1d(idx_sent, idx_dates, assume_unique=True)


dct_date_pair_indices_by_sent_label = {}
date_pair_mask_source = date_pair_masks_multicore
mp_settings = {'num_cores_kept_free': 2}
start = time()
for label in sent_labels:
    shared_data = {'sent_indices': dct_sent_label_masks[label]}
    input_data = (date_pair_mask_source,)
    dct_date_pair_indices_by_sent_label[label] = utils_multiprocess.run_shared_mem_multiproc_map(
        input_data, get_sent_date_intersect, shared_data, mp_settings)
end = time()
print('Time to process sent/date pair intersections: {} minutes'.format( (end - start) / 60) )


#%% So now we need to do feature extraction. This has to iterate over each sentiment-types index list,
#   performing feature extraction each one independently.
#   Ran into an issue with shared mem where it doesn't work with large variables past a certain size in IDEs. What's
#   Bizarre is that it works fine in the interpreter, or run in a file from command line. I altered the logic in the
#   v2 functions to use the module-scoped variable + copy-on-write pattern instead. This requires saving the sent values
#   and token values to pickles which are then loaded in the module at import.

# Need to pickle the sentiment data in order to load it at the module level for multiprocess write on copy. The tokens
# are already pickled. NOTE: These paths must correspond to the module-level variables loaded in utils_feature_extraction.py
# path_sent_column = Path(data_folder / 'sent_val_column.pickle')
# vec_sent = df_tweets['score'].to_numpy()
# pd.to_pickle(vec_sent, path_sent_column)

from local_analysis.forecasting import utils_feature_extraction
reload(utils_feature_extraction)

settings = {'sent_settings': {'extract_types': ['mean', 'median', 'max', 'min', 'std', 'variance',
                                                'skewness', 'kurtosis']},
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

#%% Format the extracted features and join them onto the bitcoin dataframe.

df_token_feats = pd.DataFrame(map_output['token_feats'])
all_tokens_per_hour = df_token_feats['token_vals'].to_list()
df_token_feats.drop(columns=['token_vals'], inplace=True)
tmp_dtypes = ['Int64'] * 20 # Note this is panda's nullable integer dtype.
dct_token_df_types = dict(zip(df_token_feats.columns[1:], tmp_dtypes))
df_token_feats = df_token_feats.astype(dct_token_df_types)

dct_sentiment_frames = {}
for label in sent_labels:
    dct_sentiment_frames[label] = pd.DataFrame(map_output['sent_feats'][label])
    dct_sentiment_frames[label].drop(columns=['sent_vals'], inplace=True)
    curr_cols = list(dct_sentiment_frames[label].columns)
    sent_cols = ['date'] + [col + '_' + label[0:3] for col in curr_cols if col not in ['date']]
    dct_sentiment_frames[label].rename(columns=dict(zip(curr_cols, sent_cols)), inplace=True)

dct_sentiment_frames['tokens'] = df_token_feats

df_bitcoin_tweet_feats_merged = df_bitcoin_post_tweets.copy()
for df in dct_sentiment_frames.values():
    df_bitcoin_tweet_feats_merged = pd.merge_asof(df_bitcoin_tweet_feats_merged, df, on='date', tolerance=pd.Timedelta(minutes=1))


path_btc_tweet_feats_merged = Path(data_folder / 'btc_tweet_feats_merged.pickle')
pd.to_pickle(df_bitcoin_tweet_feats_merged, path_btc_tweet_feats_merged)

#%% Starting point to continue from just the merged btc/feat dataframe.
path_btc_tweet_feats_merged = Path(data_folder / 'btc_tweet_feats_merged.pickle')
df_bitcoin_tweet_feats_merged = pd.read_pickle(path_btc_tweet_feats_merged)

#%% This test checks a few random hours and calculates a few different sentiment values, then compares to the
#   processed values in df_bitcoin_tweet_feats_merged.

from local_analysis.tests.test_feature_extraction import test_tweet_feat_sanity_check
test_tweet_feat_sanity_check()
#%%
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

#%% Decode tokens to see what these common ones are.
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
input_ids = [741, 42988, 11388, 6569, 15113, 7471, 417, 21416, 24987]
# print(input_ids)
# decoded = tokenizer.decode(input_ids)
for id in input_ids:
    print('{} : {}'.format(id, str(tokenizer.decode(id))))

#%% Fill strategy.

df_info = df_bitcoin_tweet_feats_merged.info()
df_describe = df_bitcoin_tweet_feats_merged.describe()
# These indices are representative of three categories of sentiment features having a significant number of nans.
idx_skew_neg_null = np.where(df_bitcoin_tweet_feats_merged['skewness_neg'].isnull().values)[0]
idx_token_null = np.where(df_bitcoin_tweet_feats_merged['freq_token_8'].isnull().values)[0]
idx_mean_neu_null = np.where(df_bitcoin_tweet_feats_merged['mean_neu'].isnull().values)[0]
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
sns.histplot(x=idx_skew_neg_null, ax=ax[0])
sns.histplot(x=idx_token_null, ax=ax[1])
sns.histplot(x=idx_mean_neu_null, ax=ax[2])
ax[0].set_title('Skewness Negative')
ax[1].set_title('Token Positive')
ax[2].set_title('Mean Neutral')
plt.show()
# Results are very uniform.

# All sentiment values are filled with interp.
interp_columns = [col for col in df_bitcoin_tweet_feats_merged.columns if any(substring in col for substring in ['pos', 'neg', 'neu'])]
# Interp makes no sense for tokens, and the index fund NaNs are usually due to being closed, so these are forward filled.
ffill_columns = [col for col in df_bitcoin_tweet_feats_merged.columns if any(substring in col for substring in ['freq', 'low', 'high', 'volume', 'open', 'close']) and not any(substring in col for substring in ['SOL', 'UST'])]
# SOL/UST didn't exist prior to some date.
zero_columns = ['close_SOL', 'close_UST']
df_bitcoin_tweet_feats_merged[interp_columns] = df_bitcoin_tweet_feats_merged[interp_columns].interpolate().copy()
df_bitcoin_tweet_feats_merged[ffill_columns] = df_bitcoin_tweet_feats_merged[ffill_columns].ffill().copy()
df_bitcoin_tweet_feats_merged[zero_columns] = df_bitcoin_tweet_feats_merged[zero_columns].fillna(0).copy()

# Finally, add missing date timestamps, forward fill, and switch the index to datetime.
df_bitcoin_tweet_feats_merged['date'] = pd.to_datetime(df_bitcoin_tweet_feats_merged['date'].dt.tz_localize(None))
date_index = pd.date_range(df_bitcoin_tweet_feats_merged['date'].iloc[0], df_bitcoin_tweet_feats_merged['date'].iloc[-1], freq='h')
df_bitcoin_tweet_feats_merged['date'] = pd.DatetimeIndex(df_bitcoin_tweet_feats_merged['date'])
df_bitcoin_tweet_feats_merged.set_index('date', inplace=True, drop=False)

# Now re-index to the forced data range and forward fill. Only a few hours out of 19,000 were missing.
df_bitcoin_tweet_feats_merged = df_bitcoin_tweet_feats_merged.reindex(date_index, method='ffill')
# Need to re-do day of the week - I did it earlier to see it in the correlation matrices.
df_bitcoin_tweet_feats_merged['day_of_week'] = df_bitcoin_tweet_feats_merged['date'].dt.day_of_week

df_bitcoin_tweet_feats_merged.info()

#%% Let's take a look at some heat maps. Doing all at once would be an info overload, and some of the features wouldn't
#   surprise anyone if they were correlated (e.g. SP500 vs Dow Jones). Similarly, if the token values were correlated
#   with the sentiment values, this doesn't reveal much. So I'll do a few different ones with what I view
#   as sensible combinations.

# Let's start here. Sentiment on the one hand, and token values on the other.
heat_map_cols_1 = ['close', 'day_of_week', 'volume_usd', 'close_ETH', 'close_SOL', 'close_UST', 'close_SP500',
                   'close_NSDQ_x', 'mean_neg', 'mean_pos', 'mean_neu']
heat_map_cols_2 = ['close', 'volume_usd'] + [col for col in list(df_bitcoin_tweet_feats_merged.columns) if 'freq' in col]
df_heatmap_1 = df_bitcoin_tweet_feats_merged[heat_map_cols_1]
df_heatmap_2 = df_bitcoin_tweet_feats_merged[heat_map_cols_2]

plt.figure(figsize=(16, 6))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(df_heatmap_1.corr(), dtype=bool))
heatmap = sns.heatmap(df_heatmap_1.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='coolwarm')
heatmap.set_title('BTC Close Corr Heatmap w/ Indices and Sentiment', fontdict={'fontsize':18}, pad=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 6))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(df_heatmap_2.corr(), dtype=bool))
heatmap = sns.heatmap(df_heatmap_2.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='coolwarm')
heatmap.set_title('BTC Close Corr Heatmap w/ Top n roBERTa tokens', fontdict={'fontsize':18}, pad=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% Some final encoding/formatting. Extracting month and one-hot encoding day/month, also adding holidays.
from pandas.tseries.holiday import USFederalHolidayCalendar

holiday_cal = USFederalHolidayCalendar()
holidays = holiday_cal.holidays(start=df_bitcoin_tweet_feats_merged['date'].iloc[0], end=df_bitcoin_tweet_feats_merged['date'].iloc[-1])
df_bitcoin_tweet_feats_merged['flag_holiday'] = df_bitcoin_tweet_feats_merged['date'].isin(holidays)
df_bitcoin_tweet_feats_merged['month'] = df_bitcoin_tweet_feats_merged['date'].dt.month

one_hot_cols = ['day_of_week', 'month']
df_bitcoin_tweet_feats_merged = pd.get_dummies(df_bitcoin_tweet_feats_merged, columns=one_hot_cols)



path_btc_tweet_merged_final = Path(data_folder / 'btc_tweet_merged_final.pickle')
pd.to_pickle(df_bitcoin_tweet_feats_merged, path_btc_tweet_merged_final)

#%% Loading point to begin modeling.

path_btc_tweet_merged_final = Path(data_folder / 'btc_tweet_merged_final.pickle')
df_btc_data = pd.read_pickle(path_btc_tweet_merged_final)

#%% Imports
from darts.timeseries import TimeSeries

#%% Note that indices were switched to dateindex upstream, and all missing hours filled, so that the time series
#   object could be created smoothly.

train_cols_univ = 'close'
train_cols_multiv = [col for col in df_btc_data.columns if col not in ['unix', 'symbol', 'date']]
ts_btc_data_univ = TimeSeries.from_dataframe(df_btc_data, value_cols=train_cols_univ)
ts_btc_data_multiv = TimeSeries.from_dataframe(df_btc_data, value_cols=train_cols_multiv)

ts_btc_data_univ.plot()
plt.show()