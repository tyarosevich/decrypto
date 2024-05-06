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
    mask_temp = np.where(df_tweets['label'] == label)[0]
    dct_sent_label_masks[label] = mask_temp
# Neutral needs to be included in some way I think. Perhaps an entire second feature category with all the same moment type features
# but just or neutral.
# sent_map = {'negative': -1, 'neutral': 0, 'positive': 1}
# df_tweets['single_val_sent'] = 0
# df_tweets['single_val_sent'] = df_tweets['label'].map(sent_map)
# # Might be a better way to do this that captures neutrality, but for monotonicity, this should be good enough for
# # cursory analysis.
# df_tweets['single_val_sent'] = df_tweets['single_val_sent'] * df_tweets['score']

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

#### IN CASE OF MP ISSUES #####
# #%% Test getting masks
#
# # To calculate the date-pairs we add something like 59m to each hourly start time.
# pair_starts = df_bitcoin_post_tweets['date'].astype('int64') / 10 ** 9
# pair_starts = pair_starts.astype(int).to_numpy()
# pair_ends = pair_starts + 59*60 + 59
# date_pairs = list(zip(pair_starts[0:-1], pair_ends[0:-1]))
#
# tweet_dates = df_tweets['date'].astype('int64') / 10 ** 9
# tweet_dates = tweet_dates.astype(int).to_numpy()
# test_pairs = date_pairs[0:1000]
# # Create index lists for span of time against the tweet timestamps.
# start = time()
# # Have to go np.where. The sparse boolean matrices are just too big.
# date_pair_masks_single_core = [np.where((tweet_dates >= date_start) & (tweet_dates <= date_end))[0] for date_start, date_end in date_pairs]
# end = time()
# print('Time taken to create date_pairs single-thread: {}'.format(end - start))
##################################


# #%% The date pairs relate all tweets to the date range for each row in the bitcoin data. This need to be further
# #   filtered into the three different sentiment types using the intersection of the sentiment label indices, and the
# #   date pair indices.
#
# date_pair_mask_source = date_pair_masks_multicore
# # Intersections were sanity checked manually.
# dct_date_pair_indices_by_sent_label = {}
# start = time()
# for label in sent_labels:
#     mask_temp = []
#     for date_pair_mask in date_pair_mask_source:
#         mask_temp.append(np.intersect1d(dct_sent_label_masks[label], date_pair_mask, assume_unique=True))
#
#     dct_date_pair_indices_by_sent_label[label] = mask_temp
# end = time()
# print('Time to process {} date pairs took {} seconds'.format(len(date_pair_masks_multicore), end - start))

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


#%% So now we need to do feature extraction. This has to iterate over each mask, performing feature extraction on both
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

df_sent_feats_neg = pd.DataFrame(map_output['sent_feats']['negative'])
all_sent_vals_per_hour = df_sent_feats_neg['sent_vals'].to_list()
df_sent_feats_neg.drop(columns=['sent_vals'], inplace=True)
curr_cols = list(df_sent_feats_neg.columns)
sent_cols = ['date'] + [col + '_neg' for col in curr_cols if col not in ['date']]
df_sent_feats_neg.rename(columns=dict(zip(curr_cols, sent_cols)), inplace=True)

df_sent_feats_pos = pd.DataFrame(map_output['sent_feats']['positive'])
all_sent_vals_per_hour = df_sent_feats_pos['sent_vals'].to_list()
df_sent_feats_pos.drop(columns=['sent_vals'], inplace=True)
curr_cols = list(df_sent_feats_pos.columns)
sent_cols = ['date'] + [col + '_pos' for col in curr_cols if col not in ['date']]
df_sent_feats_pos.rename(columns=dict(zip(curr_cols, sent_cols)), inplace=True)

df_sent_feats_neutr = pd.DataFrame(map_output['sent_feats']['neutral'])
all_sent_vals_per_hour = df_sent_feats_neutr['sent_vals'].to_list()
df_sent_feats_neutr.drop(columns=['sent_vals'], inplace=True)
curr_cols = list(df_sent_feats_neutr.columns)
sent_cols = ['date'] + [col + '_neutr' for col in curr_cols if col not in ['date']]
df_sent_feats_neutr.rename(columns=dict(zip(curr_cols, sent_cols)), inplace=True)


df_bitcoin_sent_feats_merged = df_bitcoin_post_tweets.copy()
for df in [df_sent_feats_neg, df_sent_feats_neutr, df_sent_feats_pos]:
    df_bitcoin_sent_feats_merged = pd.merge_asof(df_bitcoin_sent_feats_merged, df, on='date', tolerance=pd.Timedelta(minutes=1))


path_btc_sent_feats_merged = Path(data_folder / 'btc_sent_feats_merged.pickle')
pd.to_pickle(df_bitcoin_sent_feats_merged, path_btc_sent_feats_merged)

#%% Need to write a test to sanity check this thing. Let's pick two random hours in the btc dataframe and go slice out
#   all the data from the tweets and procedurally strip the features to compare.

#%% Save/Load current file

flag_overwrite = False
path_btc_with_sent_feats = Path(data_folder / 'btc_with_sent_feats.pickle')
if flag_overwrite:
    pd.to_pickle(path_btc_with_sent_feats, )

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
