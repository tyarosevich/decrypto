from pathlib import Path
import pandas as pd
from transformers import pipeline
from torch.cuda import is_available
assert(is_available())
from time import time
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from string import punctuation
from bs4 import BeautifulSoup


# Load the historical tweet file
path_tweets = Path('data/tweets_combined.csv')
dct_dtypes = {'id': int, 'text': str, 'lang': str, 'date': str, 'retweet_count': int, 'reply_count': int, 'like_count': int, 'quote_count': int}
# Had to load in chunks with the python engine because reasons.
chunk_size = 100000
start = time()
df_tweets = pd.DataFrame()
for chunk in pd.read_csv(path_tweets, chunksize=chunk_size, engine='python', parse_dates=['date']):
    # inner_start = time()
    df_tweets = pd.concat([df_tweets, chunk])
    # inner_end = time()
    # print('Chunk processing time = {}s'.format(inner_end - inner_start))

end = time()
print('Total processing time = {}s'.format(end - start))

thresh_retweet = 1000
mask_retweet_thresh = (df_tweets['retweet_count'] > thresh_retweet) | (df_tweets['retweet_count'] == -1)
print("{} tweets above threshold.".format(mask_retweet_thresh.sum()))
df_tweets = df_tweets[mask_retweet_thresh]
df_tweets['date'] = pd.to_datetime(df_tweets['date'])

# Sort all tweets
df_tweets.sort_values(by='date', inplace=True)
df_tweets.reset_index(drop=True, inplace=True)

tweets_thresh_path = Path('data/tweets_filt_retweet_thresh.csv')
df_tweets.to_csv(tweets_thresh_path, index=False)

#%% Process the retweet thresholded text and get accompanying token embeddings. Originally set this up to batch
#   but when I realized the gpu was not default, it became tractable to do it all at once (2.5 hours).
sent_file = Path('data/sentiment_historical.pkl')
token_file = Path('data/tokens_historical.pkl')

token_size = 64
# TODO: Batch processing needs updating if you want to use it.
flag_batch_process = False

sentiment_pipeline = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0)

if sent_file.is_file() and token_file.is_file():
    sent_processed = pd.read_pickle(sent_file)
    token_processed = pd.read_pickle(token_file)
    print("Existing files, loading.")
else:
    if flag_batch_process:
        sent_processed = np.empty((df_tweets.shape[0],))
        sent_processed.fill(np.nan)
        token_processed = np.empty((0, token_size))

    print("No processing files, creating new ones")

if flag_batch_process:
    # Find the first nan, i.e. where to start processing.
    idx_start = np.where(np.isnan(sent_processed))[0][0]
    # Something like .0035s per tweet (gpu).
    process_chunk_sz = 100
    idx_end = idx_start + process_chunk_sz
    if idx_end > sent_processed.shape[0]:
        idx_end = sent_processed.shape[0]
        print("PROCESSING FINAL CHUNK")

    tweet_batch = df_tweets['text'].iloc[idx_start:idx_end].to_list()

else:
    tweet_batch = df_tweets['text'].to_list()

#%%

start = time()
output_batch = sentiment_pipeline(tweet_batch, max_length=512, truncation=True, padding=True)
end = time()
print("Processing tweets took {} seconds".format(end - start))
df_output = pd.DataFrame(output_batch)

token_output = sentiment_pipeline.tokenizer(tweet_batch, max_length=token_size, truncation=True, padding=True)

# Process batch or process everything.
if flag_batch_process:
    sent_processed[idx_start:idx_end] = df_output['score'].to_numpy()
    token_output = [np.resize(np.array(enc.ids), token_size) for enc in token_output.encodings]
    assert(len(token_output) == df_output.shape[0])
    token_output = np.array(token_output)
    token_output = np.concatenate([token_processed, token_output]).astype(int)
    print("Chunk done!")
else:
    # sent_processed = df_output['score'].to_numpy()
    token_output = [np.resize(np.array(enc.ids), token_size) for enc in token_output.encodings]
    assert(len(token_output) == df_output.shape[0])
    mat_tokens_processed = np.array(token_output).astype(int)
    df_tweets_and_sent = pd.concat([df_tweets, df_output], axis=1)

# Save intermediate object.
# pd.to_pickle(sent_processed, sent_file)
# pd.to_pickle(token_processed, token_file)

dct_dtypes = {'id': int, 'text': str, 'lang': str, 'retweet_count': int, 'reply_count': int, 'like_count': int, 'quote_count': int}
df_tweets = df_tweets.astype(dct_dtypes)

tweet_sent_file_pkl = Path('data/tweets_and_sent.pickle')
tokens_file = Path('data/tweet_tokens.pickle')
pd.to_pickle(mat_tokens_processed, tokens_file)
start = time()
# df_tweets_and_sent.to_csv(tweet_sent_file_csv, index=False)
pd.to_pickle(df_tweets_and_sent, tweet_sent_file_pkl)
end = time()
print("Saving took {} seconds".format(end - start))
tweet_sent_file = Path('data/tweets_and_sent.parquet')
start = time()
df_tweets_and_sent.to_parquet(tweet_sent_file, engine='pyarrow') # Memory leak bug with pyarrow?
end = time()
print("Saving took {} seconds".format(end - start))
