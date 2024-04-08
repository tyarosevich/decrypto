from pathlib import Path
import pandas as pd
from transformers import pipeline
sentiment_pipeline = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0)
from torch.cuda import is_available
assert(is_available())
from time import time

#%% Initial test
test1 = "I hate everyone."
test2 = "I love all doggos."
all_tests = [test1, test2]

test_output = sentiment_pipeline(all_tests)
print(test_output)

#%% Load this annoying csv file.
path_tweets = Path('./data/tweets_combined.csv')
dct_dtypes = {'id': int, 'text': str, 'lang': str, 'date': str, 'retweet_count': int, 'reply_count': int, 'like_count': int, 'quote_count': int}
# Had to load in chunks with the python engine because reasons.
chunk_size = 100000
start = time()
df_tweets = pd.DataFrame()
for chunk in pd.read_csv(path_tweets, chunksize=chunk_size, engine='python'):
    inner_start = time()
    df_tweets = pd.concat([df_tweets, chunk])
    inner_end = time()
    print('Chunk processing time = {}s'.format(inner_end - inner_start))

end = time()
print('Total processing time = {}s'.format(end - start))

#%% This resulted in 150 hours to do all 12 million tweets roughly.
tweet_slice = df_tweets['text'].iloc[0:1000].copy().to_list()
start = time()
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':256}
output = sentiment_pipeline(tweet_slice)
end = time()
print('Total processing time = {}s'.format(end - start))
print('Total time estimate for sentiment analysis = {} minutes'.format(df_tweets.shape[0] / (end - start)/60))
print('Average time per tweet: {} seconds'.format( (end - start) / 1000))