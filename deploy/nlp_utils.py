from pysentimiento import create_analyzer
import pandas as pd
from pathlib import Path
from time import time
#%%
path = Path(r'/home/tyarosevich/Projects/deploy/data/raw_tweets_202208131844.csv')
df_test_tweets = pd.read_csv(path, low_memory=False)


#%%
analyzer = create_analyzer(task="sentiment", lang="en")

#%%
start = time()
for tweet in df_test_tweets['text'].tolist()[0:50]:
    print(tweet)
    result = analyzer.predict(tweet)
    print(result)
    print('##################################')

stop = time()
print(stop - start)

#%%
from transformers import pipeline

sentiment_analysis = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

tweets = df_test_tweets['text'].tolist()[0:50]
start = time()
for tweet in tweets:
    # print(tweet)
    result = sentiment_analysis(tweet)
    # print("Label:", result[0]['label'])
    # print("Confidence Score:", result[0]['score'])
    # print("#######################################")
stop = time()
print(stop - start)

#%% List parameter?
tweets = df_test_tweets['text'].tolist()
start = time()
result = sentiment_analysis(tweets)
elapsed = time() - start
print(elapsed)
sec_per_tweet = (elapsed / len(tweets))
print(sec_per_tweet)
sixty_thous_tweets_in_minutes = (sec_per_tweet * 60000) / 60
print(sixty_thous_tweets_in_minutes)
#%%
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

sentences = tweets[0:10]
for sentence in sentences:
    sid = SentimentIntensityAnalyzer()
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print('\n')
    print("-----------------------------------------")
