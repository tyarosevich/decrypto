from pysentimiento import create_analyzer
import pandas as pd
from pathlib import Path
from time import time
#%%
path = Path(r'/home/tyarosevich/Projects/decrypto/data/raw_tweets_202208131844.csv')
df_test_tweets = pd.read_csv(path, low_memory=False)


#%%
analyzer = create_analyzer(task="sentiment", lang="en")

#%%
start = time()
for tweet in df_test_tweets['text'].tolist()[0:10]:
    print(tweet)
    print(analyzer.predict(tweet))
    print('##################################')

stop = time()
print(stop - start)

#%%
