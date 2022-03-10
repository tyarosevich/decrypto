from pysentimiento import create_analyzer

#%%
analyzer = create_analyzer(task="sentiment", lang="en")
test_tweet = df_final.iloc[0,].loc['text']

#%%
print(analyzer.predict(test_tweet))