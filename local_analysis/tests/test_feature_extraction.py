from scipy.stats import kurtosis
from pathlib import Path
import numpy as np
import pandas as pd

def test_tweet_feat_sanity_check():
    """
    Performs a few sentiment feature extractions manually on slices of the original tweet data, and compares it against
    the results from the actual feature extraction process. Note I ran this already and it passed, and is here mostly
    as a reference if future such tests need to be written.
    Returns
    -------
    None
    """

    # Get two randomly chosen hours and slice out the tweet data in the hour following these values.
    data_folder = Path('./data')
    path_btc_tweet_feats_merged = Path(data_folder / 'btc_tweet_feats_merged.pickle')
    df_bitcoin_tweet_feats_merged = pd.read_pickle(path_btc_tweet_feats_merged)
    tweet_path = Path(data_folder / 'tweets_and_sent.pickle')
    df_tweets = pd.read_pickle(tweet_path)
    rand_hour_1 = df_bitcoin_tweet_feats_merged['date'].iloc[1760]
    rand_hour_2 = df_bitcoin_tweet_feats_merged['date'].iloc[4431]
    df_tweet_slice_1 = df_tweets[ (df_tweets['date'] >= rand_hour_1) & (df_tweets['date'] <= rand_hour_1 + pd.Timedelta(minutes=59, seconds=59)) ]
    df_tweet_slice_2 = df_tweets[ (df_tweets['date'] >= rand_hour_2) & (df_tweets['date'] <= rand_hour_2 + pd.Timedelta(minutes=59, seconds=59)) ]

    # Collect the processed values for a few random sentiment extractions from different sentiment types.
    dct_processed_vals = {'rand_hour_1': {}, 'rand_hour_2': {}}
    for hour, idx in zip(['rand_hour_1', 'rand_hour_2'], [1760, 4431]):
        for val_type in ['mean_neg', 'median_pos', 'max_neu', 'min_neg', 'kurtosis_pos']:
            dct_processed_vals[hour][val_type] = df_bitcoin_tweet_feats_merged[val_type].iloc[idx]

    # Calculate the same values directly from the dataframe slices.
    dct_sanity_vals = {'rand_hour_1': {}, 'rand_hour_2': {}}
    for df_slice, hour in zip([df_tweet_slice_1, df_tweet_slice_2], ['rand_hour_1', 'rand_hour_2']):
        for fun, val_type, sent_type in zip([np.mean, np.median, np.max, np.min, kurtosis],
                                            ['mean_neg', 'median_pos', 'max_neu', 'min_neg', 'kurtosis_pos'],
                                            ['negative', 'positive', 'neutral', 'negative', 'positive']):
            dct_sanity_vals[hour][val_type] = fun(df_slice['score'][df_slice['label'] == sent_type].to_numpy())

    # Compare the two with a 1e-05 tolerance.
    fail_vals = []
    for key1 in dct_processed_vals.keys():
        for key2 in dct_processed_vals[key1].keys():
            processed_val = dct_processed_vals[key1][key2]
            sanity_val = dct_sanity_vals[key1][key2]
            if np.abs(sanity_val - processed_val) > 1e-5:
                fail_vals.append([key1, key2])
                print('For hour {} and value {} MAE was greater than 1e-05'.format(key1, key2))

    try:
        assert len(fail_vals) == 0
    except AssertionError as e:
        print('Sentiment sanity test failed for the following:')
        for vals in fail_vals:
            print(vals)
        raise e