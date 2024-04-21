import pandas as pd
import numpy as np
from datetime import timedelta
import multiprocessing as mp
from scipy.stats import moment as scipy_moment

def par_sent_feature_extract(all_dates: np.ndarray, df_tweets: pd.DataFrame, settings: dict) -> dict:
    # So basically this needs to take in a tuple of dates from a list of such tuples, and the tweet data frame,
    # mask out the relevant tweet rows, and then do some processing on them, we would then return a dict with the results
    # and the starting timestamp (in order to sync up with the coin frame).

    # The dates in 59m59s blocks.
    all_dates = list(zip(all_dates, all_dates + timedelta(minutes=59, seconds=59)))

    num_cores = mp.cpu_count() - 2
    # TODO: Note that I have no idea how the views to dataframe slices will be handled by multiproc. At best, I assume
    #       it would have to pass in the entire dataframe? Gathering slices has the same problem, so maybe it's still
    #       best to just pass in the entire dataframe as a copy? Though, collecting the slices in a list would be 2*n
    #       memory, whereas passing in the entire dataframe would be m*n where m = processes?
    with mp.Pool(processes=num_cores) as pool:
        dct_sent_processed = pool.map(lambda item: run_extract_sent_and_token_feats(item, df_tweets, settings), all_dates)

    return dct_sent_processed

def run_extract_sent_and_token_feats(date_range: dict, df_tweets: pd.DataFrame, settings: dict) -> dict:

    mask_dates = np.where((df_tweets['date'] >= date_range[0]) & (df_tweets['date'] <= date_range[1]))
    sent_vals = df_tweets['single_val_sent'][mask_dates]
    tokens = df_tweets['tokens'][mask_dates]

    sent_feats = extract_sent_feats(sent_vals, settings['single_val_sent'])
    token_feats = extract_token_feats(tokens, settings['tokens'])

    return {'date': date_range[0], 'sent_vals': sent_vals, 'sent_feats': sent_feats, 'token_feats': token_feats}

def extract_sent_feats(sent_vals: np.ndarray, settings: dict) -> dict:
    dct_return = {}
    for extract_type in settings['extract_types']:
        fun_extract = feature_extract_factory(extract_type)
        dct_return[extract_type] = fun_extract(sent_vals)

    return dct_return
def feature_extract_factory(extraction_type):
    if extraction_type == 'mean':
        return np.mean
    if extraction_type == 'median':
        return np.median
    if extraction_type == 'max':
        return np.max
    if extraction_type == 'min':
        return np.min
    if extraction_type == 'moment':
        return get_moments
    if extraction_type == 'std':
        return np.std
    else:
        raise ValueError(f'Extraction type {extraction_type} not supported')


def get_moments(vals: np.ndarray) -> dict:
    dict_return = {}
    moments = ['exp_val', 'variance', 'skewness', 'kurtosis']
    for i, moment in enumerate(moments):
        dict_return[moment] = scipy_moment(vals, i)

    return dict_return

def extract_token_feats(tokens: np.ndarray, settings: dict) -> dict:
    """
    Extract token features from tokens. Currently just returns the top N most common tokens.
    Parameters
    ----------
    tokens: numpy.ndarray
    settings: dict

    Returns
    -------
    dict
        A dictionary of token features keyed by their names.
    """
    tokens = np.concatenate(tokens)
    unique, counts = np.unique(tokens, return_counts=True)
    idx = np.flip(np.argsort(counts))[0:settings['top_n']]
    top_tokens = unique[idx]
    stem = 'freq_token_{}'
    dict_return = {}
    for i, token in enumerate(top_tokens):
        dict_return[stem.format(i)] = token

    return dict_return