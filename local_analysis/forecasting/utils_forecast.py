import pandas as pd
import numpy as np
from datetime import timedelta
import multiprocessing as mp
from scipy.stats import moment as scipy_moment
import local_analysis.forecasting.utils_multiprocess as utils_mp
from functools import partial as func_partial
from multiprocessing import shared_memory


def multiprocess_feature_extract(date_pair_data: list, sent_scores: np.ndarray, mat_tokens: np.ndarray,
                                 settings: dict) -> dict:
    """
    Maps feature extraction results to a pair of timestamp ranges which correspond to a row of crypto data (an hour
    timespan).
    Parameters
    ----------
    date_pair_masks: list
        A list of dicts containing starting datetime objects and associated 1 hour masks for the tweet data.
    sent_scores: numpy.ndarray
        The tweet scores in a range [-1, 1].
    mat_tokens: numpy.ndarray
        The tweet tokens, restricted to M context.
    settings: dict
        The various settings for feature extraction. This is the interface for adding new feature types via the factory.

    Returns
    -------
    dict
        The feature extraction results. The dict also contains the start timestamp, which should be keyed to the crypto
        data.
    """
    num_cores = mp.cpu_count() - 2

    SENT_SCORES_SHARED_MEM_NAME = 'sent_scores_shared_mem'
    TOKENS_SHARED_MEM_NAME = 'tokens_shared_mem'
    shm_sent = utils_mp.create_shared_memory_nparray(sent_scores, SENT_SCORES_SHARED_MEM_NAME)
    shm_tokens = utils_mp.create_shared_memory_nparray(mat_tokens, TOKENS_SHARED_MEM_NAME)

    runner = func_partial(_run_extract_sent_and_token_feats, SENT_SCORES_SHARED_MEM_NAME, TOKENS_SHARED_MEM_NAME, settings)

    with mp.Pool(processes=num_cores) as pool:
        dct_sent_processed = pool.map(runner, date_pair_data)

    # If memory isn't release properly, it can cause a system crash.
    utils_mp.release_shared(SENT_SCORES_SHARED_MEM_NAME)
    utils_mp.release_shared(TOKENS_SHARED_MEM_NAME)

    return dct_sent_processed

def _run_extract_sent_and_token_feats(date_pair_data: dict, SHM_SENT_NAME: str, SHM_TOKEN_NAME: str,
                                      sent_array_shape: tuple, tokens_array_shape: tuple, settings: dict) -> dict:

    date_pair = date_pair_data['date_pair']
    date_start = date_pair_data['date_start']

    shm_sent = shared_memory.SharedMemory(name=SHM_SENT_NAME)
    shm_tokens = shared_memory.SharedMemory(name=SHM_TOKEN_NAME)
    vec_sent_vals = np.ndarray(sent_array_shape, dtype=np.float64, buffer=shm_sent.buf)
    mat_tokens = np.ndarray(tokens_array_shape, dtype=np.float64, buffer=shm_tokens.buf)

    sent_feats = _extract_sent_feats(vec_sent_vals, settings['single_val_sent'])
    token_feats = _extract_token_feats(mat_tokens, settings['tokens'])

    # TODO: This is placeholder, needs more consideration.
    sent_feats = _extract_sent_feats(vec_sent_vals, settings['sent_settings'])
    token_feats = _extract_token_feats(mat_tokens, settings['tokens'])

    return {'date': date_start[0], 'sent_vals': vec_sent_vals, 'sent_feats': sent_feats, 'token_feats': token_feats}

def _extract_sent_feats(sent_vals: np.ndarray, settings: dict) -> dict:
    dct_return = {}
    for extract_type in settings['extract_types']:
        fun_extract = _feature_extract_factory(extract_type)
        dct_return[extract_type] = fun_extract(sent_vals)

    return dct_return
def _feature_extract_factory(extraction_type):
    if extraction_type == 'mean':
        return np.mean
    if extraction_type == 'median':
        return np.median
    if extraction_type == 'max':
        return np.max
    if extraction_type == 'min':
        return np.min
    if extraction_type == 'moment':
        return _get_moments
    if extraction_type == 'std':
        return np.std
    else:
        raise ValueError(f'Extraction type {extraction_type} not supported')


def _get_moments(vals: np.ndarray) -> dict:
    dict_return = {}
    moments = ['exp_val', 'variance', 'skewness', 'kurtosis']
    for i, moment in enumerate(moments):
        dict_return[moment] = scipy_moment(vals, i)

    return dict_return

def _extract_token_feats(tokens: np.ndarray, settings: dict) -> dict:
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