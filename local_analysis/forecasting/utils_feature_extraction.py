import numpy as np
import multiprocessing as mp
import os
import pandas as pd
from scipy.stats import moment as scipy_moment
import local_analysis.forecasting.utils_multiprocess as utils_mp
from functools import partial as func_partial
from multiprocessing import shared_memory
from pathlib import Path
from pandas._libs.tslibs.timestamps import Timestamp as PdTimestamp
# data_folder = Path('../../data')
data_folder = Path('./data')
path_tokens = Path(data_folder / 'tweet_tokens.pickle')
path_sent = Path(data_folder / 'sent_val_column.pickle')
module_mat_tokens = pd.read_pickle(path_tokens)
module_sent_vals = pd.read_pickle(path_sent)
module_mat_tokens.flags.writeable = False
module_sent_vals.flags.writeable = False


def multiprocess_feature_extract_v2(dct_date_pair_idx_by_sent: dict, datetime_starts: list, date_pair_idx_all: list,
                                    settings: dict, flag_use_mp: bool, n = -1) -> dict:
    """
    Maps feature extraction results to a pair of timestamp ranges which correspond to a row of crypto data (an hour
    timespan). This v2 version uses module-level variables and write-on-copy pattern to multiprocess without copying.
    Note this has to be done once for the tokens, and once for each sentiment label.
    Parameters
    ----------
    date_pair_masks: list
        A list of dicts containing starting datetime objects and associated 1-hour masks for the tweet data.
    sent_scores: numpy.ndarray
        The tweet scores in a range [-1, 1].
    mat_tokens: numpy.ndarray
        The tweet tokens, restricted to M context.
    settings: dict
        The various settings for feature extraction. This is the interface for adding new feature types via the factory.

    Returns
    -------
    list
        The feature extraction results. List of dicts mapped to each date pair containing the feature information.
    """

    if os.name != 'posix':
        raise NotImplementedError('This function relies on Linux forks() for multiprocessing. Not intended for use on Windows.')
    
    ##### Extract sentiment features ######
    # TODO: Move this to a function.
    # Sentiment features are extracted once for each type, positive/negative/neutral. Each run is multiprocessed across
    # a list of indices that are the union of a date-pair mask and a sentiment mask. This is then used to slice
    # the sentiment array to get the right subset of sentiment values to extract features from.

    # Create a runner to pass multiple additional parameters into the map function as a tuple.
    args = settings
    runner = func_partial(_run_extract_sent_feats_v2, args=settings)

    dct_sent_features = {}
    for sent_label in dct_date_pair_idx_by_sent.keys():
        date_pair_data = dct_date_pair_idx_by_sent[sent_label]
        if flag_use_mp:
            num_cores = mp.cpu_count() - 2
            # Run multiprocess mapping with the sentiment scores and tokens (tokens is over 2 GB) across child processes
            # without copying. Note the try block is mainly here to absolutely ensure the shared memory is released, which
            # I've had issues with.
            try:
                with mp.Pool(processes=num_cores) as pool:
                    if n == -1:
                        date_pair_features = list(pool.starmap(runner, zip(date_pair_data, datetime_starts)))
                    else:
                        date_pair_features = list(pool.starmap(runner, zip(date_pair_data[0:n], datetime_starts[0:n])))
            except Exception as e:
                raise e
        else:
            try:
                if n == -1:
                    date_pair_features = list(map(runner, date_pair_data, datetime_starts))
                else:
                    date_pair_features = list(map(runner, date_pair_data[0:n], datetime_starts[0:n]))
            except Exception as e:
                raise e

        dct_sent_features[sent_label] = date_pair_features
        
    #### Extract token features #####
    # TODO: move to a function.
    # This section separately collects token features, since it has to be passed a list of date pair indices for all
    # hour slots in the crypto data regardless of sentiment label.
    runner = func_partial(_run_extract_token_feats_v2, args=settings)
    if flag_use_mp:
        num_cores = mp.cpu_count() - 2
        # Run multiprocess mapping with the sentiment scores and tokens (tokens is over 2 GB) across child processes
        # without copying. Note the try block is mainly here to absolutely ensure the shared memory is released, which
        # I've had issues with.
        try:
            with mp.Pool(processes=num_cores) as pool:
                if n == -1:
                    token_features = list(pool.starmap(runner, zip(date_pair_idx_all, datetime_starts)))
                else:
                    token_features = list(pool.starmap(runner, zip(date_pair_idx_all[0:n], datetime_starts[0:n])))
        except Exception as e:
            raise e
    else:
        try:
            if n == -1:
                token_features = list(map(runner, date_pair_idx_all, datetime_starts))
            else:
                token_features = list(map(runner, date_pair_idx_all[0:n], datetime_starts[0:n]))
        except Exception as e:
            raise e

    return {'token_feats': token_features, 'sent_feats': dct_sent_features}

def _run_extract_sent_feats_v2(idx_date_pair: np.ndarray, date_start: PdTimestamp, args: tuple) -> dict:
    """
    Map function for multiprocess feature extraction, using the module level variables and copy-on-copy pattern to
    multiprocess without copying.
    Parameters
    ----------
    date_pair_data: dict
        A dict with the human readable date start, and a tuple of the start/end in posix time.
    args
        Optional args. Just settings for this version.

    Returns
    -------
    dict
        The feature extraction dict for a given time pair.
    """

    curr_sent_vals = module_sent_vals[idx_date_pair]
    sent_feats = _extract_sent_feats(curr_sent_vals, args['sent_settings'])

    return {'date': date_start, 'sent_vals': curr_sent_vals, 'sent_feats': sent_feats}


def _run_extract_token_feats_v2(idx_date_pair: np.ndarray, date_start: PdTimestamp, args: dict) -> dict:

    curr_tokens = module_mat_tokens[idx_date_pair, :]
    token_feats = _extract_token_feats(curr_tokens, args['token_settings'])

    return {'date': date_start, 'token_vals': curr_tokens, 'token_feats': token_feats}


def multiprocess_feature_extract(date_pair_data: list, sent_scores: np.ndarray, mat_tokens: np.ndarray,
                                 settings: dict, flag_use_mp: bool) -> list:
    """
    Maps feature extraction results to a pair of timestamp ranges which correspond to a row of crypto data (an hour
    timespan).
    Parameters
    ----------
    date_pair_masks: list
        A list of dicts containing starting datetime objects and associated 1-hour masks for the tweet data.
    sent_scores: numpy.ndarray
        The tweet scores in a range [-1, 1].
    mat_tokens: numpy.ndarray
        The tweet tokens, restricted to M context.
    settings: dict
        The various settings for feature extraction. This is the interface for adding new feature types via the factory.

    Returns
    -------
    list
        The feature extraction results. List of dicts mapped to each date pair containing the feature information.
    """

    num_cores = mp.cpu_count() - 2

    # Create the shared memory addresses and critical type/shape information for multiprocessing.
    SENT_SCORES_SHARED_MEM_NAME = 'sent_scores_shared_mem'
    TOKENS_SHARED_MEM_NAME = 'tokens_shared_mem'
    shm_sent = utils_mp.create_shared_memory_nparray(sent_scores, SENT_SCORES_SHARED_MEM_NAME)
    shm_tokens = utils_mp.create_shared_memory_nparray(mat_tokens, TOKENS_SHARED_MEM_NAME)
    SENT_SHAPE = sent_scores.shape
    TOKENS_SHAPE = mat_tokens.shape
    sent_dtype = sent_scores.dtype
    tokens_dtype = mat_tokens.dtype

    # Create a runner to pass multiple additional parameters into the map function as a tuple.
    args = (SENT_SCORES_SHARED_MEM_NAME, TOKENS_SHARED_MEM_NAME, SENT_SHAPE, TOKENS_SHAPE,
            sent_dtype, tokens_dtype, settings)
    runner = func_partial(_run_extract_sent_and_token_feats, args)

    # Run multiprocess mapping with the sentiment scores and tokens (tokens is over 2 GB) across child processes
    # without copying. Note the try block is mainly here to absolutely ensure the shared memory is released, which
    # I've had issues with.
    try:
        with mp.Pool(processes=num_cores) as pool:
            date_pair_features = pool.map(runner, date_pair_data)
    except Exception as e:
        raise e
    finally:
        try:
            shm_sent.close()
            shm_tokens.close()
            shm_sent.unlink()
            shm_tokens.unlink()
        except Exception as e:
            print("Somehow, the shared memory failed to close.")
            raise e

    return date_pair_features

def _run_extract_sent_and_token_feats(date_pair_data: dict, args: tuple) -> dict:

    SHM_SENT_NAME, SHM_TOKEN_NAME, SENT_SHAPE, TOKENS_SHAPE, SENT_DTYPE, TOKEN_DTYPE, settings = args
    idx_date_pair = date_pair_data['idx_date_pair']
    date_start = date_pair_data['date_start']

    shm_sent = shared_memory.SharedMemory(name=SHM_SENT_NAME)
    shm_tokens = shared_memory.SharedMemory(name=SHM_TOKEN_NAME)
    vec_sent_vals = np.ndarray(SENT_SHAPE, dtype=SENT_DTYPE, buffer=shm_sent.buf)
    mat_tokens = np.ndarray(TOKENS_SHAPE, dtype=TOKEN_DTYPE, buffer=shm_tokens.buf)
    curr_sent_vals = vec_sent_vals[idx_date_pair]
    curr_tokens = mat_tokens[idx_date_pair, :]

    sent_feats = _extract_sent_feats(curr_sent_vals, settings['sent_settings'])
    token_feats = _extract_token_feats(curr_tokens, settings['token_settings'])

    return {'date': date_start, 'sent_vals': curr_sent_vals, 'sent_feats': sent_feats, 'token_vals': curr_tokens,
            'token_feats': token_feats}

def _extract_sent_feats(sent_vals: np.ndarray, settings: dict) -> dict:
    dct_return = {}
    for extract_type in settings['extract_types']:
        fun_extract = _feature_extract_factory(extract_type)
        if sent_vals.size != 0:
            dct_return[extract_type] = fun_extract(sent_vals)
        else:
            dct_return[extract_type] = None

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
        A dictionary of token features keyed as the n-most frequent, e.g. the most frequent is keyed by freq_token_1.
    """
    if tokens.size != 0:
        tokens = np.concatenate(tokens)
        unique, counts = np.unique(tokens, return_counts=True)
        idx = np.flip(np.argsort(counts))[0:settings['top_n']]
        top_tokens = unique[idx]
        stem = 'freq_token_{}'
        dict_return = {}
        for i, token in enumerate(top_tokens):
            dict_return[stem.format(i)] = token
    else:
        dict_return = {}

    return dict_return
