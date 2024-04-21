# Lifted directly from this fantastic writeup: https://luis-sena.medium.com/sharing-big-numpy-arrays-across-python-processes-abf0dc2a0ab2

import concurrent
import time
from multiprocessing import shared_memory
import numpy as np
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor

NUM_WORKERS = multiprocessing.cpu_count()
np.random.seed(42)
ARRAY_SIZE = int(2e8)
ARRAY_SHAPE = (ARRAY_SIZE,)
NP_SHARED_NAME = 'npshared'
NP_DATA_TYPE = np.float64
data = np.random.random(ARRAY_SIZE)


def create_shared_memory_nparray(input_data):
    # d_size = np.dtype(NP_DATA_TYPE).itemsize * np.prod(ARRAY_SHAPE)

    shm = shared_memory.SharedMemory(create=True, size=input_data.nbytes, name=NP_SHARED_NAME)
    # numpy array on shared memory buffer
    dst = np.ndarray(shape=input_data.shape, dtype=input_data.dtype, buffer=shm.buf)
    dst[:] = input_data[:]
    print(f'NP SIZE: {(dst.nbytes / 1024) / 1024}')
    return shm


def release_shared(name):
    shm = shared_memory.SharedMemory(name=name)
    shm.close()
    shm.unlink()  # Free and release the shared memory block


def get_date_mask(date_range: tuple, array_shape) -> np.ndarray:
    shm = shared_memory.SharedMemory(name=NP_SHARED_NAME)
    tweet_dates = np.ndarray(array_shape, dtype=NP_DATA_TYPE, buffer=shm.buf)
    print(tweet_dates[0])
    mask_output = np.where((tweet_dates >= date_range[0]) & (tweet_dates < date_range[1]))[0]
    return mask_output