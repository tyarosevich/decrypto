from multiprocessing import shared_memory
import numpy as np
np.random.seed(42)

# With 100,000 rows.
n = 100000
test_matrix = np.random.randn(n, 64)
d_size = np.dtype(test_matrix.dtype).itemsize * np.prod(test_matrix.shape)
shm = shared_memory.SharedMemory(create=True, size=d_size, name='test_shared_matrix')
# numpy array on shared memory buffer
dst = np.ndarray(shape=test_matrix.shape, dtype=test_matrix.dtype, buffer=shm.buf)
dst[:] = test_matrix[:]

shm.close()
shm.unlink()

# 7: SIGBUS failure With 1,000,000 Rows
n = 1000000
test_matrix = np.random.randn(n, 64)
d_size = np.dtype(test_matrix.dtype).itemsize * np.prod(test_matrix.shape)
shm = shared_memory.SharedMemory(create=True, size=d_size, name='test_shared_matrix')
# numpy array on shared memory buffer
dst = np.ndarray(shape=test_matrix.shape, dtype=test_matrix.dtype, buffer=shm.buf)
dst[:] = test_matrix[:]

shm.close()
shm.unlink()

print("I finished")