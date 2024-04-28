from local_analysis.forecasting.utils_feature_extraction import do_i_see_the_var
from time import time

start = time()
do_i_see_the_var()
end = time()
print('Time taken to do i-see-the-var: {}'.format(end - start))