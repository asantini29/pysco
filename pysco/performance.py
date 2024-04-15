import time
from datetime import timedelta
from functools import wraps

def timeit(function):

    @wraps(function)
    def wrapper(*args, **kwargs):
        tic = time.time()
        out = function(*args, **kwargs)
        toc = time.time()

        td = timedelta(seconds=toc-tic)
        print(str(function.__name__) + ' Function call took %.2e s' % (toc-tic))
        print('Time in hh:mm:ss.ms: {}'.format(td))
        return out

    return wrapper