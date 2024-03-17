import time
from functools import wraps

def timeit(function):

    @wraps(function)
    def wrapper(*args, **kwargs):
        tic = time.time()
        out = function(*args, **kwargs)
        toc = time.time()

        print(str(function.__name__) + ' Function call took %.2e s' % (toc-tic))
        return out

    return wrapper