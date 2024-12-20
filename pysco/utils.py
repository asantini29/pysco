import sys, os
import glob
import warnings
import time
from datetime import timedelta
from functools import wraps
import GPUtil

def get_free_gpus(n_gpus=1):
    '''
    Get the IDs of free GPUs (with load less than 1% and memory less than 1%).

    Parameters
    ----------
    n_gpus : int
        Number of free GPUs to return.
    
    Returns
    -------
    free_gpus : list
        List of IDs of free GPUs.
    '''

    free_gpus = GPUtil.getAvailable(order='first', limit=n_gpus, maxLoad=0.01, maxMemory=0.01)
    return free_gpus

def find_files(dir, extension):
    '''
    Find all files in a directory with a given extension.

    Parameters
    ----------
    dir : str
        Directory to search for files.
    extension : str
        File extension to search for.
    
    Returns
    -------
    files : list
        List of files found in the directory with the given extension.
    '''

    files = glob.glob(os.path.join(dir, f"*.{extension}"))
    if not files:
        warnings.warn("No files found in the directory " + dir + " with extension " + extension) 
        
    return files

def timeit(function):
    '''
    Time the execution of a decorated function.
    '''
    @wraps(function)
    def wrapper(*args, **kwargs):
        tic = time.time()
        out = function(*args, **kwargs)
        toc = time.time()

        td = timedelta(seconds=toc-tic)
        print(str(function.__name__) + ' Function call took %.2e s' % (toc-tic) + ' (i.e. {})'.format(td))
        return out

    return wrapper