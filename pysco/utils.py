import sys, os
import glob
import warnings

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