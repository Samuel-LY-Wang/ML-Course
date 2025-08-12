import numpy as np

import numpy as np

def row_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (m,1) array where each entry is the average of a row
    """
    return np.array([np.mean(x, axis=1)]).T

def column_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (n,1) array where each entry is the average of a column
    """
    return np.array([np.mean(x, axis=0)]).T

def top_bottom_features(x):
    m,_=x.shape
    x_top=x[:m//2]
    x_bottom=x[m//2:]
    return np.array([[np.mean(x_top), np.mean(x_bottom)]]).T