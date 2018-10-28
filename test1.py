import numpy as np
from scipy import ndimage
def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (int(sx/fact), int(sy/fact))
    return res

if __name__ == '__main__':
    a = [1,2,33,5]
    b = [1,2,4,5]
    print(np.corrcoef(a,b))