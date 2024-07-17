import numpy as np

def warp_function(uv):
    # input: array of shape (n, 2) uv coordinates. (n,3) input will just ignore the z coordinate
    ret = []
    for i in uv:
        ret.append([i[0], i[1], i[0]*i[1]])

    return np.array(ret)