import scipy.io as sio
import numpy as np

def load_uv_coords(path = 'BFM_UV.mat'):
    ''' load uv coords of BFM
    Args:
        path: path to data.
    Returns:  
        uv_coords: [nver, 2]. range: 0-1
    '''
    C = sio.loadmat(path)
    uv_coords = C['UV'].copy(order = 'C')
    return uv_coords

def process_uv(uv_coords, uv_h, uv_w):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords