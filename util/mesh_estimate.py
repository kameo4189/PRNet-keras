import numpy as np
from configure import config
from util import common_methods

def compute_similarity_transform(points_static, points_to_transform):
    #http://nghiaho.com/?page_id=671
    p0 = np.copy(points_static).T
    p1 = np.copy(points_to_transform).T

    t0 = -np.mean(p0, axis=1).reshape(3,1)
    t1 = -np.mean(p1, axis=1).reshape(3,1)
    t_final = t1 -t0

    p0c = p0+t0
    p1c = p1+t1

    covariance_matrix = p0c.dot(p1c.T)
    U,S,V = np.linalg.svd(covariance_matrix)
    R = U.dot(V)
    if np.linalg.det(R) < 0:
        R[:,2] *= -1

    rms_d0 = np.sqrt(np.mean(np.linalg.norm(p0c, axis=0)**2))
    rms_d1 = np.sqrt(np.mean(np.linalg.norm(p1c, axis=0)**2))

    s = (rms_d0/rms_d1)
    P = np.c_[s*np.eye(3).dot(R), t_final]
    return P

def estimate_pose(vertices):
    canonical_vertices = np.load(config.CANONICAL_VERTICES_PATH)
    P = compute_similarity_transform(vertices, canonical_vertices)
    _,R,_ = common_methods.P2sRt(P) # decompose affine matrix to s, R, t
    pose = common_methods.matrix2angle(R) 

    return P, pose