import numpy as np
from util import triangle_process, mesh_estimate
import cv2
from numba import njit

@njit
def render_colors(vertices, triangles, colors, h, w, c=3):
    ''' render mesh with colors
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3] 
        colors: [nver, 3]
        h: height
        w: width    
    Returns:
        image: [h, w, c]. 
    '''
    assert vertices.shape[0] == colors.shape[0]

    # initial 
    image = np.zeros((h, w, c))
    depth_buffer = np.zeros((h, w)) - 999999.

    for i in range(triangles.shape[0]):
        tri = triangles[i, :] # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(vertices[tri, 0]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(vertices[tri, 1]))), h-1)

        if umax<umin or vmax<vmin:
            continue
            
        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not triangle_process.isPointInTri(np.array([u,v]), vertices[tri, :2]): 
                    continue
                w0, w1, w2 = triangle_process.get_point_weight(np.array([u, v]), vertices[tri, :2])
                point_depth = w0*vertices[tri[0], 2] + w1*vertices[tri[1], 2] + w2*vertices[tri[2], 2]

                if point_depth > depth_buffer[v, u]:
                    depth_buffer[v, u] = point_depth
                    image[v, u, :] = w0*colors[tri[0], :] + w1*colors[tri[1], :] + w2*colors[tri[2], :]
    return image

def render_kps(image, vertices, kptIdxs, invertYAxis, point_size=2):
    kptImage = np.array(image * 255, np.uint8)

    kpts = vertices[kptIdxs].copy()
    if (invertYAxis):
        kpts[:, 1] = image.shape[0] - kpts[:, 1] - 1
    for i in range(kpts.shape[0]):
        cv2.circle(kptImage, (int(kpts[i,0]), int(kpts[i,1])), point_size, (0,255,0), -1)
    kptImage = kptImage / 255.
    return kptImage

def plot_vertices(image, meshInfo):
    image = image.copy()
    vertices = np.round(meshInfo.vertices).astype(np.int32)
    vertices[:, 1] = image.shape[0] - vertices[:, 1] - 1
    for i in range(0, vertices.shape[0], 2):
        st = vertices[i, :2]
        image = cv2.circle(image,(st[0], st[1]), 1, (255,0,0), -1)  
    return image

def plot_pose_box(image, meshInfo, color=(0, 0, 255), line_width=2):
    ''' Draw a 3D box as annotation of pose. Ref:https://github.com/yinguobing/head-pose-estimation/blob/master/pose_estimator.py
    Args: 
        image: the input image
        P: (3, 4). Affine Camera Matrix.
        kpt: (68, 3).
    '''
    pose, camera_matrix =  mesh_estimate.estimate_pose(meshInfo.vertices)
    kpt = meshInfo.kptPos
    kpt[:, 1] = image.shape[0] - kpt[:, 1] - 1

    image = image.copy()

    point_3d = []
    rear_size = 90
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 105
    front_depth = 110
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d image points
    point_3d_homo = np.hstack((point_3d, np.ones([point_3d.shape[0],1]))) #n x 4
    point_2d = point_3d_homo.dot(pose.T)[:,:2]
    point_2d[:,:2] = point_2d[:,:2] - np.mean(point_2d[:4,:2], 0) + np.mean(kpt[:27,:2], 0)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

    return image