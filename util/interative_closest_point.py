import numpy as np
from scipy import spatial
from datetime import datetime
from scipy import spatial, stats
import time

class PointCloud:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

        self.nx = None
        self.ny = None
        self.nz = None
        self.planarity = None

        self.no_points = len(x)
        self.sel = None

    def toX(self):
        return np.column_stack((self.x, self.y, self.z))

    def select_n_points(self, n):

        if self.no_points > n:
            self.sel = np.round(np.linspace(0, self.no_points-1, n)).astype(int)
        else:
            self.sel = np.arange(0, self.no_points, 1)

    def estimate_normals(self, neighbors):

        self.nx = np.full(self.no_points, np.nan)
        self.ny = np.full(self.no_points, np.nan)
        self.nz = np.full(self.no_points, np.nan)
        self.planarity = np.full(self.no_points, np.nan)

        kdtree = spatial.KDTree(np.column_stack((self.x, self.y, self.z)))
        query_points = np.column_stack((self.x[self.sel], self.y[self.sel], self.z[self.sel]))
        _, idxNN_all_qp = kdtree.query(query_points, k=neighbors, p=2, workers=-1)

        for (i, idxNN) in enumerate(idxNN_all_qp):
            selected_points = np.column_stack((self.x[idxNN], self.y[idxNN], self.z[idxNN]))
            C = np.cov(selected_points.T, bias=False)
            eig_vals, eig_vecs = np.linalg.eig(C)
            idx_sort = eig_vals.argsort()[::-1] # sort from large to small
            eig_vals = eig_vals[idx_sort]
            eig_vecs = eig_vecs[:,idx_sort]
            self.nx[self.sel[i]] = eig_vecs[0,2]
            self.ny[self.sel[i]] = eig_vecs[1,2]
            self.nz[self.sel[i]] = eig_vecs[2,2]
            self.planarity[self.sel[i]] = (eig_vals[1]-eig_vals[2])/eig_vals[0]

    def transform(self, H):

        XInE = np.column_stack((self.x, self.y, self.z))
        XInH = PointCloud.euler_coord_to_homogeneous_coord(XInE)
        XOutH = np.transpose(H @ XInH.T)
        XOut = PointCloud.homogeneous_coord_to_euler_coord(XOutH)

        self.x = XOut[:,0]
        self.y = XOut[:,1]
        self.z = XOut[:,2]

    @staticmethod
    def euler_coord_to_homogeneous_coord(XE):

        no_points = np.shape(XE)[0]
        XH = np.column_stack((XE, np.ones(no_points)))

        return XH

    @staticmethod
    def homogeneous_coord_to_euler_coord(XH):

        XE = np.column_stack((XH[:,0]/XH[:,3], XH[:,1]/XH[:,3], XH[:,2]/XH[:,3]))

        return XE

logging = False
previewing = False
def log(text):
    if logging:
        logtime = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print("[{}] {}".format(logtime, text))

def get_nearest_idxs(pcref, pcsel):
    kdtree = spatial.KDTree(np.column_stack((pcsel.x, pcsel.y, pcsel.z)))
    query_points = np.column_stack((pcref.x[pcref.sel], pcref.y[pcref.sel], pcref.z[pcref.sel]))
    _, sel = kdtree.query(query_points, k=1, p=2, workers=-1)
    return sel

def matching(pcfix, pcmov):
    kdtree = spatial.KDTree(np.column_stack((pcmov.x, pcmov.y, pcmov.z)))
    query_points = np.column_stack((pcfix.x[pcfix.sel], pcfix.y[pcfix.sel], pcfix.z[pcfix.sel]))
    _, pcmov.sel = kdtree.query(query_points, k=1, p=2, workers=-1)

    dx = pcmov.x[pcmov.sel] - pcfix.x[pcfix.sel]
    dy = pcmov.y[pcmov.sel] - pcfix.y[pcfix.sel]
    dz = pcmov.z[pcmov.sel] - pcfix.z[pcfix.sel]

    nx = pcfix.nx[pcfix.sel]
    ny = pcfix.ny[pcfix.sel]
    nz = pcfix.nz[pcfix.sel]

    no_correspondences = len(pcfix.sel)
    distances = np.empty(no_correspondences)
    for i in range(0, no_correspondences):
        distances[i] = dx[i]*nx[i] + dy[i]*ny[i] + dz[i]*nz[i]

    return distances

def reject(pcfix, pcmov, min_planarity, distances):

    planarity = pcfix.planarity[pcfix.sel]

    med = np.median(distances)
    sigmad = stats.median_abs_deviation(distances)

    keep_distance = [abs(d-med) <= 3*sigmad for d in distances]
    keep_planarity = [p > min_planarity for p in planarity]

    keep = keep_distance and keep_planarity

    pcfix.sel = pcfix.sel[keep]
    pcmov.sel = pcmov.sel[keep]
    distances = distances[keep]

    return distances

def estimate_rigid_body_transformation(x_fix, y_fix, z_fix, nx_fix, ny_fix, nz_fix,
                                       x_mov, y_mov, z_mov):

    A = np.column_stack((-z_mov*ny_fix + y_mov*nz_fix,
                          z_mov*nx_fix - x_mov*nz_fix,
                         -y_mov*nx_fix + x_mov*ny_fix,
                         nx_fix,
                         ny_fix,
                         nz_fix))

    l = nx_fix*(x_fix-x_mov) + ny_fix*(y_fix-y_mov) + nz_fix*(z_fix-z_mov)

    x, _, _, _ = np.linalg.lstsq(A, l)

    residuals = A @ x - l

    R = euler_angles_to_linearized_rotation_matrix(x[0], x[1], x[2])

    t = x[3:6]

    H = create_homogeneous_transformation_matrix(R, t)

    return H, residuals

def euler_angles_to_linearized_rotation_matrix(alpha1, alpha2, alpha3):

    dR = np.array([[      1, -alpha3,  alpha2],
                   [ alpha3,       1, -alpha1],
                   [-alpha2,  alpha1,       1]])

    return dR

def create_homogeneous_transformation_matrix(R, t):

    H = np.array([[R[0,0], R[0,1], R[0,2], t[0]],
                  [R[1,0], R[1,1], R[1,2], t[1]],
                  [R[2,0], R[2,1], R[2,2], t[2]],
                  [     0,      0,      0,    1]])

    return H

def check_convergence_criteria(distances_new, distances_old, min_change):

    def change(new, old):
        return np.abs((new-old)/old*100)

    change_of_mean = change(np.mean(distances_new), np.mean(distances_old))
    change_of_std = change(np.std(distances_new), np.std(distances_old))

    return True if change_of_mean < min_change and change_of_std < min_change else False

def simpleicp(X_fix, X_mov, correspondences=1000, neighbors=10, min_planarity=0.3, min_change=1,
              max_iterations=100):

    start_time = time.time()
    log("Create point cloud objects ...")
    pcfix = PointCloud(X_fix[:,0], X_fix[:,1], X_fix[:,2])
    pcmov = PointCloud(X_mov[:,0], X_mov[:,1], X_mov[:,2])

    log("Select points for correspondences in fixed point cloud ...")
    pcfix.select_n_points(correspondences)
    sel_orig = pcfix.sel

    log("Estimate normals of selected points ...")
    pcfix.estimate_normals(neighbors)

    H = np.eye(4)
    residual_distances = []
    X_movs = []

    log("Start iterations ...")
    for i in range(0, max_iterations):

        initial_distances = matching(pcfix, pcmov)

        # Todo Change initial_distances without return argument
        initial_distances = reject(pcfix, pcmov, min_planarity, initial_distances)

        dH, residuals = estimate_rigid_body_transformation(
            pcfix.x[pcfix.sel], pcfix.y[pcfix.sel], pcfix.z[pcfix.sel],
            pcfix.nx[pcfix.sel], pcfix.ny[pcfix.sel], pcfix.nz[pcfix.sel],
            pcmov.x[pcmov.sel], pcmov.y[pcmov.sel], pcmov.z[pcmov.sel])

        residual_distances.append(residuals)

        pcmov.transform(dH)

        X_movs.append(pcmov.toX())

        H = dH @ H
        pcfix.sel = sel_orig

        if i > 0:
            if check_convergence_criteria(residual_distances[i], residual_distances[i-1],
                                          min_change):
                log("Convergence criteria fulfilled -> stop iteration!")
                break

        if i == 0:
            log("{:9s} | {:15s} | {:15s} | {:15s}".format("Iteration", "correspondences",
                                                          "mean(residuals)", "std(residuals)"))
            log("{:9d} | {:15d} | {:15.4f} | {:15.4f}".format(0, len(initial_distances),
                                                              np.mean(initial_distances),
                                                              np.std(initial_distances)))
        log("{:9d} | {:15d} | {:15.4f} | {:15.4f}".format(i+1, len(residual_distances[i]),
                                                          np.mean(residual_distances[i]),
                                                          np.std(residual_distances[i])))

    log("Estimated transformation matrix H:")
    log("H = [{:12.6f} {:12.6f} {:12.6f} {:12.6f}]".format(H[0,0], H[0,1], H[0,2], H[0,3]))
    log("    [{:12.6f} {:12.6f} {:12.6f} {:12.6f}]".format(H[1,0], H[1,1], H[1,2], H[1,3]))
    log("    [{:12.6f} {:12.6f} {:12.6f} {:12.6f}]".format(H[2,0], H[2,1], H[2,2], H[2,3]))
    log("    [{:12.6f} {:12.6f} {:12.6f} {:12.6f}]".format(H[3,0], H[3,1], H[3,2], H[3,3]))
    log("Finished in {:.3f} seconds!".format(time.time()-start_time))

    if previewing:
        import open3d as o3d
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # geometry is the point cloud used in your animaiton
        target = o3d.geometry.PointCloud()
        source = o3d.geometry.PointCloud()
        color_fix = np.zeros((X_fix.shape[0], 3))
        color_fix[:] = [0,0,1]

        target.points = o3d.utility.Vector3dVector(X_fix)
        target.colors = o3d.utility.Vector3dVector(color_fix)

        color_mov = np.zeros((X_fix.shape[0], 3))
        color_mov[:] = [0,1,0]
        source.points = o3d.utility.Vector3dVector(X_fix)
        source.colors = o3d.utility.Vector3dVector(color_mov)   
        
        vis.add_geometry(source)
        vis.add_geometry(target)

        import time as t
        try:
            while True:
                for i, X in enumerate(X_movs):
                    # now modify the points of your geometry
                    # you can use whatever method suits you best, this is just an example
                    source.points = o3d.utility.Vector3dVector(X)
                    vis.update_geometry(source)
                    vis.poll_events()
                    vis.update_renderer()
                    t.sleep(0.25)
                    # vis.capture_screen_image(r"test/temp_%04d.jpg" % i)
        except KeyboardInterrupt:
            pass

    pcmov_t = PointCloud(X_mov[:,0], X_mov[:,1], X_mov[:,2])
    pcmov_t.transform(H)

    pcmov_t.select_n_points(pcmov_t.no_points)
    nearest_idxs = get_nearest_idxs(pcmov_t, pcfix)

    return H, pcmov_t.toX(), nearest_idxs

if __name__ == "__main__":
    import csv
    from mesh_info import MeshInfo
    import mesh_display

    def read_xyz(path_to_pc):
        X = []
        with open(path_to_pc) as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                X.append(list(map(float, row)))
        return X

    path_to_pc1 = 'data/icp/dragon1.xyz'
    path_to_pc2 = 'data/icp/dragon2.xyz'

    # path_to_pc1 = 'data/icp/airborne_lidar1.xyz'
    # path_to_pc2 = 'data/icp/airborne_lidar2.xyz'

    # path_to_pc1 = 'data/icp/terrestrial_lidar1.xyz'
    # path_to_pc2 = 'data/icp/terrestrial_lidar2.xyz'

    X_fix = np.array(read_xyz(path_to_pc1))
    X_mov = np.array(read_xyz(path_to_pc2))
    X_mov[:, 0] = X_mov[:, 0] + 14
    X_mov[:, 1] = X_mov[:, 1] + 15
    X_mov[:, 2] = X_mov[:, 2] + 16

    print(path_to_pc1)
    print(path_to_pc2)

    logging = True
    previewing = False
    (_, X_mov_t, _) = simpleicp(X_fix, X_mov) 

    color_fix = np.zeros((X_fix.shape[0], 3))
    color_mov_t = np.zeros((X_mov_t.shape[0], 3))
    color_mov = np.zeros((X_mov.shape[0], 3))
    color_fix[:] = [0,0,1]
    color_mov[:] = [1,1,0]
    color_mov_t[:] = [0,1,0]
    meshFix = MeshInfo(X_fix, color_fix)
    meshMove = MeshInfo(X_mov, color_mov)
    meshMoveT = MeshInfo(X_mov_t, color_mov_t)
    mesh_display.displayMeshPointCloud([meshFix, meshMove, meshMoveT])