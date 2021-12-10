import pptk
import open3d as o3d

def displayPointCloudColor(meshInfo, pointSize=1):
    v = pptk.viewer(meshInfo.vertices, meshInfo.colors)
    v.set(point_size=pointSize)

def displayPointCloudDepth(meshInfo, pointSize=1):
    v = pptk.viewer(meshInfo.vertices, meshInfo.vertices[:,2])
    v.set(point_size=pointSize)

def displayMeshPointCloud(meshInfos):
    pointClouds = []
    for meshInfo in meshInfos:
        pointCloud = o3d.geometry.PointCloud()
        pointCloud.points = o3d.utility.Vector3dVector(meshInfo.vertices)
        pointCloud.colors = o3d.utility.Vector3dVector(meshInfo.colors)
        pointClouds.append(pointCloud)  
    o3d.visualization.draw_geometries(pointClouds)

def displayMesh(meshInfo, width=640, height=480):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(meshInfo.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(meshInfo.triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(meshInfo.colors)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], width=width, height=height, mesh_show_back_face=True)