import numpy as np
import importlib
cythonModule = importlib.util.find_spec("util.cython")
setupModule = importlib.util.find_spec("util.cython.setup")
meshRenderCythonModule = importlib.util.find_spec("util.cython.mesh_render_cython")
if (cythonModule is not None and setupModule is not None 
    and meshRenderCythonModule is None):
    from subprocess import call
    call(['python', 'setup.py', "build_ext", "-i"],
        cwd=cythonModule.submodule_search_locations._path[0])

from util.cython import mesh_render_cython
def render_colors(vertices, triangles, colors, h, w, c = 3, BG = None):
    ''' render mesh with colors
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3] 
        colors: [nver, 3]
        h: height
        w: width  
        c: channel
        BG: background image
    Returns:
        image: [h, w, c]. rendered image./rendering.
    '''

    # initial 
    if BG is None:
        image = np.zeros((h, w, c), dtype = np.float32)
    else:
        assert BG.shape[0] == h and BG.shape[1] == w and BG.shape[2] == c
        image = BG
    depth_buffer = np.zeros([h, w], dtype = np.float32, order = 'C') - 999999.

    # change orders. --> C-contiguous order(column major)
    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    colors = colors.astype(np.float32).copy()
    ###
    mesh_render_cython.render_colors_core(
                image, vertices, triangles,
                colors,
                depth_buffer,
                vertices.shape[0], triangles.shape[0], 
                h, w, c)
    return image