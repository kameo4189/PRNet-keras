'''
python setup.py build_ext -i
to compile
'''

# setup.py
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

setup(
	name = 'mesh_core_cython',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("mesh_render_cython",
                 sources=["mesh_render_core_cython.pyx", "mesh_render_core.cpp"],
                 language='c++',
                 include_dirs=[numpy.get_include()])],
)

