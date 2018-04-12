from distutils.core import setup, Extension
from Cython.Build import cythonize


setup(ext_modules = cythonize(Extension(
    'max_substring',
    sources=['max_substring.pyx', 'maxsubstring.cpp'],
    language='c++',
    library_dirs=[],
    libraries=[],
    extra_compile_args=['-std=c++11'],
    extra_link_args=[]
)))