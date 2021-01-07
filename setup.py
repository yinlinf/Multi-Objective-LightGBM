from setuptools import dist
dist.Distribution().fetch_build_eggs(['Cython>=0.15.1','numpy'])


from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize, build_ext
import numpy

required_packages = [
    "setuptools>=18.0",
    "lightgbm",
    "cloudml-hypertune",
    "numpy",
    "pandas",
    "cython"
]

setup(
    name="lightgbm_ai_platform",
    author="Search Ranking",
    author_email="search-ranking-team@etsy.com",
    version="0.1.0",
    packages=find_packages(),
    install_requires=required_packages,
    python_requires=">=3.6",
    include_package_data=True,
    ext_modules = cythonize(Extension('lambdaobj',
                                      sources=['./moltr/lambdaobj.pyx','./moltr/argsort.cpp'],
                                      language="c++",
                                      include_dirs=[numpy.get_include()])),
    cmdclass = {'build_ext': build_ext},
)

