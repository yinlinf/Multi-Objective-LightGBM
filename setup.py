import setuptools
# from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize, build_ext

required_packages = [
    "lightgbm",
    "cloudml-hypertune",
    "numpy",
    "moltr"
]

exts = [Extension(name="lambdaobj",
                  sources=["moltr/lambdaobj.pyx"],
                  libraries=["argsort"],
                  library_dirs=["."],
                  extra_compile_args=["-fopenmp", "-stdlib=libc++"],
                  extra_link_args=["-lomp", "-L/usr/local/opt/libomp/lib/", "-stdlib=libc++"]
                  )]

setuptools.setup(
    name="lightgbm_moltr",
    author="Search Ranking",
    author_email="yfu@etsy.com",
    version="0.1.0",
    packages=setuptools.find_packages(),
    install_requires=required_packages,
    python_requires=">=3.6",
    include_package_data=True,
    ext_modules=cythonize(exts)
)






