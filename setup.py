import setuptools
required_packages = [
    "lightgbm",
    "cloudml-hypertune",
    "numpy",
    "Cython"
]

setuptools.setup(
    name="lightgbm_moltr_all",
    author="Search Ranking",
    author_email="yfu@etsy.com",
    version="0.1.0",
    packages=setuptools.find_packages(),
    install_requires=required_packages,
    python_requires=">=3.6",
    include_package_data=True)






