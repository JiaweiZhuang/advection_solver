import setuptools

INSTALL_REQUIRES = [
    'numpy',
    'numba'
]

setuptools.setup(
    name='advection_solver',
    version='0.0.0',
    install_requires=INSTALL_REQUIRES,
    packages=setuptools.find_packages(),
    python_requires='>=3')
