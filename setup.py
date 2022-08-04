"""The setup script."""
from setuptools import setup, find_packages

import oceantide


with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "click",
    "dask",
    "gcsfs",
    "netCDF4",
    "numpy",
    "scipy",
    "xarray",
    "zarr",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest",
]

setup(
    author=oceantide.__author__,
    author_email=oceantide.__contact__,
    description=oceantide.__description__,
    keywords=oceantide.__keywords__,
    version=oceantide.__version__,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    entry_points={
        "console_scripts": [
            "oceantide=oceantide.cli:main",
        ],
    },
    install_requires=requirements,
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    name="oceantide",
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://gitlab.com/oceanum/oceantide",
    zip_safe=False,
)
