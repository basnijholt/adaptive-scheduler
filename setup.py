#!/usr/bin/env python3

# Copyright 2019 Bas Nijholt.
#
# This file is part of adaptive_scheduler. It is subject to the license terms
# in the file LICENSE found in the top-level directory of this distribution.
# A list of adaptive_scheduler authors can be found using git, with
# `git shortlog -s HEAD` and at
# https://github.com/basnijholt/adaptive-scheduler/graphs/contributors.


import sys

from setuptools import find_packages, setup

if sys.version_info < (3, 8):
    print("adaptive-scheduler requires Python 3.8 or above.")
    sys.exit(1)


def get_version_and_cmdclass(package_name):
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(package_name, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.cmdclass


version, cmdclass = get_version_and_cmdclass("adaptive_scheduler")

with open("requirements.txt") as f:
    requirements = f.read().split()

with open("README.rst") as f:
    readme = f.read()

setup(
    name="adaptive_scheduler",
    version=version,
    cmdclass=cmdclass,
    python_requires=">=3.8",
    packages=find_packages(),
    include_package_data=True,
    maintainer="Bas Nijholt",
    maintainer_email="bas@nijho.lt",
    description="Run many `adaptive.Learner`s on many cores (>10k) using `mpi4py.futures`, `ipyparallel`, `dask-mpi`, or `process-pool`.",
    long_description=readme,
    long_description_content_type="text/x-rst",
    license="BSD-3",
    url="https://github.com/basnijholt/adaptive-scheduler",
    download_url="https://pypi.python.org/pypi/adaptive_scheduler",
    install_requires=requirements,
    extras_require={"all": ["dask_mpi"]},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Distributed Computing",
    ],
)
