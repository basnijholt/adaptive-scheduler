#!/usr/bin/env python3

# Copyright 2019 Bas Nijholt authors.
#
# This file is part of adaptive.scheduler. It is subject to the license terms
# in the file LICENSE found in the top-level directory of this distribution.
# A list of adaptive.scheduler authors can be found using git, with
# `git shortlog -s HEAD` and at
# https://github.com/basnijholt/adaptive-scheduler/graphs/contributors.


from setuptools import setup, find_packages
import sys


if sys.version_info < (3, 6):
    print("adaptive requires Python 3.6 or above.")
    sys.exit(1)


def description(filename):
    """Provide a short description."""
    with open(filename) as fp:
        for lineno, line in enumerate(fp):
            if lineno < 3:
                continue
            line = line.strip()
            if len(line) > 0:
                return line


def get_version_and_cmdclass(package_name):
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(package_name, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.cmdclass


version, cmdclass = get_version_and_cmdclass("adaptive/scheduler")


setup(
    name="adaptive.scheduler",
    version=version,
    cmdclass=cmdclass,
    python_requires=">=3.6",
    namespace_packages=["adaptive"],
    packages=find_packages(),
    include_package_data=True,
    maintainer="Bas Nijholt",
    maintainer_email="bas@nijho.lt",
    description=description("README.md"),
    license="BSD-3",
    # url='https://adaptive-scheduler.readthedocs.io',
    # download_url='https://pypi.python.org/pypi/adaptive.scheduler',
    install_requires=["adaptive", "mpi4py", "pyzmq", "tinydb"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Distributed Computing",
    ],
)
