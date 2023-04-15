#!/usr/bin/env python3
"""Setup for adaptive_scheduler."""
# Copyright 2019 Bas Nijholt.
#
# This file is part of adaptive_scheduler. It is subject to the license terms
# in the file LICENSE found in the top-level directory of this distribution.
# A list of adaptive_scheduler authors can be found using git, with
# `git shortlog -s HEAD` and at
# https://github.com/basnijholt/adaptive-scheduler/graphs/contributors.


from setuptools import setup


def get_version_and_cmdclass(package_name: str):  # noqa: ANN201
    """Get the version and cmdclass using miniver."""
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(
        "version",
        os.path.join(package_name, "_version.py"),  # noqa: PTH118
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.cmdclass


version, cmdclass = get_version_and_cmdclass("adaptive_scheduler")

setup(
    version=version,
    cmdclass=cmdclass,
)
