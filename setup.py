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
from versioningit import get_cmdclasses

setup(cmdclass=get_cmdclasses())
