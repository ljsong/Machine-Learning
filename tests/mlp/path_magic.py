#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys


base_path = os.path.dirname(os.path.realpath('..')).split(os.sep)
module_path = os.sep.join(base_path + ['src/mlp'])
sys.path.append(module_path)
module_path = os.sep.join(base_path + ['src/cnn'])
sys.path.append(module_path)
