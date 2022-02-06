"""
petrolia_predictions.py

Models predicted plumes for Petrolia landfill based on a range of inputs.

Author: Shiqi Xu
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import os

import numpy as np
import pandas as pd
from pathlib2 import Path

from libs import coord_convert
import gaussian_plume


if __name__ == "__main__":

    ## timestamp for distinguishing outputs
    time_run = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    ## source info
    src_info = pd.read_csv(
        Path.cwd() / "data" / "petrolia_landfill_source_coords.csv",
        index_col="source location"
    )
