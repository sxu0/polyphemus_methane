# pylint: disable=E1136, E1137, E0611, E0401

"""
transect_mapping_german_mills.py

Author: Shiqi Xu

Converts latitude/longitude to local coordinate system in meters.
"""


import os

import pandas as pd

from libs.coord_convert import convert_coord


## german mills settlers park, 2018-08-28 drive

DOMAIN_SIZE = 3000  # square

## suspected_source: Leslie Pumping Station
suspected_source = {"lat": 43.81258, "lon": -79.37350}
suspected_source["E"], suspected_source["N"] = convert_coord(
    suspected_source["lat"], suspected_source["lon"]
)
# print(suspected_source)

transect_coords = pd.read_csv(
    os.path.join(
        os.path.dirname(__file__),
        "german_mills_transects",
        "german_mills_transect_2018-08-28.txt",
    ),
    sep="\t",
    header=None,
    index_col=False,
)
transect_coords.rename(columns={0: "lat", 1: "lon"}, inplace=True)
# print(transect_coords)

transect_E, transect_N = [], []
for i in range(len(transect_coords.index)):
    E_i, N_i = convert_coord(
        transect_coords.loc[i, "lat"], transect_coords.loc[i, "lon"]
    )
    transect_E.append(E_i)
    transect_N.append(N_i)
transect_coords["E"] = transect_E
transect_coords["N"] = transect_N
# print(transect_coords)

# transect_E_range = max(transect_E) - min(transect_E)
# transect_N_range = max(transect_N) - min(transect_N)
# print("transect E min:\t" + str(min(transect_E)))
# print("transect N min:\t" + str(min(transect_N)))
# print("transect E range:\t" + str(transect_E_range))
# print("transect N range:\t" + str(transect_N_range))

domain = {}
domain["E_min"] = suspected_source["E"] - DOMAIN_SIZE / 2
domain["N_min"] = suspected_source["N"] - DOMAIN_SIZE / 2
transect_x = transect_coords["E"] - domain["E_min"]
transect_y = transect_coords["N"] - domain["N_min"]
transect_coords["x"] = transect_x
transect_coords["y"] = transect_y
# print(transect_coords)

transect_xy = pd.DataFrame({"transect_x": transect_x, "transect_y": transect_y})
# print(transect_xy)
transect_xy.to_csv(
    os.path.join(
        os.path.dirname(__file__),
        "german_mills_transects",
        "german_mills_transect_mapped_2018-08-28.txt",
    ),
    sep="\t",
    header=False,
    index=False,
)
