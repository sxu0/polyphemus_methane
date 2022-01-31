"""
transect_interp_petrolia.py

Converts lat/lon anchor points to local coords
and interpolates points along transects.

Author: Shiqi Xu
"""

import os

import numpy as np
import pandas as pd

from libs import coord_convert


def import_anchors(filename):
    """Reads csv file with reference coordinates and saves data as pandas DataFrame.

    Args:
        filename (str): Path + filename, CSV format.

    Returns:
        df_anchors_lat_lon (DataFrame): DataFrame containing reference coordinates.
    """

    df_anchors_lat_lon = pd.read_csv(filename, index_col=0)

    return df_anchors_lat_lon


def convert_anchors_to_local_coords(df_anchors_lat_lon):
    """Converts lat/lon coordinates to local coordinate system, measured in metres
    from southwest corner.

    Args:
        df_anchors_lat_lon (DataFrame): DataFrame containing lat/lon coordinates.
        First set of coordinates are reference coordinates for conversion.

    Returns:
        df_anchors_local (DataFrame): DataFrame containing local coordinates,
        in metres.
    """

    cvrt_anchors_utm = df_anchors_lat_lon.apply(
        lambda row: coord_convert.convert_coord(row["lat"], row["lon"]), axis=1
    )
    df_anchors_utm = pd.DataFrame(
        cvrt_anchors_utm.tolist(), columns=["E", "N"], index=cvrt_anchors_utm.index
    )
    # print(df_anchors_utm)

    zero_anchor_utm = df_anchors_utm.loc["southwest corner", :]
    df_anchors_local = df_anchors_utm.sub(zero_anchor_utm, axis="columns")

    return df_anchors_local


def interp_transect_points(start_local, end_local):
    """Interpolates a series of coordinates along a line given start and end
    coordinates, in metres.

    Args:
        start_local (array-like): 1 by 2 array-like object containing x and y
            coordinates of starting point, in metres.
        end_local (array-like): 1 by 2 array-like object containing x and y
            coordinates of ending point, in metres.

    Returns:
        df_transect (DataFrame): DataFrame (101 by 2) containing x and y
            coordinates along the line, in metres.
    """

    transect_x = np.linspace(start_local[0], end_local[0], 101)
    transect_y = np.linspace(start_local[1], end_local[1], 101)

    df_transect = pd.DataFrame({"x": transect_x, "y": transect_y})

    return df_transect


if __name__ == "__main__":

    anchor_filename = os.path.join(
        os.path.dirname(__file__), "petrolia_landfill_reference_coords.csv"
    )
    anchors_lat_lon = import_anchors(anchor_filename)
    # print(anchors_lat_lon)

    anchors_local = convert_anchors_to_local_coords(anchors_lat_lon)
    print(anchors_local)

    transect_north = interp_transect_points(
        anchors_local.loc["northwest intersection", :],
        anchors_local.loc["northeast intersection", :],
    )
    # print(transect_north)
    transect_south = interp_transect_points(
        anchors_local.loc["southwest intersection", :],
        anchors_local.loc["southeast intersection", :],
    )
    # print(transect_south)
    transect_east = interp_transect_points(
        anchors_local.loc["southeast intersection", :],
        anchors_local.loc["northeast intersection", :],
    )
    # print(transect_east)
    transect_west = interp_transect_points(
        anchors_local.loc["southwest intersection", :],
        anchors_local.loc["northwest intersection", :],
    )
    # print(transect_west)
    transect_mid = interp_transect_points(
        anchors_local.loc["south-centre intersection", :],
        anchors_local.loc["north-centre intersection", :],
    )
    # print(transect_mid)

    path_save = os.path.join(os.path.dirname(__file__), "petrolia_transects")
    try:
        os.mkdir(path_save)
    except OSError:
        pass
    transect_north.to_csv(
        os.path.join(path_save, "petrolia_transect_north.csv"), index=False
    )
    transect_south.to_csv(
        os.path.join(path_save, "petrolia_transect_south.csv"), index=False
    )
    transect_east.to_csv(
        os.path.join(path_save, "petrolia_transect_east.csv"), index=False
    )
    transect_west.to_csv(
        os.path.join(path_save, "petrolia_transect_west.csv"), index=False
    )
    transect_mid.to_csv(
        os.path.join(path_save, "petrolia_transect_mid.csv"), index=False
    )
