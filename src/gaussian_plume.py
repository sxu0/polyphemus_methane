"""
gaussian_plume.py

Main functions for running Polyphemus' Gaussian plume model,
as well as for visualizing its outputs.

Authors: Shiqi Xu, Sebastien Ars
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import math
import os
import shutil

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import scipy
from scipy import integrate as integ
from scipy import stats
from pathlib2 import Path

from libs.polyphemus.include import atmopy
from libs.polyphemus.include.atmopy.display import *

from libs import model_sims, coord_convert


def run_model(
    sources,
    flux,
    window,
    stabil_class,
    wind_dir,
    wind_speed,
    pressure,
    temp,
    pbl,
    site,
    date,
):
    # type: (np.ndarray, np.ndarray, float, str, float, float, float, float, float, str, str) -> None
    """Runs the Gaussian plume model using the libs/model_sims.py module.

    Args:
        sources (np.ndarray[float]): 2D array containing information about
            sources. Each outer element corresponds to a source, whose elements
            entail the following:
                [0] x-coordinate in local coordinate system, in metres;
                [1] y-coordinate in local coordinate system, in metres;
                [2] source diameter, in metres;
                [3] source height above ground, in metres.
        flux (np.ndarray): 1D array containing flux rate of each source, in g/s.
        window (float): Width of square domain, in metres.
        stabil_class (str): Pasquill-Gifford stability class.
            One of 'A', 'B', 'C', 'D', 'E', 'F', and 'G'.
        wind_dir (float): Originating wind direction in degrees,
            measured clockwise from North.
        wind_speed (float): Wind speed, in m/s.
        pressure (float): Pressure, in hPa.
        temp (float): Temperature, in degrees Celsius.
        pbl (float): Planetary boundary layer height, in meters.
        site (str): Name of site being modelled, for populating paths.
        date (str): Date of measurements, in "YYYY-MM-DD" format.
    """
    ## 1_ create meteo file for Polyphemus
    model_sims.fichier_meteo(site, stabil_class, wind_dir, wind_speed, pressure, temp, pbl)
    ## 2_ run model
    model_sims.plume(sources, window, flux, temp, site, date)

    return


def calc_conversion_factor(pressure, temp):
    # type: (float, float) -> float
    """Calculates g/m^3 to ppb conversion factor given pressure and temperature.

    Notes:
    * methane molar mass = 16.04 g/mol
    * standard volume = 22.414 L/mol (at 0 deg C and 1013.25 hPa)
    * standard pressure = 1013.25 hPa

    Args:
        pressure (float): Pressure, in hPa.
        temp (float): Temperature, in degrees Celsius.

    Returns:
        conv_factor (float): g/m^3 to ppb conversion factor.
    """
    V_m = 22.414 * 1013.25 / pressure * (temp + 273.15) / 273.15
    conv_factor = V_m / 16.04

    return conv_factor


def plot_plume_horiz_slice(
    sources,
    transect_x,
    transect_y,
    window,
    conv_factor,
    site,
    date="",
    time_run="",
    save_fig=False,
    location_save="",
):
    # type: (np.ndarray, np.ndarray, np.ndarray, float, float, str, str, str, bool, str) -> None
    """Generates contour plot of concentrations across a horizontal slice of domain.

    Args:
        sources (np.ndarray): 2D array containing information about
            sources. Each outer element corresponds to a source, whose elements
            entail the following:
                [0] x-coordinate in local coordinate system, in metres;
                [1] y-coordinate in local coordinate system, in metres;
                [2] source diameter, in metres;
                [3] source height above ground, in metres.
        transect_x (np.ndarray): Local x-coordinates of points along transect.
        transect_y (np.ndarray): Local y-coordinates of points along transect.
        window (float): Width of square domain, in metres.
        conv_factor (float): g/m^3 to ppb conversion factor.
        site (str): Name of site being modelled, for populating paths.
        date (str, optional): Date of measurements, in "YYYY-MM-DD" format.
            Defaults to "".
        time_run (str, optional): Time of analysis, for distinguishing outputs.
            Defaults to "".
        save_fig (bool, optional): Whether to save output figure. Defaults to False.
        location_save (str, optional): Path to save output figure. Defaults to "".
    """
    ch4_bin_path = Path.cwd() / "polyphemus" / site / "graphes" / "config" / "results"
    d = getd(filename=ch4_bin_path / "Methane.bin", Nt=1, Nz=7, Ny=window, Nx=window)
    d *= conv_factor * 1e6  # conversion to ppb

    ## colour scheme by concentration
    ppb_levels = [
        0,
        0.01,
        0.05,
        0.10,
        0.5,
        1.0,
        5.0,
        10.0,
        15.0,
        25.0,
        50.0,
        100.0,
        500.0,
        1000.0,
        5000.0,
        10000.0,
    ]
    colour_scheme = [
        "aqua",
        "LightBlue",
        "SlateBlue",
        "blue",
        "darkblue",
        "forestgreen",
        "SeaGreen",
        "palegreen",
        "lemonchiffon",
        "yellow",
        "gold",
        "orange",
        "red",
        "darkred",
        "black",
    ]

    plt.figure()
    plt.contourf(d[0, 0], levels=ppb_levels, colors=colour_scheme)
    axis("equal")
    for i in range(len(sources)):
        plt.plot(sources[i][0], sources[i][1], "k-o")
    plt.plot(transect_x, transect_y, color="black")
    plt.xlim([0, window])
    plt.ylim([0, window])
    c = colorbar(ticks=ppb_levels)
    c.set_label("Methane Concentrations (ppb)")
    plt.title("Methane Plume Model")
    plt.xlabel("East-West Domain (m)")
    plt.ylabel("North-South Domain (m)")

    if save_fig:
        if time_run != "":
            time_run = time_run + "_"
        if date:
            date = "_" + date
        plt.savefig(os.path.join(location_save, time_run + site + date + ".png"))
        plt.close("all")
    else:
        plt.show()

    return


def plot_plume_transect(
    transect_x,
    transect_y,
    measured_ch4,
    window,
    conv_factor,
    site,
    date="",
    time_run="",
    save_fig=False,
    location_save="",
):
    # type: (np.ndarray, np.ndarray, np.ndarray, float, float, str, str, str, bool, str) -> None
    """Generates plot of modelled (and measured) concentrations along transect.

    Args:
        transect_x (np.ndarray): Local x-coordinates of points along transect.
        transect_y (np.ndarray): Local y-coordinates of points along transect.
        measured_ch4 (np.ndarray): Measured CH4 concentrations along
            transect, in ppm.
        window (float): Width of square domain, in metres.
        conv_factor (float): g/m^3 to ppb conversion factor.
        site (str): Name of site being modelled, for populating paths.
        date (str, optional): Date of measurements, in "YYYY-MM-DD" format.
            Defaults to "".
        time_run (str, optional): Time of analysis, for distinguishing outputs.
            Defaults to "".
        save_fig (bool, optional): Whether to save output figure. Defaults to False.
        location_save (str, optional): Path to save output figure. Defaults to "".
    """
    x_min, y_min = 0.0, 0.0
    delta_x, delta_y = 1.0, 1.0
    Nx, Ny = window, window
    Nz = 7
    Ns = 1

    root = Path.cwd() / "polyphemus" / site
    filename = root / "graphes" / "config" / "results" / "Methane.bin"
    d = np.fromfile(filename, dtype=np.float32)
    d = d.reshape(Ns, Nz, Ny, Nx)
    d *= conv_factor * 1e6  # conversion to ppb

    ## extract data from model output binary file
    plume_transect = []
    for ii in range(0, len(transect_x)):
        loc_x, loc_y = transect_x[ii], transect_y[ii]
        ind_y = int((loc_y - y_min) / delta_y)
        ind_x = int((loc_x - x_min) / delta_x)
        if (0 <= ind_y < Ny - 1) and (0 <= ind_x < Nx - 1):
            coeff_y = (loc_y - y_min - delta_y * ind_y) / delta_y
            coeff_x = (loc_x - x_min - delta_x * ind_x) / delta_x
            loc_conc = (
                (1.0 - coeff_y) * (1.0 - coeff_x) * d[:, :, ind_y, ind_x]
                + coeff_y * coeff_x * d[:, :, ind_y + 1, ind_x + 1]
                + coeff_y * (1.0 - coeff_x) * d[:, :, ind_y + 1, ind_x]
                + (1.0 - coeff_y) * coeff_x * d[:, :, ind_y, ind_x + 1]
            )
            plume_transect.append(loc_conc[0, 0])
            # print("\nloc_conc:\n", loc_conc)
    # print("\nplume_transect:\n", plume_transect)

    ## distance along transect
    dist_along_transect = [0]
    dist = 0
    for i in range(len(transect_x) - 1):
        dist_segment = (
            (transect_x[i + 1] - transect_x[i]) ** 2
            + (transect_y[i + 1] - transect_y[i]) ** 2
        ) ** 0.5
        dist += dist_segment
        dist_along_transect.append(dist)

    if measured_ch4 is not None:
        measured_ch4 *= 1000  # convert to ppb
        bkg = min(measured_ch4)
        measured_ch4 -= bkg  # subtract background concentration

    plt.figure()
    plt.plot(dist_along_transect, plume_transect, color="red", label="modelled")
    if measured_ch4 is not None:
        plt.plot(dist_along_transect, measured_ch4, color="blue", label="measured")
    plt.title("Plume Cross Section at Transect")
    plt.xlabel("Distance Along Transect (m)")
    plt.ylabel("Concentration (ppb)")
    if measured_ch4 is not None:
        plt.legend()

    if save_fig:
        if time_run != "":
            time_run = time_run + "_"
        if date:
            date = "_" + date
        plt.savefig(
            os.path.join(location_save, time_run + "transect_" + site + date + ".png")
        )
        plt.close("all")
    else:
        plt.show()

    return


# # ! transect coords converted in transect_mapping_germain_mills.py
# transect = pd.read_csv(transect_path, sep="\t", header=None, index_col=False)
# transect_x = transect.iloc[:, 0]
# transect_y = transect.iloc[:, 1]


# ## measured ch4
# # ! load in prepared (pre-cropped i think) file with measured ch4
# measured_ch4_file = pd.read_csv(measured_ch4_path, index_col=False)
# if date[:4] == "2018":
#     measured_ch4 = measured_ch4_file.loc[:, "ch4_cal"]
# else:
#     measured_ch4 = measured_ch4_file.loc[:, "ch4d"]
# # measured_ch4_file = pd.read_csv("/home/centos7/Documents/Polyphemus/german_mills_transects/german_mills_transect_2018-08-28_measured_ch4.txt", header=None, index_col=False)
# # measured_ch4 = measured_ch4_file.iloc[:, 0]
# measured_ch4 = measured_ch4.to_numpy()


# # ! calculate max height and "area" under transect concentrations
# #########################
# ## plot plume transect ##
# #########################

# print("maximum height:\t" + str(max(plume_transect)))
# print("area:\t\t" + str(model_transect_area))
