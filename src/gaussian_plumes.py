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

from polyphemus.include import atmopy
from polyphemus.include.atmopy.display import *

from libs import model_sims, coord_convert

def run_model(
    site: str,
    date: str,
    sources: np.ndarray,
    flux: np.ndarray,
    window: float,
    pbl: float,
    stabil_class: str,
    pressure: float,
    temp: float,
    wind_dir: float,
    wind_speed: float,
):
    # xs, ys, d, h

    ## 1_ create meteo file for Polyphemus
    model_sims.fichier_meteo(site, temp, wind_dir, wind_speed, pbl, stabil_class, pressure)
    ## 2_ run model
    model_sims.plume(site, sources, date, flux, temp, window)

    ## directories created
    root = Path.cwd() / "polyphemus" / site
    chemin1 = root / "preprocessing" / "dep"
    chemin2 = root / "processing" / "gaussian"
    chemin_fichiers = Path.cwd() / "polyphemus" / "fichiers" / "preprocessing" / "dep"
    chemin3 = root / "graphes" / "config"
    chemin4 = chemin3 / "results"

    return chemin4


def calc_conversion_factor(pressure, temp):
    ## g/m3 to ppb
    ## methane molar mass = 16.04 g/mol
    ## 22.414 L/mol (at 0 deg C and 1013.25 hPa)
    ## 1013.25 hPa (standard pressure)

    V_m = 22.414 * 1013.25 / pressure * (temp + 273.15) / 273.15
    facteur = V_m / 16.04

    return facteur


# ! transect coords converted in transect_mapping_germain_mills.py
# transect = pd.read_csv(transect_path, sep="\t", header=None, index_col=False)
# transect_x = transect.iloc[:, 0]
# transect_y = transect.iloc[:, 1]

def plume_birdseye_plot(
    sources,
    window,
    transect_x,
    transect_y,
    facteur,
    site,
    date=None,
    time_run=None,
    save_fig=False,
):
    d = getd(filename="Methane.bin", Nt=1, Nz=7, Ny=window, Nx=window)
    v = [
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
    ]  # definition des niveaux
    col = [
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
    d *= 1000000 * facteur  # conversion en ppb

    plt.figure()
    plt.contourf(d[0, 0], levels=v, colors=col)
    axis("equal")
    for i in range(len(sources)):
        plt.plot(sources[i][0], sources[i][1], "k-o")
    plt.plot(transect_x, transect_y, color="black")
    plt.xlim([0, window])
    plt.ylim([0, window])
    c = colorbar(ticks=v)
    c.set_label("Methane Concentrations (ppb)")
    plt.title("Methane Plume Model: German Mills Settlers Park")
    plt.xlabel("East-West Domain (m)")
    plt.ylabel("North-South Domain (m)")

    if save_fig:
        if time_run:
            time_run_filename = time_run + "_"
        else:
            time_run_filename = ""
        if date:
            date_filename = "_" + date
        else:
            date_filename = ""
        plt.savefig(time_run_filename + site + date_filename + ".png")
        plt.close("all")
    else:
        plt.show()

def plume_transect_plot(
    window,
    transect_x,
    transect_y,
    facteur,
    site,
):




######################
###    transect    ###
######################

x_min, y_min = 0.0, 0.0
Delta_x, Delta_y = 1.0, 1.0
Nx, Ny = window, window
Nz = 7
Ns = 1

root = Path.cwd() / "polyphemus" / site
filename = root / "graphes" / "config" / "results" / "Methane.bin"
d = np.fromfile(filename, dtype=np.float32)
d = d.reshape(Ns, Nz, Ny, Nx)
d *= 1000000 * facteur  # conversion to ppb

plume_transect = []

# extract data to
for ii in range(0, len(transect_x)):
    loc_x, loc_y = transect_x[ii], transect_y[ii]
    ind_y = int((loc_y - y_min) / Delta_y)
    ind_x = int((loc_x - x_min) / Delta_x)

    if (0 <= ind_y < Ny - 1) and (0 <= ind_x < Nx - 1):

        coeff_y = (loc_y - y_min - Delta_y * ind_y) / Delta_y
        coeff_x = (loc_x - x_min - Delta_x * ind_x) / Delta_x

        loc_conc = (
            (1.0 - coeff_y) * (1.0 - coeff_x) * d[:, :, ind_y, ind_x]
            + coeff_y * coeff_x * d[:, :, ind_y + 1, ind_x + 1]
            + coeff_y * (1.0 - coeff_x) * d[:, :, ind_y + 1, ind_x]
            + (1.0 - coeff_y) * coeff_x * d[:, :, ind_y, ind_x + 1]
        )

        plume_transect.append(loc_conc[0, 0])

        # print("\nloc_conc:")
        # print(loc_conc)

# print("\nplume_transect:")
# print(plume_transect)


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


## measured ch4
measured_ch4_file = pd.read_csv(measured_ch4_path, index_col=False)
if date[:4] == "2018":
    measured_ch4 = measured_ch4_file.loc[:, "ch4_cal"]
else:
    measured_ch4 = measured_ch4_file.loc[:, "ch4d"]
# measured_ch4_file = pd.read_csv("/home/centos7/Documents/Polyphemus/german_mills_transects/german_mills_transect_2018-08-28_measured_ch4.txt", header=None, index_col=False)
# measured_ch4 = measured_ch4_file.iloc[:, 0]
measured_ch4 *= 1000  # convert to ppb
bkg = min(measured_ch4)
measured_ch4 -= bkg  # subtract background concentration
measured_ch4 = measured_ch4.values.tolist()


#########################
## plot plume transect ##
#########################

fig = plt.figure()
plt.plot(dist_along_transect, plume_transect, color="red", label="modelled")
plt.plot(dist_along_transect, measured_ch4, color="blue", label="measured")
plt.savefig(time_run + "_transect_german_mills_" + date + ".png")
plt.title("Plume Cross Section at Transect")
plt.xlabel("Distance Along Transect (m)")
plt.ylabel("Concentration (ppb)")
plt.legend()
plt.close("all")


print("maximum height:\t" + str(max(plume_transect)))
print("area:\t\t" + str(model_transect_area))
