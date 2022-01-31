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

from polyphemus.include import atmopy
from polyphemus.include.atmopy.display import *

from libs.model_functions import *
from libs.coord_convert import *


time_run = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

##########################
####      inputs      ####
##########################

site = "german_mills_settlers_park"
date = "2018-08-16"

fenetre = 6000  # window size (m)
nbsource = 1

xs1, ys1 = 3199, 3044  # source coords in window (m)
# xs1, ys1 = 1199, 1044
diameter1 = 0.001  # source diameter (m)
height1 = 2.5  # source height above ground (m)
flux1 = 7.5  # source flux (g/s)
# xs2, ys2 = 200, 300
# diameter2 = 0.001
# height2 = 1
# flux2 = 2

total_no_transects = 1  # total no. transects made at that location on that day
transect_no = 1  # transect being processed (__th on that day)
car_heading = "south"  # general qualitative direction

transect_path = "german_mills_transects/german_mills_transect_2018-08-16_mapped_around_sabiston_landfill.txt"
measured_ch4_path = "/home/centos7/Documents/Polyphemus/german_mills_transects/german_mills_transect_2018-08-16.csv"

pbl = 500  # boundary layer height (m)
sc = "C"  # stability class
press = 992.9  # pressure (hPa)
temp = 24.8  # temperature (C)
wd = 280 - 21  # wind direction (deg)
ws = 0.833  # wind speed (m/s)

wind_data_src = "Buttonville winds at end of hour"


##########################
##  source coordinates  ##
##########################

# see above for source inputs

sources, flux = [], []
sources = np.zeros((nbsource, 4))
flux = np.zeros((nbsource))

for ii in range(nbsource):
    sources[ii, 0] = locals()["xs{}".format(ii + 1)]
    sources[ii, 1] = locals()["ys{}".format(ii + 1)]
    sources[ii, 2] = locals()["diameter{}".format(ii + 1)]
    sources[ii, 3] = locals()["height{}".format(ii + 1)]
    flux[ii] = locals()["flux{}".format(ii + 1)]

# transect coords converted in transect_mapping_germain_mills.py
transect = pd.read_csv(transect_path, sep="\t", header=None, index_col=False)
transect_x = transect.iloc[:, 0]
transect_y = transect.iloc[:, 1]

# path='/home/sars/Data/' + site + '/' + date + '/'	# /!\ Change Directory


####################################
####		METEO DATA			####
####################################

# see above for meteo inputs

######## calculation of the conversion factor: g/m3 to ppb

Vm = 22.414 * 1013.25 / press * (temp + 273.15) / 273.15
print("Vm:", Vm)

facteur = Vm / 16.04  # Methane molar mass 16.04 g/mol
print("facteur:", facteur)


#################################
##   creation of directories   ##
#################################

# /!\ change directory of 'raccourci' and 'chemin_fichiers'
raccourci = "/home/centos7/Documents/Polyphemus/Polyphemus-1.8.1/" + site + "/"
path = raccourci + "preprocessing/"
chemin = path + "dep/"
path2 = raccourci + "processing/"
chemin2 = path2 + "gaussian/"
chemin_fichiers = (
    "/home/centos7/Documents/Polyphemus/Polyphemus-1.8.1/fichiers/preprocessing/dep/"
)
path3 = raccourci + "graphes/"
chemin3 = path3 + "config/"
chemin4 = chemin3 + "results/"


# 1- Create meteo file for Polyphemus
fichier_meteo(site, temp, wd, ws, pbl, sc, press)

# 2- Run the model
plume(site, sources, date, flux, temp, fenetre, facteur)
os.chdir(chemin4)


############################
##  Plot Map              ##
############################

### visualise les concentrations et ajoute colorbar
d = getd(filename="Methane.bin", Nt=1, Nz=7, Ny=fenetre, Nx=fenetre)
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
]  # Definition des niveaux
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

fig = plt.figure()
plt.contourf(d[0, 0], levels=v, colors=col)
axis("equal")
for ii in range(nbsource):
    plt.plot(
        (locals()["xs{}".format(ii + 1)]), (locals()["ys{}".format(ii + 1)]), "k-o"
    )
plt.plot(transect_x, transect_y, color="black")
plt.xlim([0, fenetre])
plt.ylim([0, fenetre])
c = colorbar(ticks=v)
c.set_label("Methane Concentrations (ppb)")
plt.title("Methane Plume Model: German Mills Settlers Park")
plt.xlabel("East-West Domain (m)")
plt.ylabel("North-South Domain (m)")
plt.savefig(time_run + "_german_mills_" + date + ".png")
plt.close("all")


######################
###    transect    ###
######################

x_min, y_min = 0.0, 0.0
Delta_x, Delta_y = 1.0, 1.0
Nx, Ny = fenetre, fenetre
Nz = 7
Ns = 1

filename = raccourci + "graphes/config/results/Methane.bin"
d = np.fromfile(filename, dtype=np.float32)
d = d.reshape(Ns, Nz, Ny, Nx)
d *= 1000000 * facteur  # conversion to ppb

plume_transect = []

# Extract data to
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


## flux scaling factor

# specific to 2021-08-09_2 and 2018-08-16 (want to include only main peak)
if date == "2021-08-09":
    dist_index = 0
    while dist_along_transect[dist_index] < 600:
        dist_index += 1
    model_transect_area = scipy.integrate.trapz(
        plume_transect[dist_index:], dist_along_transect[dist_index:]
    )
    measured_ch4_area = scipy.integrate.trapz(
        measured_ch4[dist_index:], dist_along_transect[dist_index:]
    )

elif date == "2018-08-16":
    dist_i_start = 0
    while dist_along_transect[dist_i_start] < 600:
        dist_i_start += 1
    dist_i_end = dist_i_start + 1
    while dist_along_transect[dist_i_end] < 1750:
        dist_i_end += 1
    model_transect_area = scipy.integrate.trapz(
        plume_transect[dist_i_start:dist_i_end],
        dist_along_transect[dist_i_start:dist_i_end],
    )
    measured_ch4_area = scipy.integrate.trapz(
        measured_ch4[dist_i_start:dist_i_end],
        dist_along_transect[dist_i_start:dist_i_end],
    )

else:
    model_transect_area = scipy.integrate.trapz(plume_transect, dist_along_transect)
    measured_ch4_area = scipy.integrate.trapz(measured_ch4, dist_along_transect)

flux_scaling_factor = measured_ch4_area / model_transect_area
print("flux scaling factor: " + str(flux_scaling_factor))


#########################
##    shift source?    ##
#########################

# index of model peak along transect
model_peak_i_start = plume_transect.index(max(plume_transect))
model_peak_i_stop = model_peak_i_start
while (model_peak_i_stop < len(plume_transect) - 1) and (
    plume_transect[model_peak_i_stop + 1] == plume_transect[model_peak_i_stop]
):
    model_peak_i_stop += 1
model_peak_index = int((model_peak_i_start + model_peak_i_stop) / 2)
# index of measured peak along transect
meas_peak_i_start = measured_ch4.index(max(measured_ch4))
meas_peak_i_stop = meas_peak_i_start
while (meas_peak_i_stop < len(measured_ch4) - 1) and (
    measured_ch4[meas_peak_i_stop + 1] == measured_ch4[meas_peak_i_stop]
):
    meas_peak_i_stop += 1
meas_peak_index = int((meas_peak_i_start + meas_peak_i_stop) / 2)

# difference in coordinates between measured & modelled peaks
src_delta_x = transect_x[meas_peak_index] - transect_x[model_peak_index]
src_delta_y = transect_y[meas_peak_index] - transect_y[model_peak_index]
print("src_delta_x:\t" + str(src_delta_x))
print("src_delta_y:\t" + str(src_delta_y))


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


##################################################################
##  find angular deviation between modelled and measured peaks  ##
##################################################################

# model_peak_index & meas_peak_index found in "shift source" section above

# modelled peak location along transect
model_peak_x_coord = transect_x[model_peak_index]
model_peak_y_coord = transect_y[model_peak_index]
# measured peak distance along transect
meas_peak_x_coord = transect_x[meas_peak_index]
meas_peak_y_coord = transect_y[meas_peak_index]

# modelled transect peak distance from source
model_peak_dist_from_src = (
    (model_peak_x_coord - xs1) ** 2 + (model_peak_y_coord - ys1) ** 2
) ** 0.5
# measured transect peak distance from source
meas_peak_dist_from_src = (
    (meas_peak_x_coord - xs1) ** 2 + (meas_peak_y_coord - ys1) ** 2
) ** 0.5
# distance between modelled peak and measured peak
dist_model_meas_peaks = (
    (model_peak_x_coord - meas_peak_x_coord) ** 2
    + (model_peak_y_coord - meas_peak_y_coord) ** 2
) ** 0.5

# cosine law: gamma = arccos( (a^2 + b^2 - c^2) / (2ab) )
angle_dev = round(
    math.degrees(
        math.acos(
            (
                model_peak_dist_from_src ** 2
                + meas_peak_dist_from_src ** 2
                - dist_model_meas_peaks ** 2
            )
            / (2 * model_peak_dist_from_src * meas_peak_dist_from_src)
        )
    ),
    2,
)
print("angular deviation btwn maxima:\t" + str(angle_dev) + "°")


###############################################################
##  correlation coeff between measured & modelled transects  ##
###############################################################

correlation_matrix = np.corrcoef(plume_transect, measured_ch4)
correlation_xy = correlation_matrix[0, 1]
R_sq = correlation_xy ** 2
# print(correlation_matrix)
print("R_squared = " + str(R_sq))


####################################
##  generate summary config file  ##
####################################

config_filename = time_run + "_cfg_" + site + "_" + date + ".txt"

config_file = open(config_filename, "a")  # append mode

config_file.write("INPUTS\n")
config_file.write("======\n\n")

config_file.write("site:\t\t\t\t" + site + "\n")
config_file.write("date:\t\t\t\t" + date + "\n")
config_file.write("suspected source:\t" + "former_sabiston_landfill" + "\n")
config_file.write("window size:\t\t" + str(fenetre) + "\n")

for i in range(nbsource):
    config_file.write("\nsource #" + str(i + 1) + ":\n")
    config_file.write("- - - - -\n")
    config_file.write(
        "source location:\t(" + str(sources[i, 0]) + ", " + str(sources[i, 1]) + ")\n"
    )
    config_file.write("source diameter:\t" + str(sources[i, 2]) + " m\n")
    config_file.write("flux:\t\t\t\t" + str(flux[i]) + " g/s\n")

config_file.write("\n---------------------------------------\n\n")

config_file.write("PBL:\t\t\t\t" + str(pbl) + " m\n")
config_file.write("stability class:\t" + sc + "\n")
config_file.write("pressure:\t\t\t" + str(press) + " hPa\n")
config_file.write("temperature:\t\t" + str(temp) + " °C\n")
config_file.write("wind direction:\t\t" + str(wd) + " °\n")
config_file.write("wind speed:\t\t\t" + str(ws) + " m/s\n")

config_file.write("\n---------------------------------------\n\n")

config_file.write("additional notes:\n")
if total_no_transects > 1:
    config_file.write(
        "- this is TRANSECT #"
        + str(transect_no)
        + " on "
        + date
        + ", there are "
        + str(total_no_transects)
        + " total\n"
    )
config_file.write(
    "- driving " + car_heading + " (distance travelled along transect is plotted)\n"
)
config_file.write("- used " + wind_data_src + "\n")
config_file.write(
    "- plotted measured ch4 (background removed) alongside model transect\n"
)
# config_file.write("- \n")
config_file.write("\n\n")

config_file.write("MODEL TRANSECT\n")
config_file.write("==============\n\n")

config_file.write("max height:\t\t\t" + str(max(plume_transect)) + "\n")
config_file.write(
    "area:\t\t\t\t"
    + str(scipy.integrate.trapz(plume_transect, dist_along_transect))
    + "\n"
)
config_file.write("\n\n")

config_file.write("CORRELATION\n")
config_file.write("===========\n\n")

config_file.write(
    "flux scaling factor (mod/meas):\t\t" + str(flux_scaling_factor) + "\n"
)
config_file.write("angular deviation btwn maxima:\t\t" + str(angle_dev) + " °\n")
config_file.write("R_squared:\t\t\t\t\t\t\t" + str(R_sq) + "\n")

config_file.write("\n")
config_file.close()
