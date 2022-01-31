#!/usr/bin/env python


import datetime
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import colors
# from matplotlib import pyplot as plt
# import pandas as pd
# import scipy
# from scipy import integrate as integ
# from scipy import stats

from polyphemus.include import atmopy
from polyphemus.include.atmopy.display import *

from coord_convert import *


def convert_to_time(x):
    """Converts gps_time strings into datetime objects."""
    return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")


def fichier_meteo(site, temp, wd, ws, pbl, SC, press):
    # directory creation
    raccourci = (
        "/home/centos7/Documents/Polyphemus/Polyphemus-1.8.1/" + site + "/"
    )  # chemin dossier
    path = raccourci + "preprocessing/"
    chemin = path + "dep/"
    path2 = raccourci + "processing/"
    chemin2 = path2 + "gaussian/"
    chemin_fichiers = "/home/centos7/Documents/Polyphemus/Polyphemus-1.8.1/fichiers/preprocessing/dep/"
    path3 = raccourci + "graphes/"
    chemin3 = path3 + "config/"
    chemin4 = chemin3 + "results/"

    try:
        os.makedirs(chemin)  # creation des dossiers inexistants du chemin
    except OSError:
        pass

    try:
        os.makedirs(chemin2)
    except OSError:
        pass

    try:
        os.makedirs(chemin3)
    except OSError:
        pass

    try:
        os.makedirs(chemin4)
    except OSError:
        pass

    #######################################################
    ### copie des fichiers non modifies dans les dossiers #
    #######################################################

    pathfile = chemin_fichiers + "gaussian-deposition.cfg"  # chemin du fichier a copier
    shutil.copy(pathfile, chemin)  # collage du fichier copie
    shutil.copy(pathfile, chemin3)
    pathfile = chemin_fichiers + "gaussian-deposition.cpp"
    shutil.copy(pathfile, chemin)
    shutil.copy(pathfile, chemin3)
    pathfile = chemin_fichiers + "gaussian-species.dat"
    shutil.copy(pathfile, chemin)
    shutil.copy(pathfile, chemin3)
    pathfile = chemin_fichiers + "SConstruct"
    shutil.copy(pathfile, chemin)

    #####################################################
    # conversion des directions du vent pour polyphemus #
    #####################################################
    #
    wd = 450 - wd - 180  #
    #
    #####################################################

    ###################################
    ### creation du fichier meteo.dat #
    ###################################

    nom = open("%s/meteo.dat" % (chemin), "w")
    nom.write(
        "[situation]\n\n# Temperature (Celsius degrees)\nTemperature = %s\n\n# Wind angle (degrees)\nWind_angle = %s\n\n# Wind speed (m/s)\nWind = %s\n\n# Boundary height (m)\n\
        Boundary_height =  %s\n\n# Stability class\nStability = %s\n\n# Rainfall rate (mm/hr)\nRainfall_rate = 0.\n\n# Pressure (Pa)\nPressure = %s\n\n"
        % (temp, wd, ws, pbl, SC, press)
    )
    nom.close()

    pathfile = (
        chemin + "meteo.dat"
    )  # copie/colle le fichier meteo que l on vient de creer
    shutil.copy(pathfile, chemin3)

    #############################################################################
    ### Compiler gaussian-deposition  dans le bon dossier pour le preprocessing #
    #############################################################################

    os.chdir(chemin)
    os.system(
        "/home/centos7/Documents/Polyphemus/Polyphemus-1.8.1/utils/scons.py gaussian-deposition"
    )

    #############################################
    ### lancer l'executable gaussian-deposition #
    #############################################

    os.chdir(chemin3)
    os.system(
        raccourci + "preprocessing/dep/gaussian-deposition gaussian-deposition.cfg"
    )

    pathfile = (
        chemin + "meteo.dat"
    )  # copie/colle le fichier meteo que l on vient de creer
    shutil.copy(pathfile, chemin3)

    return


def plume_response_function(site, source, date, rate, temp, window, facteur):

    raccourci = (
        "/home/centos7/Documents/Polyphemus/Polyphemus-1.8.1/" + site + "/"
    )  # chemin dossier
    path = raccourci + "preprocessing/"
    chemin = path + "dep/"
    path2 = raccourci + "processing/"
    chemin2 = path2 + "gaussian/"
    chemin_fichiers = "/home/centos7/Documents/Polyphemus/Polyphemus-1.8.1/fichiers/preprocessing/dep/"
    chemin_fichiers2 = "/home/centos7/Documents/Polyphemus/Polyphemus-1.8.1/fichiers/processing/gaussian/"
    path3 = raccourci + "graphes/"
    chemin3 = path3 + "config/"
    chemin4 = chemin3 + "results/"

    #####################################
    # creation fichier plume-source.dat #
    #####################################

    nom = open("%s/plume-source.dat" % (chemin2), "w")
    nom.write(
        "[source]\n\n# Source coordinates (meters)\nAbscissa: %s\nOrdinate: %s\nAltitude: %s\n\n# Species name\nSpecies: Methane\n\nType: continuous\nDate_beg: %s\nDate_end: %s\n\n\
        # Source rate (mass/s)\nRate: %s\n# Source velocity (m/s)\nVelocity = 0.\n\n# Source temperature (Celsius degrees)\nTemperature = %s\n\n#Source diameter (m)\nDiameter = %s\n\n"
        % (source[0], source[1], source[3], date, date, rate, temp, source[2])
    )
    nom.close()

    pathfile = (
        chemin2 + "plume-source.dat"
    )  # copie/colle le fichier meteo que l on vient de creer
    shutil.copy(pathfile, chemin3)

    #####################################
    # creation fichier plume.cfg 	    #
    #####################################

    nom = open("%s/plume.cfg" % (chemin2), "w")
    nom.write(
        '[display]\n\n\
        Show_iterations: yes\nShow_meteorological_data: yes\nShow_date: yes\n\n\n\
        \
        [domain]\n\n\
        ## Domain where species concentrations are computed.\n\
        Date_min: %s	Delta_t = 1.0	Nt = 1\n\
        x_min = 0.0		Delta_x = 1.	Nx = %s\n\
        y_min = 0.0		Delta_y = 1.	Ny = %s\n\n\
        Nz = 7\nVertical_levels: plume-levels.dat\n\n# Land category: rural or urban.\nLand_category: rural\n\n# Time of the day: night or day.\nTime: day\n\n\
        # File containing the species data.\nSpecies: gaussian-species.dat\n\n\n\
        \
        [gaussian]\n\n\
        With_plume_rise: no\nWith_plume_rise_breakup: no\nWith_radioactive_decay: no\nWith_biological_decay: no\nWith_scavenging: no\nWith_dry_deposition: no\n\n\
        # Parameterization to compute standard deviations: "Briggs", "Doury" or\n# "similarity_theory".\n\
        Sigma_parameterization: Briggs\n\n# Is there a particular formula for standard deviation above the boundary layer?\
        # If "Gillani" is provided, the vertical sigma is computed with this formula. Otherwise, the formula is the same above and below the boundary layer.\n\
        Above_BL: none\n\n# Alternative parameterization. Useful only when using similarity theory. It is recommended to use it for elevated sources (about 200 m).\n\
        With_HPDM: no\n\n# Plume rise parameterization: put "HPDM", "Concawe" or "Holland".\n\
        Plume_rise_parameterization: HPDM\n\n# File containing the meteorological data.\nFile_meteo:  gaussian-meteo.dat\n\n# File containing the source data.\nFile_source:  plume-source.dat\n\n\
        # File containing the correction coefficients (used with line sources only).\nFile_correction: correction_coefficients.dat\n\n\n\
        \
        [deposition]\n\n\
        # Deposition model: "Chamberlain" or "Overcamp".\nDeposition_model: Overcamp\n\n# Number of points to compute the Chamberlain integral.\nNchamberlain: 100\n\n\n\
        \
        [uncertainty]\n\n\
        File_perturbation: perturbation.cfg\nNumber_samples = 10\n# Newran seed directory (with "/" at the end), or "current_time" for random\n\
        # seeds generated with current time, or a given seed number (in ]0, 1[).\nRandom_seed = 0.5\n\n\n\
        \
        [output]\n\n\
        \
        # File describing which concentrations are saved.\nConfiguration_file:  plume-saver.cfg'
        % (date, window, window)
    )
    nom.close()

    pathfile = (
        chemin2 + "plume.cfg"
    )  # copie/colle le fichier meteo que l on vient de creer
    shutil.copy(pathfile, chemin3)

    ###################################################################
    ### copie des fichiers qui ne sont pas modifies dans les dossiers #
    ###################################################################

    pathfile = chemin_fichiers2 + "plume.cpp"
    shutil.copy(pathfile, chemin2)
    shutil.copy(pathfile, chemin3)
    pathfile = chemin_fichiers2 + "plume-levels.dat"
    shutil.copy(pathfile, chemin2)
    shutil.copy(pathfile, chemin3)
    pathfile = chemin_fichiers2 + "SConstruct"
    shutil.copy(pathfile, chemin2)
    pathfile = chemin_fichiers2 + "plume-saver.cfg"
    shutil.copy(pathfile, chemin2)
    shutil.copy(pathfile, chemin3)
    pathfile = chemin3 + "gaussian-meteo.dat"
    shutil.copy(pathfile, chemin2)

    ###########################################################
    ### Compiler plume dans le bon dossier pour le processing #
    ###########################################################

    os.chdir(chemin2)
    os.system(
        "/home/centos7/Documents/Polyphemus/Polyphemus-1.8.1/utils/scons.py plume"
    )

    ###############################
    ### lancer l'executable plume #
    ###############################

    os.chdir(chemin3)
    os.system(raccourci + "processing/gaussian/plume plume.cfg")

    return


def window(lat_s, lon_s, lat, lon, ws, wd, last):
    x_ext1, y_ext1 = convert_coord(lat[0], lon[0])
    x_ext2, y_ext2 = convert_coord(lat[last], lon[last])
    x_s, y_s = convert_coord(lat_s, lon_s)
    xmin, ymin, xmax, ymax = (
        min(x_ext1, x_ext2, x_s),
        min(y_ext1, y_ext2, y_s),
        max(x_ext1, x_ext2, x_s),
        max(y_ext1, y_ext2, y_s),
    )

    ############ size of the window
    r = 6371e3  # earth's radius (m)
    lat_s, lat, lon_s, lon = map(np.radians, [lat_s, lat, lon_s, lon])

    ch_lat = lat[0] - lat_s
    ch_lon = lon[0] - lon_s
    a = (np.sin(ch_lat / 2) ** 2) + (
        np.cos(lat_s) * np.cos(lat[0]) * np.sin(ch_lon / 2) ** 2
    )
    c = 2 * math.atan2(np.sqrt(a), np.sqrt(1 - a))  # the angular distance in radian
    d_s_ext1 = r * c

    ch_lat = lat[last] - lat_s
    ch_lon = lon[last] - lon_s
    a = (np.sin(ch_lat / 2) ** 2) + (
        np.cos(lat_s) * np.cos(lat[0]) * np.sin(ch_lon / 2) ** 2
    )
    c = 2 * math.atan2(np.sqrt(a), np.sqrt(1 - a))  # the angular distance in radian
    d_s_ext2 = r * c

    ch_lat = lat[last] - lat[0]
    ch_lon = lon[last] - lon[0]
    a = (np.sin(ch_lat / 2) ** 2) + (
        np.cos(lat[0]) * np.cos(lat[last]) * np.sin(ch_lon / 2) ** 2
    )
    c = 2 * math.atan2(np.sqrt(a), np.sqrt(1 - a))  # the angular distance in radian
    d_ext1_ext2 = r * c

    window = max(d_s_ext1, d_s_ext2, d_ext1_ext2)
    window -= window % -100  # round window up to next multiple of 100
    window = window + 200
    print("Window:", window)

    ######### origin of the window

    u, v = (-ws * math.sin(math.radians(wd))), (-ws * math.cos(math.radians(wd)))

    if u > 0 and v > 0:
        # x0, y0 = xmin - 200 , ymin - 200
        x0, y0 = xmin - 200, ymin - 200
    if u < 0 and v < 0:
        # x0, y0 = xmax - window + 200, ymax - window + 200
        x0, y0 = xmax - window + 200, ymax - window + 200
    if u > 0 and v < 0:
        # x0, y0 = xmin - 200, ymax - window + 200
        x0, y0 = xmin - 200, ymax - window + 200
    if u < 0 and v > 0:
        # x0, y0 = xmax - window + 200, ymin - 200
        x0, y0 = xmax - window + 200, ymin - 200

    return window, x0, y0


def plume(site, sources, date, rate, temp, window, facteur):

    raccourci = (
        "/home/centos7/Documents/Polyphemus/Polyphemus-1.8.1/" + site + "/"
    )  # chemin dossier
    path = raccourci + "preprocessing/"
    chemin = path + "dep/"
    path2 = raccourci + "processing/"
    chemin2 = path2 + "gaussian/"
    chemin_fichiers = "/home/centos7/Documents/Polyphemus/Polyphemus-1.8.1/fichiers/preprocessing/dep/"
    chemin_fichiers2 = "/home/centos7/Documents/Polyphemus/Polyphemus-1.8.1/fichiers/processing/gaussian/"
    path3 = raccourci + "graphes/"
    chemin3 = path3 + "config/"
    chemin4 = chemin3 + "results/"

    #####################################
    # creation fichier plume-source.dat #
    #####################################

    nom = open("%s/plume-source.dat" % (chemin2), "w")
    for ii in range(0, len(sources)):
        nom.write(
            "[source]\n\n# Source coordinates (meters)\nAbscissa: %s\nOrdinate: %s\nAltitude: %s\n\n# Species name\nSpecies: Methane\n\nType: continuous\nDate_beg: %s\nDate_end: %s\n\n\
            # Source rate (mass/s)\nRate: %s\n# Source velocity (m/s)\nVelocity = 0.\n\n# Source temperature (Celsius degrees)\nTemperature = %s\n\n#Source diameter (m)\nDiameter = %s\n\n"
            % (
                sources[ii, 0],
                sources[ii, 1],
                sources[ii, 3],
                date,
                date,
                float(rate[ii]),
                temp,
                sources[ii, 2],
            )
        )
    nom.close()

    pathfile = (
        chemin2 + "plume-source.dat"
    )  # copie/colle le fichier meteo que l on vient de creer
    shutil.copy(pathfile, chemin3)

    #####################################
    # creation fichier plume.cfg 	    #
    #####################################

    nom = open("%s/plume.cfg" % (chemin2), "w")
    nom.write(
        '[display]\n\n\
        Show_iterations: yes\nShow_meteorological_data: yes\nShow_date: yes\n\n\n\
        \
        [domain]\n\n\
        ## Domain where species concentrations are computed.\n\
        Date_min: %s	Delta_t = 1.0	Nt = 1\n\
        x_min = 0.0		Delta_x = 1.	Nx = %s\n\
        y_min = 0.0		Delta_y = 1.	Ny = %s\n\n\
        Nz = 7\nVertical_levels: plume-levels.dat\n\n# Land category: rural or urban.\nLand_category: rural\n\n# Time of the day: night or day.\nTime: day\n\n\
        # File containing the species data.\nSpecies: gaussian-species.dat\n\n\n\
        \
        [gaussian]\n\n\
        With_plume_rise: no\nWith_plume_rise_breakup: no\nWith_radioactive_decay: no\nWith_biological_decay: no\nWith_scavenging: no\nWith_dry_deposition: no\n\n\
        # Parameterization to compute standard deviations: "Briggs", "Doury" or\n# "similarity_theory".\nSigma_parameterization: Briggs\n\n\
        # Is there a particular formula for standard deviation above the boundary layer?\
        # If "Gillani" is provided, the vertical sigma is computed with this formula. Otherwise, the formula is the same above and below the boundary layer.\n\
        Above_BL: none\n\n# Alternative parameterization. Useful only when using similarity theory. It is recommended to use it for elevated sources (about 200 m).\n\
        With_HPDM: no\n\n# Plume rise parameterization: put "HPDM", "Concawe" or "Holland".\nPlume_rise_parameterization: HPDM\n\n# File containing the meteorological data.\nFile_meteo:  gaussian-meteo.dat\n\n\
        # File containing the source data.\nFile_source:  plume-source.dat\n\n# File containing the correction coefficients (used with line sources only).\nFile_correction: correction_coefficients.dat\n\n\n\
        \
        [deposition]\n\n\
        # Deposition model: "Chamberlain" or "Overcamp".\nDeposition_model: Overcamp\n\n# Number of points to compute the Chamberlain integral.\nNchamberlain: 100\n\n\n\
        \
        [uncertainty]\n\n\
        File_perturbation: perturbation.cfg\nNumber_samples = 10\n# Newran seed directory (with "/" at the end), or "current_time" for random\n\
        # seeds generated with current time, or a given seed number (in ]0, 1[).\nRandom_seed = 0.5\n\n\n\
        [output]\n\n\
        \
        # File describing which concentrations are saved.\nConfiguration_file:  plume-saver.cfg'
        % (date, window, window)
    )
    nom.close()

    pathfile = (
        chemin2 + "plume.cfg"
    )  # copie/colle le fichier meteo que l on vient de creer
    shutil.copy(pathfile, chemin3)

    ###################################################################
    ### copie des fichiers qui ne sont pas modifies dans les dossiers #
    ###################################################################

    pathfile = chemin_fichiers2 + "plume.cpp"
    shutil.copy(pathfile, chemin2)
    shutil.copy(pathfile, chemin3)
    pathfile = chemin_fichiers2 + "plume-levels.dat"
    shutil.copy(pathfile, chemin2)
    shutil.copy(pathfile, chemin3)
    pathfile = chemin_fichiers2 + "SConstruct"
    shutil.copy(pathfile, chemin2)
    pathfile = chemin_fichiers2 + "plume-saver.cfg"
    shutil.copy(pathfile, chemin2)
    shutil.copy(pathfile, chemin3)
    pathfile = chemin3 + "gaussian-meteo.dat"
    shutil.copy(pathfile, chemin2)

    ###########################################################
    ### Compiler plume dans le bon dossier pour le processing #
    ###########################################################

    os.chdir(chemin2)
    os.system(
        "/home/centos7/Documents/Polyphemus/Polyphemus-1.8.1/utils/scons.py plume"
    )

    ###############################
    ### lancer l'executable plume #
    ###############################

    os.chdir(chemin3)
    os.system(raccourci + "processing/gaussian/plume plume.cfg")

    ####################################
    ### Tracer graphiques sur la route #
    ####################################

    return
