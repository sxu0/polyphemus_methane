#!/usr/bin/env python


import datetime
import os
import shutil
from pathlib2 import Path

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


def convert_to_time(x: str) -> datetime.datetime:
    """Converts gps_time strings into datetime objects."""
    return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")


def fichier_meteo(
    site_name: str,
    stabil_class: str,
    wind_dir: float,
    wind_speed: float,
    pressure: float,
    temp: float,
    pbl: float,
) -> None:
    """Generates `meteo.dat` file containing relevant meteorological data. Executes
    the preprocessing step of the gaussian plume model.

    Args:
        site_name (str): Name of site being modelled, for populating
            paths and filenames.
        stabil_class (str): Pasquill-Gifford stability class.
            One of 'A', 'B', 'C', 'D', 'E', 'F', and 'G'.
        wind_dir (float): Originating wind direction in degrees,
            measured clockwise from North.
        wind_speed (float): Wind speed, in m/s.
        pressure (float): Pressure, in hPa.
        temp (float): Temperature, in degrees Celsius.
        pbl (float): Planetary boundary layer height, in meters.
    """
    model_path = Path.cwd() / "polyphemus" / site_name
    chemin1 = model_path / "preprocessing" / "dep"
    chemin2 = model_path / "processing" / "gaussian"
    chemin_fichiers = Path.cwd() / "polyphemus" / "fichiers" / "preprocessing" / "dep"
    chemin3 = model_path / "graphes" / "config"
    chemin4 = chemin3 / "results"

    try:
        os.makedirs(chemin1)
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

    path_file = chemin_fichiers / "gaussian-deposition.cfg"
    shutil.copy(path_file, chemin1)
    shutil.copy(path_file, chemin3)
    path_file = chemin_fichiers / "gaussian-deposition.cpp"
    shutil.copy(path_file, chemin1)
    shutil.copy(path_file, chemin3)
    path_file = chemin_fichiers / "gaussian-species.dat"
    shutil.copy(path_file, chemin1)
    shutil.copy(path_file, chemin3)
    path_file = chemin_fichiers / "SConstruct"
    shutil.copy(path_file, chemin1)

    #####################################################
    # conversion des directions du vent pour polyphemus #
    #####################################################

    wind_dir = 450 - wind_dir - 180

    ###################################
    ### creation du fichier meteo.dat #
    ###################################

    nom = open(chemin1 / "meteo.dat", "w")
    nom.write(
        "[situation]\n\n\
        \
        # Temperature (Celsius degrees)\n\
        Temperature = %s\n\n\
        \
        # Wind angle (degrees)\n\
        Wind_angle = %s\n\n\
        \
        # Wind speed (m/s)\n\
        Wind = %s\n\n\
        \
        # Boundary height (m)\n\
        Boundary_height =  %s\n\n\
        \
        # Stability class\n\
        Stability = %s\n\n\
        \
        # Rainfall rate (mm/hr)\n\
        Rainfall_rate = 0.\n\n\
        \
        # Pressure (Pa)\n\
        Pressure = %s\n\n"
        % (temp, wind_dir, wind_speed, pbl, stabil_class, pressure)
    )
    nom.close()

    path_file = (
        chemin1 / "meteo.dat"
    )  # copie/colle le fichier meteo que l on vient de creer
    shutil.copy(path_file, chemin3)

    ############################################################################
    ### Compiler gaussian-deposition dans le bon dossier pour le preprocessing #
    ############################################################################

    os.chdir(chemin1)
    os.system(
        str(Path.cwd() / "polyphemus" / "utils" / "scons.py") + " gaussian-deposition"
    )

    #############################################
    ### lancer l'executable gaussian-deposition #
    #############################################

    os.chdir(chemin3)
    os.system(
        str(model_path / "preprocessing" / "dep" / "gaussian-deposition")
        + " gaussian-deposition.cfg"
    )

    # ? SX: is this meant to be done twice?
    path_file = (
        chemin1 / "meteo.dat"
    )  # copie/colle le fichier meteo que l on vient de creer
    shutil.copy(path_file, chemin3)

    return


def plume_response_function(
    site_name: str,
    source: np.array,
    date: str,
    rate: float,
    temp: float,
    window: float,
) -> None:
    """TODO"""
    model_path = Path.cwd() / "polyphemus" / site_name
    chemin1 = model_path / "preprocessing" / "dep"
    chemin2 = model_path / "processing" / "gaussian"
    chemin_fichiers = Path.cwd() / "polyphemus" / "fichiers" / "preprocessing" / "dep"
    chemin_fichiers2 = (
        Path.cwd() / "polyphemus" / "fichiers" / "processing" / "gaussian"
    )
    chemin3 = model_path / "graphes" / "config"
    chemin4 = chemin3 / "results"

    #####################################
    # creation fichier plume-source.dat #
    #####################################

    nom = open(chemin2 / "plume-source.dat", "w")
    nom.write(
        "[source]\n\n\
        \
        # Source coordinates (meters)\n\
        Abscissa: %s\n\
        Ordinate: %s\n\
        Altitude: %s\n\n\
        \
        # Species name\n\
        Species: Methane\n\n\
        \
        Type: continuous\n\
        Date_beg: %s\n\
        Date_end: %s\n\n\
        \
        # Source rate (mass/s)\n\
        Rate: %s\n\n\
        \
        # Source velocity (m/s)\n\
        Velocity = 0.\n\n\
        \
        # Source temperature (degrees Celsius)\n\
        Temperature = %s\n\n\
        \
        # Source diameter (m)\n\
        Diameter = %s\n\n"
        % (source[0], source[1], source[3], date, date, rate, temp, source[2])
    )
    nom.close()

    path_file = (
        chemin2 / "plume-source.dat"
    )  # copie/colle le fichier meteo que l on vient de creer
    shutil.copy(path_file, chemin3)

    #####################################
    # creation fichier plume.cfg 	    #
    #####################################

    nom = open(chemin2 / "plume.cfg", "w")
    nom.write(
        '[display]\n\n\
        \
        Show_iterations: yes\n\
        Show_meteorological_data: yes\n\
        Show_date: yes\n\n\n\
        \
        \
        [domain]\n\n\
        \
        ## Domain where species concentrations are computed.\n\
        Date_min: %s	Delta_t = 1.0	Nt = 1\n\
        x_min = 0.0		Delta_x = 1.	Nx = %s\n\
        y_min = 0.0		Delta_y = 1.	Ny = %s\n\n\
        \
        Nz = 7\n\
        Vertical_levels: plume-levels.dat\n\n\
        \
        # Land category: rural or urban.\n\
        Land_category: rural\n\n\
        \
        # Time of the day: night or day.\n\
        Time: day\n\n\
        \
        # File containing the species data.\n\
        Species: gaussian-species.dat\n\n\n\
        \
        \
        [gaussian]\n\n\
        \
        With_plume_rise: no\n\
        With_plume_rise_breakup: no\n\
        With_radioactive_decay: no\n\
        With_biological_decay: no\n\
        With_scavenging: no\n\
        With_dry_deposition: no\n\n\
        \
        # Parameterization to compute standard deviations: "Briggs", "Doury" or\n\
        # "similarity_theory".\n\
        Sigma_parameterization: Briggs\n\n\
        \
        # Is there a particular formula for standard deviation above the boundary layer?\n\
        # If "Gillani" is provided, the vertical sigma is computed with this formula.\n\
        # Otherwise, the formula is the same above and below the boundary layer.\n\
        Above_BL: none\n\n\
        \
        # Alternative parameterization. Useful only when using similarity theory.\n\
        # It is recommended to use it for elevated sources (about 200 m).\n\
        With_HPDM: no\n\n\
        \
        # Plume rise parameterization: put "HPDM", "Concawe" or "Holland".\n\
        Plume_rise_parameterization: HPDM\n\n\
        \
        # File containing the meteorological data.\n\
        File_meteo: gaussian-meteo.dat\n\n\
        \
        # File containing the source data.\n\
        File_source: plume-source.dat\n\n\
        \
        # File containing the correction coefficients (used with line sources only).\n\
        File_correction: correction_coefficients.dat\n\n\n\
        \
        \
        [deposition]\n\n\
        \
        # Deposition model: "Chamberlain" or "Overcamp".\n\
        Deposition_model: Overcamp\n\n\
        \
        # Number of points to compute the Chamberlain integral.\n\
        Nchamberlain: 100\n\n\n\
        \
        \
        [uncertainty]\n\n\
        \
        File_perturbation: perturbation.cfg\n\
        Number_samples = 10\n\
        # Newran seed directory (with "/" at the end), or "current_time" for random\n\
        # seeds generated with current time, or a given seed number (in ]0, 1[).\n\
        Random_seed = 0.5\n\n\n\
        \
        \
        [output]\n\n\
        \
        # File describing which concentrations are saved.\n\
        Configuration_file: plume-saver.cfg'
        % (date, window, window)
    )
    nom.close()

    path_file = (
        chemin2 / "plume.cfg"
    )  # copie/colle le fichier meteo que l on vient de creer
    shutil.copy(path_file, chemin3)

    ###################################################################
    ### copie des fichiers qui ne sont pas modifies dans les dossiers #
    ###################################################################

    path_file = chemin_fichiers2 / "plume.cpp"
    shutil.copy(path_file, chemin2)
    shutil.copy(path_file, chemin3)
    path_file = chemin_fichiers2 / "plume-levels.dat"
    shutil.copy(path_file, chemin2)
    shutil.copy(path_file, chemin3)
    path_file = chemin_fichiers2 / "SConstruct"
    shutil.copy(path_file, chemin2)
    path_file = chemin_fichiers2 / "plume-saver.cfg"
    shutil.copy(path_file, chemin2)
    shutil.copy(path_file, chemin3)
    path_file = chemin3 / "gaussian-meteo.dat"
    shutil.copy(path_file, chemin2)

    ###########################################################
    ### Compiler plume dans le bon dossier pour le processing #
    ###########################################################

    os.chdir(chemin2)
    os.system(str(Path.cwd() / "polyphemus" / "utils" / "scons.py") + " plume")

    ###############################
    ### lancer l'executable plume #
    ###############################

    os.chdir(chemin3)
    os.system(str(model_path / "processing" / "gaussian" / "plume") + " plume.cfg")

    return


def window(lat_s, lon_s, lat, lon, wind_speed, wind_dir, last):
    """TODO"""
    x_ext1, y_ext1 = convert_coord(lat[0], lon[0])
    x_ext2, y_ext2 = convert_coord(lat[last], lon[last])

    x_s, y_s = convert_coord(lat_s, lon_s)

    x_min, y_min, x_max, y_max = (
        min(x_ext1, x_ext2, x_s),
        min(y_ext1, y_ext2, y_s),
        max(x_ext1, x_ext2, x_s),
        max(y_ext1, y_ext2, y_s),
    )

    ## window size

    r = 6371e3  # earth's radius (m)
    lat_s, lat, lon_s, lon = map(np.radians, [lat_s, lat, lon_s, lon])

    ch_lat = lat[0] - lat_s
    ch_lon = lon[0] - lon_s
    a = (np.sin(ch_lat / 2) ** 2) + (
        np.cos(lat_s) * np.cos(lat[0]) * np.sin(ch_lon / 2) ** 2
    )
    c = 2 * math.atan2(np.sqrt(a), np.sqrt(1 - a))  # angular distance (rad)
    d_s_ext1 = r * c

    ch_lat = lat[last] - lat_s
    ch_lon = lon[last] - lon_s
    a = (np.sin(ch_lat / 2) ** 2) + (
        np.cos(lat_s) * np.cos(lat[0]) * np.sin(ch_lon / 2) ** 2
    )
    c = 2 * math.atan2(np.sqrt(a), np.sqrt(1 - a))  # angular distance (rad)
    d_s_ext2 = r * c

    ch_lat = lat[last] - lat[0]
    ch_lon = lon[last] - lon[0]
    a = (np.sin(ch_lat / 2) ** 2) + (
        np.cos(lat[0]) * np.cos(lat[last]) * np.sin(ch_lon / 2) ** 2
    )
    c = 2 * math.atan2(np.sqrt(a), np.sqrt(1 - a))  # angular distance (rad)
    d_ext1_ext2 = r * c

    window = max(d_s_ext1, d_s_ext2, d_ext1_ext2)
    window -= window % -100  # round window up to next multiple of 100
    window = window + 200
    print("Window:", window)

    ### window origin

    u, v = (-wind_speed * math.sin(math.radians(wind_dir))), (
        -wind_speed * math.cos(math.radians(wind_dir))
    )

    if u > 0 and v > 0:
        x0, y0 = x_min - 200, y_min - 200
    if u < 0 and v < 0:
        x0, y0 = x_max - window + 200, y_max - window + 200
    if u > 0 and v < 0:
        x0, y0 = x_min - 200, y_max - window + 200
    if u < 0 and v > 0:
        x0, y0 = x_max - window + 200, y_min - 200

    return window, x0, y0


def plume(
    site_name: str,
    sources: np.array,
    date: str,
    rate: float,
    temp: float,
    window: float,
) -> None:
    """Generates `plume-source.dat` and `plume.cfg` files, creates necessary
    directories and files, and runs the processing step of the gaussian plume model.

    Args:
        site_name (str): Name of site being modelled. Name of parent directory.
        sources (np.ndarray[float]): 2D numpy array containing information about
            sources. Each outer element corresponds to a source, whose elements
            entail the following:
                [0] x-coordinate in local coordinate system, in metres;
                [1] y-coordinate in local coordinate system, in metres;
                [2] source diameter, in metres;
                [3] source height above ground, in metres.
        date (str): Date of model (of measurements if applicable), in format
            "YYYY-MM-DD".
        rate (float): Source flux rate, in g/s.
        temp (float): Temperature, in degrees Celsius.
        window (float): Width of square window, in metres.
    """
    model_path = Path.cwd() / "polyphemus" / site_name
    chemin1 = model_path / "preprocessing" / "dep"
    chemin2 = model_path / "processing" / "gaussian"
    chemin_fichiers = Path.cwd() / "polyphemus" / "fichiers" / "preprocessing" / "dep"
    chemin_fichiers2 = (
        Path.cwd() / "polyphemus" / "fichiers" / "processing" / "gaussian"
    )
    chemin3 = model_path / "graphes" / "config"
    chemin4 = chemin3 / "results"

    #####################################
    # creation fichier plume-source.dat #
    #####################################

    nom = open(chemin2 / "plume-source.dat", "w")
    for ii in range(0, len(sources)):
        nom.write(
            "[source]\n\n\
            \
            # Source coordinates (meters)\n\
            Abscissa: %s\n\
            Ordinate: %s\n\
            Altitude: %s\n\n\
            \
            # Species name\n\
            Species: Methane\n\n\
            \
            Type: continuous\n\
            Date_beg: %s\n\
            Date_end: %s\n\n\
            \
            # Source rate (mass/s)\n\
            Rate: %s\n\n\
            \
            # Source velocity (m/s)\n\
            Velocity = 0.\n\n\
            \
            # Source temperature (degrees Celsius)\n\
            Temperature = %s\n\n\
            \
            # Source diameter (m)\n\
            Diameter = %s\n\n"
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

    path_file = (
        chemin2 / "plume-source.dat"
    )  # copie/colle le fichier meteo que l on vient de creer
    shutil.copy(path_file, chemin3)

    #####################################
    # creation fichier plume.cfg 	    #
    #####################################

    nom = open(chemin2 / "plume.cfg", "w")
    nom.write(
        '[display]\n\n\
        \
        Show_iterations: yes\n\
        Show_meteorological_data: yes\n\
        Show_date: yes\n\n\n\
        \
        \
        [domain]\n\n\
        \
        ## Domain where species concentrations are computed.\n\
        Date_min: %s	Delta_t = 1.0	Nt = 1\n\
        x_min = 0.0		Delta_x = 1.	Nx = %s\n\
        y_min = 0.0		Delta_y = 1.	Ny = %s\n\n\
        \
        Nz = 7\n\
        Vertical_levels: plume-levels.dat\n\n\
        \
        # Land category: rural or urban.\n\
        Land_category: rural\n\n\
        \
        # Time of the day: night or day.\n\
        Time: day\n\n\
        \
        # File containing the species data.\n\
        Species: gaussian-species.dat\n\n\n\
        \
        \
        [gaussian]\n\n\
        \
        With_plume_rise: no\n\
        With_plume_rise_breakup: no\n\
        With_radioactive_decay: no\n\
        With_biological_decay: no\n\
        With_scavenging: no\n\
        With_dry_deposition: no\n\n\
        \
        # Parameterization to compute standard deviations: "Briggs", "Doury" or\n\
        # "similarity_theory".\n\
        Sigma_parameterization: Briggs\n\n\
        \
        # Is there a particular formula for standard deviation above the boundary layer?\n\
        # If "Gillani" is provided, the vertical sigma is computed with this formula.\n\
        # Otherwise, the formula is the same above and below the boundary layer.\n\
        Above_BL: none\n\n\
        \
        # Alternative parameterization. Useful only when using similarity theory.\n\
        # It is recommended to use it for elevated sources (about 200 m).\n\
        With_HPDM: no\n\n\
        \
        # Plume rise parameterization: put "HPDM", "Concawe" or "Holland".\n\
        Plume_rise_parameterization: HPDM\n\n\
        \
        # File containing the meteorological data.\n\
        File_meteo: gaussian-meteo.dat\n\n\
        \
        # File containing the source data.\n\
        File_source: plume-source.dat\n\n\
        \
        # File containing the correction coefficients (used with line sources only).\n\
        File_correction: correction_coefficients.dat\n\n\n\
        \
        \
        [deposition]\n\n\
        \
        # Deposition model: "Chamberlain" or "Overcamp".\n\
        Deposition_model: Overcamp\n\n\
        \
        # Number of points to compute the Chamberlain integral.\n\
        Nchamberlain: 100\n\n\n\
        \
        \
        [uncertainty]\n\n\
        \
        File_perturbation: perturbation.cfg\n\
        Number_samples = 10\n\
        # Newran seed directory (with "/" at the end), or "current_time" for random\n\
        # seeds generated with current time, or a given seed number (in ]0, 1[).\n\
        Random_seed = 0.5\n\n\n\
        \
        \
        [output]\n\n\
        \
        # File describing which concentrations are saved.\n\
        Configuration_file: plume-saver.cfg'
        % (date, window, window)
    )
    nom.close()

    path_file = (
        chemin2 / "plume.cfg"
    )  # copie/colle le fichier meteo que l on vient de creer
    shutil.copy(path_file, chemin3)

    ###################################################################
    ### copie des fichiers qui ne sont pas modifies dans les dossiers #
    ###################################################################

    path_file = chemin_fichiers2 / "plume.cpp"
    shutil.copy(path_file, chemin2)
    shutil.copy(path_file, chemin3)
    path_file = chemin_fichiers2 / "plume-levels.dat"
    shutil.copy(path_file, chemin2)
    shutil.copy(path_file, chemin3)
    path_file = chemin_fichiers2 / "SConstruct"
    shutil.copy(path_file, chemin2)
    path_file = chemin_fichiers2 / "plume-saver.cfg"
    shutil.copy(path_file, chemin2)
    shutil.copy(path_file, chemin3)
    path_file = chemin3 / "gaussian-meteo.dat"
    shutil.copy(path_file, chemin2)

    ###########################################################
    ### Compiler plume dans le bon dossier pour le processing #
    ###########################################################

    os.chdir(chemin2)
    os.system(str(Path.cwd() / "polyphemus" / "utils" / "scons.py") + " plume")

    ###############################
    ### lancer l'executable plume #
    ###############################

    os.chdir(chemin3)
    os.system(str(model_path / "processing" / "gaussian" / "plume") + " plume.cfg")

    ####################################
    ### Tracer graphiques sur la route #
    ####################################

    return
