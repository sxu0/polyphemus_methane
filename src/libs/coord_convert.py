# pylint: disable=all

"""
coord_convert.py

Author: Sebastien Ars
Modified by Shiqi Xu

Converts between latitude/longitude (degrees) and UTM coordinates (meters).
"""

#!/usr/bin/env python


import numpy as np


def convert_coord(lat, lon):
    """Converts latitude and longitude (degrees) to UTM coordinates (m).

    Args:
        lat (float): Latitude, in degrees.
        lon (float): Longitude, in degrees.

    Returns:
        E: UTM easting, in meters.
        N: UTM northing, in meters.
    """

    # convert latitude and longitude to
    # spherical coordinates in radians
    phi = np.radians(lat)
    l = np.radians(lon)

    # WGS 84
    a = 6378137  # semi-major axis of ellipsoid (m)
    e = 0.08181919106  # first eccentricity of ellipsoid
    f = 1 / 298.257223563  # inverse flattening
    N0 = 0  # 0 for north and 10 000 km for south
    k0 = 0.9996
    E0 = 500000

    # projection parameters
    l0 = np.radians(-81)  # reference longitude

    # preliminary values
    n = f / (2 - f)
    A = a / (1 + n) * (1 + n**2 / 4 + n**4 / 64 + n**6 / 256)

    alpha1 = 1/2 * n - 2/3 * n**2 + 5/16 * n**3
    alpha2 = 13/48 * n**2 - 5/3 * n**3
    alpha3 = 61/240 * n**3

    t = np.sinh(
        np.arctanh(np.sin(phi))
        - 2 * np.sqrt(n) / (1 + n) * np.arctanh(2 * np.sqrt(n) / (1 + n) * np.sin(phi))
    )
    eta = np.arctanh(np.sin(l - l0) / np.sqrt(1 + t * t))
    xi = np.arctan(t / np.cos(l - l0))

    # UTM coordinates
    E = E0 + k0 * A * (
        eta
        + alpha1 * np.cos(2 * xi) * np.sinh(2 * eta)
        + alpha2 * np.cos(2 * 2 * xi) * np.sinh(2 * 2 * eta)
        + alpha3 * np.cos(2 * 3 * xi) * np.sinh(2 * 3 * eta)
    )
    N = N0 + k0 * A * (
        xi
        + alpha1 * np.sin(2 * xi) * np.cosh(2 * eta)
        + alpha2 * np.sin(2 * 2 * xi) * np.cosh(2 * 2 * eta)
        + alpha3 * np.sin(2 * 3 * xi) * np.cosh(2 * 3 * eta)
    )

    return E, N


def convert_proj(X, Y):
    """Converts UTM coordinates (m) to latitude and longitude (degrees).

    Args:
        X: UTM easting, in meters.
        Y: UTM northing, in meters.

    Returns:
        lat (float): Latitude, in degrees.
        lon (float): Longitude, in degrees.
    """

    # WGS 84
    a = 6378137  # semi-major axis of ellipsoid (m)
    e = 0.08181919106  # first eccentricity of ellipsoid
    f = 1 / 298.257223563  # inverse flattening
    N0 = 0  # 0 for north and 10 000 km for south
    k0 = 0.9996
    E0 = 500000

    # projection parameters
    l0 = np.radians(-81)  # reference longitude

    # preliminary values
    n = f / (2 - f)
    A = a / (1 + n) * (1 + n**2 / 4 + n**4 / 64 + n**6 / 256)

    beta1 = 1/2 * n - 2/3 * n**2 + 37/96 * n**3
    beta2 = 1/48 * n**2 + 1/15 * n**3
    beta3 = 17/480 * n**3

    delta1 = 2 * n - 2/3 * n**2 - 2 * n**3
    delta2 = 7/3 * n**2 - 8/5 * n**3
    delta3 = 56/15 * n**3

    eta = (Y - N0) / (k0 * A)
    nu = (X - E0) / (k0 * A)

    eta_p = eta - (
        beta1 * np.sin(2 * 1 * eta) * np.cosh(2 * 1 * nu)
        + beta2 * np.sin(2 * 2 * eta) * np.cosh(2 * 2 * nu)
        + beta3 * np.sin(2 * 3 * eta) * np.cosh(2 * 3 * nu)
    )
    nu_p = nu - (
        beta1 * np.cos(2 * 1 * eta) * np.sinh(2 * 1 * nu)
        + beta2 * np.cos(2 * 2 * eta) * np.sinh(2 * 2 * nu)
        + beta3 * np.cos(2 * 3 * eta) * np.sinh(2 * 3 * nu)
    )

    xi = np.arcsin(np.sin(eta_p) / np.cosh(nu_p))

    # lat/lon coordinates
    phi = xi + (
        delta1 * np.sin(2 * 1 * xi)
        + delta2 * np.sin(2 * 2 * xi)
        + delta3 * np.sin(2 * 3 * xi)
    )
    l = l0 + np.arctan(np.sinh(nu_p) / np.cos(eta_p))

    # coordonnes du point traduire
    lat = np.degrees(phi)
    lon = np.degrees(l)

    return lat, lon
