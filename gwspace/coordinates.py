#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Wang 2026
"""Coordinate transformation."""

import numpy as np
from astropy.coordinates import cartesian_to_spherical
from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic, GeocentricTrueEcliptic
import astropy.units as u


def icrs_to_ecliptic(ra, dec, center='bary'):
    """Convert ICRS(Equatorial) to Ecliptic frame.
     https://docs.astropy.org/en/stable/coordinates/index.html
     Reminder: Both dec & latitude range (pi/2, -pi/2) [instead of (0, pi)].
    :param ra: float, right ascension
    :param dec: float, declination
    :param center: {'bary', str}, 'bary' or 'geo'  # Actually it won't have too much difference
    :return: longitude, latitude: float
    """
    co = SkyCoord(ra*u.rad, dec*u.rad)
    if center == 'bary':
        cot = co.transform_to(BarycentricTrueEcliptic)
    elif center == 'geo':
        cot = co.transform_to(GeocentricTrueEcliptic)
    else:
        raise ValueError("'center' should be 'bary' or 'geo'")
    return cot.lon.rad, cot.lat.rad  # (Lambda, Beta)


def ecliptic_longitude_from_icrs(ra, dec, center='bary'):
    return icrs_to_ecliptic(ra, dec, center)[0]


def ecliptic_latitude_from_icrs(ra, dec, center='bary'):
    return icrs_to_ecliptic(ra, dec, center)[1]


def ecliptic_to_icrs(lon, lat, center='bary'):
    if center == 'bary':
        co = SkyCoord(lon * u.rad, lat * u.rad, frame='barycentrictrueecliptic')
    elif center == 'geo':
        co = SkyCoord(lon * u.rad, lat * u.rad, frame='geocentrictrueecliptic')
    else:
        raise ValueError("'center' should be 'bary' or 'geo'")
    cot = co.transform_to('icrs')
    return cot.ra.rad, cot.dec.rad


def ra_from_ecliptic(lon, lat, center='bary'):
    return ecliptic_to_icrs(lon, lat, center)[0]


def dec_from_ecliptic(lon, lat, center='bary'):
    return ecliptic_to_icrs(lon, lat, center)[1]


def ssb_to_tianqin_frame(lam, beta):
    """Convert Ecliptic frame to TianQin detector frame.
     Matrix of ecliptic to detector: see Hu et al. https://iopscience.iop.org/article/10.1088/1361-6382/aab52f
     Reminder: Both beta & latitude range (-pi/2, pi/2).
    :param lam: float, ecliptic longitude (Lambda)
    :param beta: float, ecliptic latitude (Beta)
    :return: longitude, latitude: float
    """
    # from gwspace.constants import J0806_phi, J0806_theta
    # phi_j, theta_j = J0806_phi, np.pi/2 - J0806_theta
    # rz = np.array([[np.sin(phi_j), np.cos(phi_j), 0],
    #                [-np.cos(phi_j), np.sin(phi_j), 0],
    #                [0, 0, 1]])
    # rx = np.array([[1, 0, 0],
    #                [0, np.sin(theta_j), np.cos(theta_j)],
    #                [0, -np.cos(theta_j), np.sin(theta_j)]])
    # trans = rz.dot(rx)
    # trans_inv = np.linalg.inv(trans)

    car = np.array([np.cos(beta)*np.cos(lam), np.cos(beta)*np.sin(lam), np.sin(beta)])
    trans_inv = np.array([[0.8616291604415259,   0.5075383629607039,   0.],
                          [0.04158693653353248, -0.07060060839873289, -0.9966373868180366],
                          [-0.5058317077710601,  0.8587318348686612,  -0.08193850863004093]])
    car_t = cartesian_to_spherical(*trans_inv.dot(car))
    return car_t[2].rad, car_t[1].rad  # lon, lat


def tianqin_frame_to_ssb(lon, lat):
    """Convert TianQin detector frame to Ecliptic frame.
     Matrix of ecliptic to detector: see Hu et al. https://iopscience.iop.org/article/10.1088/1361-6382/aab52f
     Reminder: Both beta & latitude range (-pi/2, pi/2).
    :param lon: float, longitude in detector frame
    :param lat: float, latitude in detector frame
    :return: Lambda, Beta: float
    """
    car = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    trans = np.array([[0.8616291604415259,  0.04158693653353248, -0.5058317077710601],
                      [0.5075383629607039, -0.07060060839873289,  0.8587318348686612],
                      [0.,                 -0.9966373868180366,  -0.08193850863004093]])
    car_t = cartesian_to_spherical(*trans.dot(car))
    return car_t[2].rad, car_t[1].rad  # (Lambda, Beta)


__all__ = ['icrs_to_ecliptic', 'ecliptic_to_icrs',
           'ecliptic_longitude_from_icrs', 'ecliptic_latitude_from_icrs',
           'ra_from_ecliptic', 'dec_from_ecliptic',
           'ssb_to_tianqin_frame', 'tianqin_frame_to_ssb']
