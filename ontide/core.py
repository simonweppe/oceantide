# -*- coding: utf-8 -*-

"""Core tools for tidal analysis and prediction."""

import numpy as np
from .constituents import PERIODS, OMEGA, V0U


def nodal(time, con):
    """ Nodal correction
	    Derived from the tide model driver matlab scipt: nodal.m
	"""
    rad = np.pi / 180.0
    s, h, p, omega = astrol(time)
    sinn = np.sin(omega * rad)
    cosn = np.cos(omega * rad)
    sin2n = np.sin(2 * omega * rad)
    cos2n = np.cos(2 * omega * rad)
    sin3n = np.sin(3 * omega * rad)

    ndict = {
        "M2": {
            "f": np.sqrt(
                (1.0 - 0.03731 * cosn + 0.00052 * cos2n) ** 2
                + (0.03731 * sinn - 0.00052 * sin2n) ** 2
            ),
            "u": np.arctan(
                (-0.03731 * sinn + 0.00052 * sin2n)
                / (1.0 - 0.03731 * cosn + 0.00052 * cos2n)
            )
            / rad,
        },
        "S2": {"f": 1.0, "u": 0.0},
        "K1": {
            "f": np.sqrt(
                (1.0 + 0.1158 * cosn - 0.0029 * cos2n) ** 2
                + (0.01554 * sinn - 0.0029 * sin2n) ** 2
            ),
            "u": np.arctan(
                (-0.1554 * sinn + 0.0029 * sin2n)
                / (1.0 + 0.1158 * cosn - 0.0029 * cos2n)
            )
            / rad,
        },
        "O1": {
            "f": np.sqrt(
                (1.0 + 0.189 * cosn - 0.0058 * cos2n) ** 2
                + (0.189 * sinn - 0.0058 * sin2n) ** 2
            ),
            "u": 10.8 * sinn - 1.3 * sin2n + 0.2 * sin3n,
        },
        "N2": {
            "f": np.sqrt(
                (1.0 - 0.03731 * cosn + 0.00052 * cos2n) ** 2
                + (0.03731 * sinn - 0.00052 * sin2n) ** 2
            ),
            "u": np.arctan(
                (-0.03731 * sinn + 0.00052 * sin2n)
                / (1.0 - 0.03731 * cosn + 0.00052 * cos2n)
            )
            / rad,
        },
        "P1": {"f": 1.0, "u": 0.0},
        "K2": {
            "f": np.sqrt(
                (1.0 + 0.2852 * cosn + 0.0324 * cos2n) ** 2
                + (0.3108 * sinn + 0.0324 * sin2n) ** 2
            ),
            "u": np.arctan(
                -(0.3108 * sinn + 0.0324 * sin2n)
                / (1.0 + 0.2852 * cosn + 0.0324 * cos2n)
            )
            / rad,
        },
        "Q1": {
            "f": np.sqrt((1.0 + 0.188 * cosn) ** 2 + (0.188 * sinn) ** 2),
            "u": np.arctan(0.189 * sinn / (1.0 + 0.189 * cosn)) / rad,
        },
    }

    # Prepare the output data
    ncon = len(con)
    pu = np.zeros((ncon, 1))
    pf = np.ones((ncon, 1))
    v0u = np.zeros((ncon, 1))

    for ii, vv in enumerate(con):
        if vv in ndict:
            pu[ii, :] = ndict[vv]["u"] * rad
            pf[ii, :] = ndict[vv]["f"]
        if vv in V0U.keys():
            v0u[ii, :] = V0U[vv]

    return pu, pf, v0u


def astrol(time):
    """ Computes the basic astronomical mean longitudes  s, h, p, N.
        Note N is not N', i.e. N is decreasing with time.
        These formulae are for the period 1990 - 2010, and were derived
        by David Cartwright (personal comm., Nov. 1990).
        time is UTC in decimal MJD.
        All longitudes returned in degrees.
        R. D. Ray    Dec. 1990
        Non-vectorized version. Re-make for matlab by Lana Erofeeva, 2003
    
        usage: s, h, p, N = astrol(time)
        
        time, MJD
        circle = 360;
        T = time - 51544.4993
        mean longitude of moon
        ----------------------
        s = 218.3164 + 13.17639648 * T
        mean longitude of sun
        ---------------------
        h = 280.4661 +  0.98564736 * T
        mean longitude of lunar perigee
        -------------------------------
        p =  83.3535 +  0.11140353 * T
        mean longitude of ascending lunar node
        --------------------------------------
        N = 125.0445D0 -  0.05295377D0 * T
        s = mod(s, circle)
        h = mod(h, circle)
        p = mod(p, circle)
        N = mod(N, circle)

    """
    circle = 360
    T = time - 51544.4993
    # mean longitude of moon
    # ----------------------
    s = 218.3164 + 13.17639648 * T
    # mean longitude of sun
    # ---------------------
    h = 280.4661 + 0.98564736 * T
    # mean longitude of lunar perigee
    # -------------------------------
    p = 83.3535 + 0.11140353 * T
    # mean longitude of ascending lunar node
    # --------------------------------------
    N = 125.0445 - 0.05295377 * T
    #
    s = np.mod(s, circle)
    h = np.mod(h, circle)
    p = np.mod(p, circle)
    N = np.mod(N, circle)

    return s, h, p, N
