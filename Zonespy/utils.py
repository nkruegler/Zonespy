import os
import numpy as np
import pandas as pd

from . import sp    #SpiceyPy
from . import jm    #JupiterMag
from . import MODULE_DATA_DIR

"""
General utility functions for the Zonespy module.

Includes:
    - Loading of reference datasets available in published work, 
      including Juno/Waves auroral densities and reference auroral 
      ovals, and from non-published work, including auroral zone 
      boundaries and footprint ephemeris
    - Determining start and end times for event labels from the 
      Juno/Waves density data or the zone boundaries
    - Interpolating footprint locations at a finer resolution than 1 
      minute
    - Calculating the angle between a footprint and the auroral oval
"""

# ----------------------------------------------------------------------
# Pre-loading module data
# ----------------------------------------------------------------------

# --- Juno/Waves auroral density dataset (Kruegler et al., 2025) ---
# Loads from local cache if available, otherwise downloaded from 
#       Zenodo and cached
try:
    _waves_df = pd.read_csv(
        os.path.join(MODULE_DATA_DIR, "Kruegler2025_AuroralDensities.dat"),
        sep=r"\s+"
    )
except:
    _waves_df = pd.read_csv(
        "https://zenodo.org/records/14713895/files/Kruegler2025_AuroralDensities.dat",
        sep=r"\s+"
    )
    _waves_df.to_csv(
        os.path.join(MODULE_DATA_DIR, "Kruegler2025_AuroralDensities.dat"),
        sep="\t"
    )
# Convert times to a timezone-naive datetime object for comparison 
#       with other naive objects
_waves_df["datetime"] = pd.to_datetime(_waves_df["time(UTC)"]).dt.tz_localize(None)


# --- Reference auroral ovals (Head et al., 2024) ---
# Based on the average position of the main emission between Juno 
#       perijoves 1 and 54
# Loads from local cache if available, otherwise downloaded from 
#       Zenodo and cached
try:
    _NHead = pd.read_csv(os.path.join(MODULE_DATA_DIR, "NHead.csv"), sep=r"\s+")
    _SHead = pd.read_csv(os.path.join(MODULE_DATA_DIR, "SHead.csv"), sep=r"\s+")
except:
    _NHead = pd.read_csv("https://zenodo.org/records/12067692/files/UVSRefOval_N.csv")
    _SHead = pd.read_csv("https://zenodo.org/records/12067692/files/UVSRefOval_S.csv")
    _NHead.to_csv(os.path.join(MODULE_DATA_DIR, "NHead.csv"), sep="\t")
    _SHead.to_csv(os.path.join(MODULE_DATA_DIR, "SHead.csv"), sep="\t")
# Pre-calculate longitude and colatitude in radians to avoid repeated 
#       conversion at call time
_NHead["lon_r"] = np.radians(_NHead["longitude"])
_NHead["colat_r"] = np.radians(90 - _NHead["latitude"])
_SHead["lon_r"] = np.radians(_SHead["longitude"])
_SHead["colat_r"] = np.radians(90 - _SHead["latitude"])


# --- Auroral zone boundaries ---
# Determined from electron intensities and field-aligned currents
# See Mauk et al. (2020) and Sulaiman et al. (2022) for details
_zones_df = pd.read_csv(os.path.join(MODULE_DATA_DIR, "zone_boundary.dat"), sep=r"\s+")
_zones_df["Start"] = pd.to_datetime(_zones_df["Start"], format="%Y-%m-%dT%H:%M:%S")
_zones_df["End"] = pd.to_datetime(_zones_df["End"], format="%Y-%m-%dT%H:%M:%S")


# --- Juno magnetic footprints (credit: Masafumi Imai) ---
# Pre-calculated on a 1-min cadence
_footdf_60s = pd.read_csv(os.path.join(MODULE_DATA_DIR, "footprints_60s.dat"), sep=r"\s+")


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def get_start_end_time(source, event, buffer=10):
    """
    Returns the start and end datetimes for the named event.

    Time window can be determined from two data sources:
      - "omode": uses the start and end time of the Juno/Waves 
        auroral density dataset with a configurable buffer 
        surrounding the window
      - "zone": uses the start and end time of classified 
        auroral zones

    Parameters
    ----------
    source : str 
        Data source to derive the time window from
    event : str 
        String used to identify the relevant event
        (based on the event labeling in Kruegler et al., 2025)
    buffer : int, optional
        Number of minutes to pad before and after the auroral zones or 
        Waves auroral density time window

    Returns
    -------
    start_dt : pandas.Timestamp
        Start time of the event window
    end_dt : pandas.Timestamp
        End time of the event window
    """
    if source.lower() == "omode":
        # Filter Waves data to the requested event
        waves_df = _waves_df[_waves_df["event"].str.contains(event)]
        # Calculate the time window from the earliest and latest 
        #       times with buffer
        start_dt = waves_df["datetime"].min() + pd.Timedelta(minutes=-buffer)
        end_dt = waves_df["datetime"].max() + pd.Timedelta(minutes=buffer)

    elif source.lower() == "zone":
        # Filter auroral zones data to the requested event
        zones_df = _zones_df[_zones_df["Event"].str.contains(event)]
        # Calculate the time window from the earliest and latest times
        start_dt = zones_df["Start"].min() + pd.Timedelta(minutes=-buffer)
        end_dt = zones_df["End"].max() + pd.Timedelta(minutes=buffer)

    return start_dt, end_dt


def interpFootprint(interp_datetimes, UV = True):
    """
    Interpolates the footprint at a given time.

    Pre-calculated footprint data is stored at 1-minute cadence. This 
    function resamples footprints to the requested times using linear 
    interpolation. The interpolation is performed in 3D Cartesian 
    coordinates to avoid longitudinal discontinuity at 360°. (e.g. 
    interpolating between 1° and 359° should yield 0°, not 180°)

    SPICE ephemeris time is used as the interpolating variable because 
    np.interp requires a float value

    Parameters
    ----------
    interp_times : array-like of datetime-like
        Times at which the footprint should be calculated
    UV : bool, optional
        If True (default), returns the footprint at 400km above the 
        1-bar level (UV reference altitude). 
        If False, returns the footprint at 1-bar level.

    Returns
    -------
    interp_dict : dict
        Dictionary with the following keys:
        "interp_x", "interp_y", "interp_z"
            Interpolated Cartesian components of footprint in 
            System III coordinates (km)
        "interp_lats", "interp_lons"
            Interpolated latitude/west longitude of footprint (deg)
    """
    # Ensure inputs are pandas Timestamps so sp.datetime2et receives 
    #       timezone-aware objects
    interp_datetimes = pd.to_datetime(interp_datetimes)
    interp_ets = sp.datetime2et(interp_datetimes)
    ets = sp.str2et(_footdf_60s["utc"])

    # Select the appropriate latitude and longitude columns
    if UV:
        footlats = _footdf_60s["footlatUV"]
        footlons = _footdf_60s["footlonUV"]
    else:
        footlats = _footdf_60s["footlat"]
        footlons = _footdf_60s["footlon"]

    # Convert from System III latitude/longitude to Cartesian 
    #       unit vectors
    # Theta is colatitude and phi is the east longitude (west 
    #       longitude is used in Sys III)
    foot_theta = np.radians(90 - footlats)  # Colatitude
    foot_phi = np.radians(360 - footlons)   # East longitude

    x = np.sin(foot_theta) * np.cos(foot_phi)
    y = np.sin(foot_theta) * np.sin(foot_phi)
    z = np.cos(foot_theta)

    # Linearly interpolate Cartesian coordinates ate requested times
    interp_x = np.interp(interp_ets, ets, x)
    interp_y = np.interp(interp_ets, ets, y)
    interp_z = np.interp(interp_ets, ets, z)

    # Convert interpolated Cartesian coordinates to spherical components
    rho = np.sqrt(interp_x**2 + interp_y**2)    # Cylindrical radius
    theta = np.arctan2(rho, interp_z)       # Polar angle (colatitude)
    phi = np.arctan2(interp_y, interp_x) % (2 * np.pi)  # azimuthal angle

    # Convert interpolated spherical components to Sys III coordinates
    interp_lats = 90 - np.degrees(theta)    # Latitude
    interp_lons = 360 - np.degrees(phi)     # West longitude

    interp_dict = {"interp_x":interp_x,
                   "interp_y":interp_y,
                   "interp_z":interp_z,
                   "interp_lats":interp_lats,
                   "interp_lons":interp_lons}

    return interp_dict


def calcAngleFromOval(foot_lat, foot_lon):
    """
    Calculates the angle between the footprint position and the 
    statistical auroral oval.

    The "alpha" parameter is the central angle (arc length on the unit 
    sphere) between the footprint and the nearest point on the 
    reference oval. The sign encodes whether the footprint is 
    poleward (negative) or equatorward (positive) of the oval, 
    determined by comparing the footprint's angular distance from the 
    oval's geometric centroid against the closest oval point's 
    distance from the same centroid.    

    Parameters
    ----------
    foot_lat : float
        System III latitude of footprint (deg)
    foot_lon :float 
        System III west longitude of footprint (deg)

    Returns
    -------
    alpha : float
        Angle between the position vectors of the footprint and the 
        closest point of the statistical oval.
        Negative --> footprint is poleward of (inside) the oval.
        Positive --> footprint is equatorward of (outside) the oval.
    """
    # Convert input angles to radians (all angles are radians 
    #       going forward)
    foot_colat = np.radians(90 - foot_lat)
    foot_lon = np.radians(foot_lon)

    # Select reference oval and centroid based on hemisphere
    # Centroid coordinates from Greathouse et al. (2021)
    if foot_lat > 0:
        centroid_lon = np.radians(178.1)
        centroid_colat = np.radians(90 - 71.2)
        oval_df = _NHead
    if foot_lat < 0:
        centroid_lon = np.radians(30.4)
        centroid_colat = np.radians(90 - (-82.5))
        oval_df = _SHead

    oval_colats = oval_df["colat_r"]
    oval_lons = oval_df["lon_r"]

    # Calculates the central angle between the footprint position 
    #       vector and the array of oval position vectors
    dot_prods = (np.sin(foot_colat) * np.sin(oval_colats) * np.cos(foot_lon - oval_lons) 
                 + np.cos(foot_colat) * np.cos(oval_colats))
    alphas = np.degrees(np.arccos(dot_prods))

    # Identify the colatitude/longitude of the closest point on the oval
    min_index = np.argmin(alphas)
    oval_colat = oval_colats[min_index]
    oval_lon = oval_lons[min_index]
    alpha = alphas[min_index]

    # Determine sign of alpha (inside vs. outside oval)
    # Footprint is inside oval if it is closer to the centroid than the 
    #       nearest oval point; outside if it is further

    # Angle between footprint and centroid
    foot_dot_prod = (np.sin(foot_colat) * np.sin(centroid_colat) * np.cos(foot_lon - centroid_lon)
                     + np.cos(foot_colat) * np.cos(centroid_colat))
    foot_angle = np.arccos(foot_dot_prod)

    # Angle between centroid and closest oval point
    oval_dot_prod = (np.sin(oval_colat) * np.sin(centroid_colat) * np.cos(oval_lon - centroid_lon)
                     + np.cos(oval_colat) * np.cos(centroid_colat))
    oval_angle = np.arccos(oval_dot_prod)

    # Multiply angle by -1 if footprint outside oval
    alpha *= (oval_angle - foot_angle) // abs(oval_angle - foot_angle)

    return alpha


def trace_to_equator(pos0,step=0.01):
    """
    Trace along the magnetic field direction

    Parameters
    ----------
    pos0 : np.ndarray, shape (3,) 
        Initial coordinates along field line in Jovian radii
    step : float, optional
        Step size of field line tracing in Jovian radii
    """
    # Initialize list of positions
    pos_list = [pos0]
    
    # Set hemisphere based on starting z component to determine 
    #       tracing direction
    z0 = pos0[2]
    hemi = np.sign(z0)

    # Trace towards equator until z-component changes sign
    z_abs = hemi * z0
    while z_abs > -step / 2:
        # Calculate magnetic field unit vector from previous position
        Bfield = np.array(jm.Internal.Field(*pos_list[-1])).T[0]
        Bmag = np.sqrt((Bfield**2).sum(axis=0))
        Bhat = Bfield / Bmag

        # Step along magnetic field direction towards the equator
        pos_new = pos_list[-1] + hemi * step * Bhat

        # Add new position to the list of positions and update 
        #       z-component
        pos_list.append(pos_new)     
        z_abs = hemi * pos_new[2]

    return np.array(pos_list)
