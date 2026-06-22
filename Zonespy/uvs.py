import os
import datetime as datetime
import pandas as pd

import astropy.io.fits as fits
import astropy.time as atime

from . import DATA_DIR, MODULE_DATA_DIR


"""
Functions to extract relevant data from the published Juno/UVS profiles.

Data is sourced from https://doi.org/10.58119/ULG/SRUBKL. Unlike the 
other modules, data download through this submodule has not been 
implemented. Data must be manually downloaded.

This module is adapted from code by Bertrand Bonfond and Linus Head:
Copyright (c) 2026 - University of Liège 
Author: Bertrand Bonfond (b.bonfond@uliege.be), FRS-FNRS - University of Liège 
        Linus Head, University of Liège 
This file is licensed under the BSD-3 license 
(https://opensource.org/license/BSD-3-clause) 
"""


class UVSData:
    """
    Loads and unpacks Juno/UVS brightness data for a given time range.

    Class Attributes
    ----------------
    UVS_DIR : str 
        Directory to which UVS data is saved
    version_bri_fits_profil : str 
        Version of brightness profiles

    Instance Attributes
    -------------------
    datetime : np.ndarray, shape (N,)
        np.datetime64 values for each data point
    time_delay : np.ndarray, shape (N,)
        Delay between the SPICE ephemeris time and the UVS 
        measurement (seconds)
    bri_strip : np.ndarray, shape (101, N)
        Strip of uncorrected 155nm-162mn brightness         
        **Use for morphology only** - contains non calibrated data
    bri_prof : np.ndarray, shape (5, N)
        Corrected 155nm-162mn brightness (kR) (see Vinesse et al. 2026)
        Rows correspond to brightness profile at:
            0: Juno JRM33+Con2020 track at 400 km above 1bar level
            1: Same track shifted 1000km north
            2: Same track shifted 1000km east
            3: Same track shifted 1000km south
            4: Same track shifted 1000km west
    bri_prof_unc : np.ndarray, shape (5, N) 
        Uncertainty on the corrected 155nm-162mn (kR) 
        (Rows match bri_prof)
    bri_prof_nnv : np.ndarray, shape (5, N)
        Corrected 155nm-162mn brightness (non-nominal voltage) (kR) 
        (Same rows as bri_prof)
    """
    
    UVS_DIR = DATA_DIR + "UVS/"
    version_bri_fits_profil = "04"
    _perijove_df = pd.read_csv(os.path.join(MODULE_DATA_DIR, "perijove_list.dat"), sep=r"\s+")


    def __init__(self, start_dt, end_dt):
        """
        Initializes an instance of the UVSData class for the perijove 
        pass containing the requested times

        Parameters
        ----------
        start_dt : pandas.Timestamp
            Start time of the requested window
        end_dt : pandas.Timestamp
            End time of the requested window
        """
        self.start_dt = start_dt
        self.end_dt = end_dt

        # Identify the perijove containing the rime window
        before_pj = pd.to_datetime(UVSData._perijove_df["Time_Before_PJ_at_Radius_12Rj"])
        after_pj = pd.to_datetime(UVSData._perijove_df["Time_After_PJ_at_Radius_12Rj"])
        pj_filter = ((before_pj < self.start_dt) & (after_pj > self.end_dt))
        self._pj = int(UVSData._perijove_df[pj_filter]["Perijove"].values[0][2:])

        self._load_unpack_fits()    


    def _load_unpack_fits(self):
        """
        Opens the locally stored UVS FITS file and unpacks the 
        quantities important to UV brightness analysis.

        The complete raw data file is retained in the _raw_data 
        attribute for access to additional data
        """
        # Loading the data files.
        fits_file = (
            UVSData.UVS_DIR 
            + f"UVS_bri_derived_profil_new_jrm33_PJ{self._pj}"
            + f"_{UVSData.version_bri_fits_profil}.fits"
        )
        data = fits.open(fits_file)
        self._raw_data = data

        # Convert from julian time to datetime using astropy.time
        times = atime.Time(data[1].data["JUL_TIME"][0], format="jd")
        times.format = "datetime64"
        # Create filter to limit to requested time window
        time_filter = (times.value > self.start_dt) & (times.value < self.end_dt)
        self._time_filter = time_filter
        self.datetime = times.value[time_filter]

        # Time delay between SPICE ephemeris time and measurement time
        self.time_delay = data[4].data["TIME_DELAY"][:,:,time_filter][0]

        # Raw brightness strip
        self.bri_strip = data[5].data["BRI_STRIP"][:,:,time_filter][0]

        # Brightness profiles along trajectory and its uncertainty
        self.bri_prof = data[2].data['BRI_H2LY_COR'][:,:,time_filter][0]
        self.bri_prof_unc = data[2].data['UNCERTAINTY_BRI_H2LY_COR'][:,:,time_filter][0]

        # Non-nominal voltage brightness profiles
        self.bri_prof_nnv = data[6].data['BRI_H2LY_NNV_COR'][:,:,time_filter][0]