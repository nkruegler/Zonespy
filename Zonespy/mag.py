import os
import requests
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.constants import mu_0

from . import jm    #JupiterMag
from . import sp    #SpiceyPy

from . import DATA_DIR


"""
Defines a class to process Juno/MAG data into magnetic perturbations 
and field-aligned current (FAC) estimates.

Data is sourced from the Planetary Data System 
    (https://doi.org/10.17189/1519711)
This submodule accesses this data through the PDS file system 
(see MAG_PDS_URL variable)
"""


class MAGData:
    """
    Loads, processes, and saves Juno/MAG data for a given time range.

    Class Attributes
    ----------------
    MAG_PDS_URL : str 
        Base URL for the Juno/MAG data on the PDS
    MAG_DIR :str 
        Local directory where downloaded MAG files are stored

    Instance Attributes
    -------------------
    start_dt : pandas.Timestamp
        Start time of the requested window
    end_dt : pandas.Timestamp
        End time of the requested window
    datetime : np.ndarray, shape (N,)
        np.datetime64 values for each 1-second MAG sample
    et : np.ndarray, shape (N,)
        SPICE ephemeris times (seconds past J2000)
    Bfgm : np.ndarray, shape (3, N)
        Magnetic field vector measured by FGM in IAU_JUPITER frame (nT)
    Bmag : np.ndarray, shape (N,)
        Magnitude of the FGM magnetic field (nT)
    Bhat : np.ndarray, shape (3, N)
        Unit vector along the FGM magnetic field direction
    MAG_range : np.ndarray, shape (N)
        MAG dynamic range for each sample
    position : np.ndarry, shape (3, N)
        Juno position in IAU_JUPITER frame (km)
    velocity : np.ndarray, shape (3, N)
        Juno velocity in IAU_JUPITER frame (km/s)
    Bmodel : np.ndarray, shape (3, N)
        Model magnetic field (JRM33 + Con2020) at Juno's position (nT)
    delB : np.ndarray, shape (3, N)
        Magnetic field perturbation (Bfgm - Bmodel) (nT)
    delB_vxB : np.ndarray, shape (N,)
        Perturbation field component along the v-cross-B direction. 
        Used for FAC estimation via the Luhr method (nT)
    delB_velperp : np.ndarray, shape (N,)
        Perturbation field component along the velocity direction 
        perpendicular to B (nT)
    delB_perp : np.ndarray, shape (N,)
        Magnitude of the perturbation field perpendicular to B (nT)
    j_vxB : np.ndarray, shape (N,)
        Field-aligned current density estimated via the Luhr 
        method (μA/m^2)
    """

    MAG_PDS_URL = "https://pds-ppi.igpp.ucla.edu/data/JNO-J-3-FGM-CAL-V1.0/"
    MAG_DIR = DATA_DIR + "MAG/"

    # --- PDS Index File ---
    # Contains file names and the time ranges for all MAG products
    # Filtered to the planetocentric 1-second (PC 1 SECOND) files
    # Loads from local cache if available, otherwise downloaded from 
    #       PDS and cached
    try:
        _mag_index_df = pd.read_csv(DATA_DIR + "MAG_INDEX.TAB")
    except:
        _mag_index_df = pd.read_csv(
            "https://pds-ppi.igpp.ucla.edu/data/JNO-J-3-FGM-CAL-V1.0/INDEX/INDEX.TAB"
        )
        # Strip whitespace from column names and save to file
        _mag_index_df.columns = _mag_index_df.columns.str.strip()
        _mag_index_df.to_csv(DATA_DIR + "MAG_INDEX.TAB")
    
    # Keep only 1-second planetocentric files
    _mag_index_df = _mag_index_df[_mag_index_df["SID"].str.contains("PC 1 SECOND")]
    # Parse start/stop times from the day-of-year format
    _mag_index_df["START_TIME"] = pd.to_datetime(
        _mag_index_df["START_TIME"], format="%Y-%jT%H:%M:%S.%f"
    )
    _mag_index_df["STOP_TIME"] = pd.to_datetime(
        _mag_index_df["STOP_TIME"], format="%Y-%jT%H:%M:%S.%f"
    )


    def __init__(self, start_dt, end_dt, do_30s_avg=False, remove_5min_trend=False):
        """
        Initializes a MAGData instance and fully processes the data

        Processing includes:
            _parse_mag_file: read and trim MAG data
            _calc_ephemeris: compute position and velocity via SPICE
            _calc_model_perturbation_field: obtains the model and 
                perturbation magnetic fields (with optional smoothing)
            _calc_Luhr: estimates the FAC density and related 
                magnetic field components 

        Parameters
        ----------
        start_dt : pandas.Timestamp
            Start time of the requested window
        end_dt : pandas.Timestamp
            End time of the requested window
        do_30s_avg : bool, optional
            If True, applies a 30-second moving average to 
            perturbation field to supress spin modulation. Default is
            false
        remove_5min_trend : bool, optional
            If True, subtracts a 5min moving average from the
            perturbation field to remove large-scale background trends.
            Default is false
        """
        self.start_dt = start_dt
        self.end_dt = end_dt
        self._do_30s_avg = do_30s_avg
        self._remove_5min_trend = remove_5min_trend

        self._parse_MAG_file()
        self._calc_ephemeris()
        self._calc_model_perturbation_field()
        self._calc_Luhr()


    @staticmethod
    def _get_filepath_list(start_dt, end_dt):
        """
        Returns the paths (relative to MAG_PDS_URL) for all perijove MAG 
        files that overlap the requested time window

        Parameters
        ----------
        start_dt : pandas.Timestamp
            Start time of the requested window
        end_dt : pandas.Timestamp
            End time of the requested window

        Returns
        -------
        filepath_list : list of str
            File paths relative to the MAG_PDS_URL with the .lbl 
            extension replaced by .sts
            Returns empty list if no files available for the time range
        """
        # Filter out files that end before start_dt or start 
        #       after end_dt
        time_mask = ~(
            (MAGData._mag_index_df["STOP_TIME"] <= start_dt) 
            | (MAGData._mag_index_df["START_TIME"] >= end_dt)
        )
        # Restrict to perijove only files
        pj_file = MAGData._mag_index_df["PRODUCT_ID"].str.contains("PJ")

        # Extract file paths and change the file extension
        filepath_list = (
            MAGData._mag_index_df[time_mask & pj_file]["FILE_SPECIFICATION_NAME"]
            .str.strip()
            .values
        )
        filepath_list = [file[:-4] + ".sts" for file in filepath_list]
        
        return filepath_list


    @staticmethod
    def downloadMAGLocal(start_dt, end_dt, overwrite = False):
        """
        Downloads all MAG perijove files needed to cover the 
        time range from the PDS and saves them to MAG_DIR

        Files that are already present locally are skipped unless
        ``overwrite=True`` is specified

        Parameters
        ----------
        start_dt : pandas.Timestamp
            Start time of the requested window
        end_dt : pandas.Timestamp
            End time of the requested window
        overwrite : bool, optional
            If True, re-downloads and recompiles files even if the file 
            already exists locally. Default if False
        """
        filepath_list = MAGData._get_filepath_list(start_dt, end_dt)

        if len(filepath_list) == 0:
            raise Exception("Files not present on PDS. Download Failed.")

        for filepath in filepath_list:
            # Get filename without the directory path
            filename = filepath.split("/")[-1]

            url = MAGData.MAG_PDS_URL + filepath
            save_file = MAGData.MAG_DIR + filename

            # Skip download if the file already exists locally
            if not overwrite:
                if os.path.isfile(save_file):
                    print(
                        f"{filename} already downloaded. "
                        "Use 'overwrite' if new version requested."
                    )
                    continue

            print(f"Downloading {filename} from PDS...")
            r = requests.get(url)

            print("Download done. Now processing and writing to file...")
            with open(save_file, "w") as f:
                f.write(r.text.replace("\r\n", "\n"))


    def _parse_MAG_file(self):
        """
        Reads locally stored MAG files and populates time and magnetic 
        field attributes

        This method identifies the required files via 
        _get_filepath_list, reads the files (skipping the header rows),
        reconstructs datetime information from separate 
        Year/DOY/Hour/Min/Sec/MSec columns, trims to the requested time 
        window, and stores the result
        """
        filepath_list = MAGData._get_filepath_list(self.start_dt, self.end_dt)
        filename_list = [filepath.split("/")[-1] for filepath in filepath_list]
    
        if len(filename_list) == 0:
            raise Exception(f"MAG data not downloaded. Use MAGData.downloadMAGLocal")

        # Column names for the MAG data file
        names = [
            "Year", "DOY", "Hour", "Min", "Sec", "MSec",
            "DecimalDay", 
            "Bx", "By", "Bz", 
            "Range", 
            "X", "Y", "Z"
        ]

        # Read and concatenate all files covering the time range
        #       (should be one file per perijove, but the loop handles 
        #       possible edges cases where the window spans a boundary)
        df_list = []
        for filename in filename_list:
            with open(MAGData.MAG_DIR + filename) as f:
                rows = f.read().splitlines()
                head_idx = [idx for idx, line in enumerate(rows) if "END_OBJECT" in line][-1]
            
            df = pd.read_csv(
                MAGData.MAG_DIR + filename, 
                sep=r"\s+", 
                skiprows=head_idx + 1, 
                names=names
            )
            df_list.append(df)
        df = pd.concat(df_list)

        # Reconstruct datetime info from separate columns
        # Year*1000 + DOY produces a string parseable as "%Y%j" 
        #       (e.g. 2017005)
        df["datetime"] = (
            pd.to_datetime(df["Year"] * 1000 + df["DOY"], format="%Y%j")
            + pd.to_timedelta(df["Hour"], unit="h")
            + pd.to_timedelta(df["Min"],  unit="m")
            + pd.to_timedelta(df["Sec"],  unit="s")
            + pd.to_timedelta(df["MSec"], unit="ms")
        )

        # Trim to desired time window and reset the index to start at 0
        df = df[df["datetime"].between(self.start_dt, self.end_dt)]
        df = df.reset_index(drop=True)

        self.datetime = df["datetime"].to_numpy()
        self.et = sp.datetime2et(df["datetime"])

        self.Bfgm = np.array([df["Bx"], df["By"], df["Bz"]])
        self.Bmag = np.sqrt((self.Bfgm**2).sum(axis=0))
        self.Bhat = self.Bfgm / self.Bmag
        self.MAG_range = df["Range"].to_numpy()


    def _calc_ephemeris(self):
        """
        Calculates Juno's position and velocity in the IAU_JUPITER 
        coordinate frame using SPICE
        """
        [state, ltime] = sp.spkezr("Juno", self.et, "IAU_JUPITER", "None", "Jupiter")
        self.position = np.array(state)[:,:3].T
        self.velocity = np.array(state)[:,3:].T


    def _calc_model_perturbation_field(self):
        """
        Calculates the magnetic field perturbation by subtracting the 
        model magnetic field from the measured field

        Includes optional smoothing steps (30-second moving average to 
        supress spin modulation and 5-minute trend removal to isolate 
        short-period variations)
        """
        # Evaluate internal (JRM33) and external (Con2020) model 
        #       magnetic fields
        # JupiterMag requires positions in the units of Jupiter radii
        Bint = np.array(jm.Internal.Field(*self.position / 71492))
        Bext = np.array(jm.Con2020.Field(*self.position / 71492))
        self.Bmodel = Bint + Bext

        delB = self.Bfgm - self.Bmodel

        # Optional 30-second average to minimize spin modulation
        if self._do_30s_avg:
            delB = np.array([
                uniform_filter1d(delB[0], 30), 
                uniform_filter1d(delB[1], 30), 
                uniform_filter1d(delB[2], 30)
            ])
        # Optional 5-minute trend removal to isolate short-period 
        #       variations
        if self._remove_5min_trend:
            delB = np.array([
                delB[0] - uniform_filter1d(delB[0], 300), 
                delB[1] - uniform_filter1d(delB[1], 300), 
                delB[2] - uniform_filter1d(delB[2], 300)
            ])

        self.delB = delB


    def _calc_Luhr(self):
        """
        Estimates the field-aligned current density (FAC) via the Luhr 
        method (Luhr et al., 1996)

        This method estimates the FAC from the along-track gradient
        of the perturbation field component in the v-cross-B direction
        (perpendicular to both the spacecraft velocity and the 
        background field). Two additional perturbation components are 
        computed for completeness but are not used in the FAC estimate.
        """
        # --- delB in v-cross-B direction ---
        # Serves as the main perturbation component for the FAC
        vxB = np.cross(self.velocity, self.Bhat, axisa=0, axisb=0, axisc=0)
        vxB_mag = np.sqrt((vxB**2).sum(axis=0))
        vxB_hat = vxB / vxB_mag
        self.delB_vxB = (vxB_hat * self.delB).sum(axis=0)

        # --- delB in direction of velocity perpendicular to B ---
        # Not used for FAC, but provided for completeness
        vel_perp = self.velocity - ((self.velocity * self.Bhat).sum(axis=0) * self.Bhat)
        vel_perp_mag = np.sqrt((vel_perp**2).sum(axis=0))
        vel_perp_hat = vel_perp / vel_perp_mag
        self.delB_velperp = (vel_perp_hat * self.delB).sum(axis=0)

        # --- delB perpendicular to B ---
        # Not used for FAC, but provided for completeness
        delB_perp = self.delB - ((self.delB * self.Bhat).sum(axis=0) * self.Bhat)
        self.delB_perp = np.sqrt((delB_perp**2).sum(axis=0))

        # --- Luhr FAC ---
        # mu_0 from scipy in units of kg⋅m⋅s−2⋅A−2 (equiv. to km⋅nT/μA)
        j_vxB = (
            -1 / (mu_0 * vel_perp_mag) 
            * np.gradient(self.delB_vxB, self.et)   # μA/km^2
        )  
        j_vxB *= 1e-6   # Convert to μA/m^2
        self.j_vxB = j_vxB


    def _calc_current_Cartesian(self):
        """
        Estimates the field-aligned current density (FAC) using the 
        curl of delB in Cartesian coordinates

        Warning: This method is unreliable for single-spacecraft 
        measurements and may result in artificial and non-uniform 
        enhancement of the current along the spacecraft track. This 
        method also introduces spikes at times where any component of 
        the velocity approaches 0

        Returns
        -------
        j_par : np.ndarray, shape (N,)
            FAC density projected along the measured field direction
        """
        term1 = (
            np.gradient(self.delB[2], self.position[1]) 
            - np.gradient(self.delB[1], self.position[2])
        )
        term2 = (
            np.gradient(self.delB[0], self.position[2]) 
            - np.gradient(self.delB[2], self.position[0])
        )
        term3 = (
            np.gradient(self.delB[1], self.position[0]) 
            - np.gradient(self.delB[0], self.position[1])
        )

        # Estimate current in cartesian coords.
        # mu_0 from scipy in units of kg⋅m⋅s−2⋅A−2 (equiv. to km⋅nT/μA)
        j = 1 / mu_0 * np.array([term1, term2, term3])  #μA/km^2
        j *= 1e-6   # Convert to μA/m^2

        # Project current vector onto the measured field direction
        j_par = (j * self.Bfgm).sum(axis=0) / self.Bmag

        return j_par

    
    def to_tsv(self, filename, dir=MAG_DIR):
        """
        Write processed MAG data to a tab-separated file

        Parameters
        ----------
        filename : str
            Name of processed file (including extension)
        dir : str
            Directory where processed file is saved. Default is MAG_DIR
        """
        names = [
            "utc",
            "x", "y", "z",
            "vx", "vy", "vz",
            "Bfgm_x", "Bfgm_y", "Bfgm_z",
            "MAG_Range",
            "Bmodel_x", "Bmodel_y", "Bmodel_z",
            "delB_x", "delB_y", "delB_z",
            "delB_vxB",
            "j_vxB"
        ]
        
        # Convert SPICE ephermeis time to UTC strings for the file
        date_str = sp.et2utc(self.et, "ISOC", 3)
        data_arr = [
            date_str,
            *self.position,
            *self.velocity,
            *self.Bfgm,
            self.MAG_range,
            *self.Bmodel,
            *self.delB,
            self.delB_vxB,
            self.j_vxB
        ]

        mag_df = pd.DataFrame(data_arr, index=names).T
        mag_df.to_csv(dir + filename, sep="\t", index=False)