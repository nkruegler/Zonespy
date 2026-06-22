import glob
import os
import requests
import numpy as np
import pandas as pd
from scipy.constants import e

from . import jm    #JupiterMag
from . import sp    #SpiceyPy

from . import DATA_DIR
from . import utils


"""
Defines a class to process Juno/JEDI High Energy Resolution Electron 
Spectra (HIERSESP) data

Data is sourced from the Planetary Data System 
    (https://doi.org/10.17189/1519713)
This submodule accesses this data through the PDS file system 
    (see JEDI_PDS_URL variable)
"""

class JEDIData:
    """
    Loads, processes, and saves Juno/JEDI HIERSESP data for a given 
    time range.

    Class Attributes
    ----------------
    JEDI_PDS_URL : str
        Base URL for the Juno/JEDI data on the PDS
    JEDI_DIR : str
        Local directory where downloaded JEDI files are stored

    Instance Attributes
    -------------------
    start_dt : pandas.Timestamp
        Start time of the requested window, floored to t_res cadence
    end_dt : pandas.Timestamp
        End time of the requested window, ceiled to t_res cadence
    lower_energy : float, optional
        Lower bound of the output energy range (keV)
    upper_energy : float, optional
        Upper bound of the output energy range (keV)
    num_en_bins : int, optional
        Number of log-spaced output energy bins
    pitch_angle_res : float, optional
        Pitch angle bin width (degrees), adjusted if it does not 
        divide 180° evenly
    t_res : str, optional
        Time resolution as a pandas offset alias (e.g. "1s")
    num_times : int
        Number of time bins
    time_bins : np.ndarray, shape (num_times + 1,)
        Edges of the resampled time bins
    mean_times : np.ndarray, shape (num_times,)
        Midpoints of each time bin
    et : np.ndarray, shape (num_times,)
        SPICE ephemeris time at each mean_times value 
        (seconds past J2000)
    position : np.ndarray, shape (3, num_times)
        Juno position in IAU_JUPITER frame (km)
    radius_rj : np.ndarray, shape (num_times,)
        Juno radial distance from Jupiter's center (Jovian radii)
    hemi : str or None
        Hemisphere of the observation: "N", "S", or None if the 
        trajectory crosses the equator
    up_direction : str or None
        Field-line direction that points away from Jupiter 
        ("parallel" or "antiparallel")
    down_direction : str or None
        Field-line direction that points toward Jupiter 
        ("parallel" or "antiparallel")
    para_direction : str or None
        Spatial direction corresponding to the parallel field direction 
        ("up" or "down")
    anti_direction : str or None
        Spatial direction corresponding to the anti-parallel field 
        direction ("up" or "down")
    loss_cone_angle : np.ndarray, shape (num_times,)
        Loss cone angle at each mean_times value
    num_pitch_angle_bins : int
        Number of pitch angle bins
    pitch_angle_bins : np.ndarray, shape (num_pitch_angle_bins + 1,)
        Edges of the pitch angle bins from 0° to 180°
    mean_pitch_angles : np.ndarray, shape (num_pitch_angle_bins,)
        Midpoint of each pitch angle bin (degrees)
    energy_bins : np.ndarray, shape (num_en_bins + 1,)
        Edges of the log-spaced output energy bins (keV)
    mean_energies : np.ndarray, shape (num_en_bins,)
        Arithmetic mean of the log-spaced output energy bins (keV)
    geom_mean_energies : np.ndarray, shape (num_en_bins,)
        Geometric mean of the log-spaced output energy bins (keV)
    delE : np.ndarray, shape (num_en_bins,)
        Width of the log-spaced output energy bins (keV)
    datablock : np.ndarray, shape (num_times, num_en_bins, num_pitch_angle_bins)
        Mean differential intensity (keV/cm^2/s/sr/keV) in each 
        time-energy-pitch angle cell, averaged over all contributing 
        telescopes
    max_intensity : float, optional
        Maximum value of intensity to allow into the rebinned 
        datablock. Removes extremely high intensities that 
        complicate subsequent analyses
    """    
    
    JEDI_PDS_URL = "https://pds-ppi.igpp.ucla.edu/data/JNO-J-JED-3-CDR-V1.0/"
    JEDI_DIR = DATA_DIR + "JEDI/"

    # Chunk duraction used when splitting downloads into smaller files
    _chunk_res = "15min"

    # JEDI detector labels
    _dets = ["090", "180", "270"]

    # --- PDS Index File ---
    # Contains file names and the time ranges for all JEDI products
    # Filtered to the HIERSESP (High Energy Resolution Electron Spectra) 
    #       files only
    # Loads from local cache if available, otherwise downloaded from 
    #       PDS and cached
    try:
        _jedi_index_df = pd.read_csv(DATA_DIR + "JEDI_INDEX.TAB")
    except:
        _jedi_index_df = pd.read_csv(
            "https://pds-ppi.igpp.ucla.edu/data/JNO-J-JED-3-CDR-V1.0/INDEX/INDEX.TAB"
        )
        _jedi_index_df.columns = _jedi_index_df.columns.str.strip()
        _jedi_index_df.to_csv(DATA_DIR + "JEDI_INDEX.TAB")

    # Keep only HIERSESP files
    _jedi_index_df = _jedi_index_df[_jedi_index_df["SID"].str.contains("HIERSESP")]
    # Parse start/stop times from the day-of-year format
    _jedi_index_df["START_TIME"] = pd.to_datetime(
        _jedi_index_df["START_TIME"], 
        format="%Y-%jT%H:%M:%S.%f"
    )
    _jedi_index_df["STOP_TIME"] = pd.to_datetime(
        _jedi_index_df["STOP_TIME"], 
        format="%Y-%jT%H:%M:%S.%f"
    )


    def __init__(
        self, 
        start_dt, 
        end_dt, 
        lower_energy=30, 
        upper_energy=1200, 
        num_en_bins=19, 
        pitch_angle_res=4.5, 
        t_res="1s", 
        use_approx_loss_cone_angle=True,
        max_intensity = 1e20
    ):
        """
        Initializes a JEDIData instance and fully processes the data

        Processing includes:
            _parse_JEDI_file: read raw JEDI data for each detectpr
            _calc_positional_parameters: comoute position, hemisphere 
                and loss cone angle via SPICE
            _make_pitch_angle_bins: define pitch angle grid
            _make_energy_bins: define log-spaced energy grid
            _create_datablock: rebins the intensity data into custom 
                time-energy-pitch angle cells in the form of a 3D numpy 
                array

        Parameters
        ----------
        start_dt : pandas.Timestamp
            Start time of the requested window
        end_dt : pandas.Timestamp
            End time of the requested window
        lower_energy : float, optional
            Lower bound of the output energy range (keV)
        upper_energy : float, optional
            Upper bound of the output energy range (keV)
        num_en_bins : int, optional
            Number of log-spaced output energy bins
        pitch_angle_res : float, optional
            Pitch angle bin width (degrees), adjusted if it does not 
            divide 180° evenly
        t_res : str, optional
            Time resolution as a pandas offset alias (e.g. "1s")
        use_approx_loss_cone_angle : bool, optional
            If True (default), use the approximate expression for the 
            loss cone angle from Mauk et al. (2017). If False, calculate 
            the loss cone angle explicitly from the model magnetic 
            field at Juno and Juno's magnetic footprint (slower)
        max_intensity : float, optional
            Maximum value of intensity to allow into the rebinned 
            datablock. Removes extremely high intensities that 
            complicate subsequent analyses
        """
        # Floor/ceil start and end times to the requested time 
        #       resolution to ensure neat time bins
        self.start_dt = start_dt.floor(t_res)
        self.end_dt = end_dt.ceil(t_res)
        self.lower_energy = lower_energy
        self.upper_energy = upper_energy
        self.num_en_bins = num_en_bins
        self.pitch_angle_res = pitch_angle_res
        self.t_res = t_res
        self._use_approx_loss_cone_angle = use_approx_loss_cone_angle
        self.max_intensity = max_intensity

        # Parse raw data and native energy bins for each detector
        self._raw_data = {}
        self._native_en_bins = {}
        for det in JEDIData._dets:
            self._parse_JEDI_file(det)

        # Build the uniform time grid
        self.time_bins = pd.date_range(self.start_dt, self.end_dt, freq=self.t_res).to_numpy()
        self.mean_times = self.time_bins[:-1] + pd.to_timedelta(self.t_res) / 2
        self.num_times = len(self.mean_times)
        self.et = sp.datetime2et(pd.to_datetime(self.mean_times))

        self._calc_positional_parameters()
        self._make_pitch_angle_bins()
        self._make_energy_bins()
        self._create_datablock()


    @staticmethod
    def _get_filepath_list(start_dt, end_dt):
        """
        Returns the paths (relative to JEDI_PDS_URL) for all JEDI 
        HIERSESP files that overlap the requested time window

        Parameters
        ----------
        start_dt : pandas.Timestamp
            Start time of the requested window
        end_dt : pandas.Timestamp
            End time of the requested window

        Returns
        -------
        filepath_list : list of str
            File paths relative to the JEDI_PDS_URL with the .LBL 
            extension replaced by .TAB
            Returns empty list if no files available for the time range
        """
        # Filter out files that end before start_dt or start 
        #       after end_dt
        time_mask = ~(
            (JEDIData._jedi_index_df["STOP_TIME"] <= start_dt) 
            | (JEDIData._jedi_index_df["START_TIME"] >= end_dt)
        )

        # Extract file paths and change the file extension
        filepath_list = (
            JEDIData._jedi_index_df[time_mask]["FILE_SPECIFICATION_NAME"]
            .str.strip()
            .values
        )
        filepath_list = [file[:-4] + ".TAB" for file in filepath_list]
        
        return filepath_list


    @staticmethod
    def _get_chunked_save_filepath(save_filepath, chunk_start_dt, chunk_end_dt):
        """
        Constructs a local save path for a 15-minute data chunk by 
        inserting a "HHMM_HHMM" time label into the file name

        For example, a file named "JED_..._2017139_V03.TAB" covering 
        06:30 to 06:45 becomes "JED_..._2017139_0630_0645_V03.TAB"

        Parameters
        ----------
        save_filepath : str
            Base local file path
        chunk_start_dt : pandas.Timestamp
            Start time of the chunk window
        chunk_end_dt : pandas.Timestamp
            End time of the chunk window

        Returns
        -------
        chunked_save_filepath : str
            Modified file path with the chunk time label inserted after 
            the year/day-of-year label
        """
        # Set the labels that will be inserted into the file name
        chunk_start_label = f"{chunk_start_dt.hour:02}{chunk_start_dt.minute:02}"
        chunk_end_label = f"{chunk_end_dt.hour:02}{chunk_end_dt.minute:02}"

        # Insert the time label immediately after the 
        #       year/day-of-year label
        save_filepath_split = save_filepath.split("_")
        save_filepath_split.insert(-1, f"{chunk_start_label}_{chunk_end_label}")
        chunked_save_filepath = "_".join(save_filepath_split)

        return chunked_save_filepath


    @staticmethod
    def downloadJEDILocal(start_dt, end_dt, overwrite=False, verbose=False):
        """
        Downloads all JEDI HIERSESP files needed to cover the 
        time range from the PDS and saves them locally as 15-minute 
        chunks to JEDI_DIR

        Each PDS file is downloaded once and split into 15-minute 
        chunks. Only chunks that overlap the requested time window are 
        saved. The chunking strategy keeps file sizes small and prevents 
        duplicate storage when overlapping time windows are requested 
        later. Files that are already present locally are skipped unless
        ``overwrite=True`` is specified.

        Note: Download is expected to take a while (each day of raw data 
        is approximately 2-3 GB)

        Parameters
        ----------
        start_dt : pandas.Timestamp
            Start time of the requested window
        end_dt : pandas.Timestamp
            End time of the requested window
        overwrite : bool, optional
            If True, re-downloads and recompiles files even if the file 
            already exists locally. Default if False
        verbose : bool, optional
            If True, prints progress messages during download and 
            chunking. Default is False
        """
        # Round requested time window to chunk boundaries before 
        #       querying the index file
        start_dt_round = start_dt.floor(JEDIData._chunk_res)
        end_dt_round = end_dt.ceil(JEDIData._chunk_res)

        pds_filepath_list = JEDIData._get_filepath_list(start_dt_round, end_dt_round)

        if len(pds_filepath_list) == 0:
            raise Exception("Files not present on PDS. Download Failed.")

        for pds_filepath in pds_filepath_list:
            downloaded = False  # Track if the file has been downloaded yet

            url = JEDIData.JEDI_PDS_URL + pds_filepath
            filename = pds_filepath.split("/")[-1]
            save_filepath = JEDIData.JEDI_DIR + filename
            
            # Iterate over all 15-minute chunks in requested window
            time_bins = pd.date_range(start_dt_round, end_dt_round, freq=JEDIData._chunk_res)
            for i in range(len(time_bins) - 1):
                # Skip chunk windows that do not correspond to the day 
                #       specified by the pds filepath
                if pds_filepath.split("_")[-2] != (
                    f"{time_bins[i].year:03}{time_bins[i].dayofyear:03}"
                ):
                    continue

                chunked_save_filepath = JEDIData._get_chunked_save_filepath(
                    save_filepath, time_bins[i], time_bins[i+1]
                )

                # Check if this chunk already exists locally
                if not overwrite:
                    if os.path.isfile(chunked_save_filepath):
                        if verbose:
                            print(
                                f"{chunked_save_filepath.split("/")[-1]} already downloaded. "
                                "Use 'overwrite' if new version requested."
                            )
                        continue

                # Download the full file on the first chunk that needs it
                if not downloaded:
                    if verbose:
                        print(f"Downloading {filename} from PDS...")
                    r = requests.get(url)
                    downloaded = True
                    
                    if verbose:
                        print("Download done. Now chunking and writing to files...")

                    # Split the download text into header and data rows
                    rows = r.text.split("\r\n")
                    header_rows = rows[:5]
                    header_rows = "\n".join(header_rows) + "\n"
                    rows = rows[5:-1]

                    # Extract SPICE ephemeris time values from the 
                    #       ETLookDirection column for use in 
                    #       chunk masking
                    ets = np.array([x.split(",") for x in rows])[:,2].astype("float")

                # Create and apply time mask to isolate rows belonging 
                #       to this 15-minute chunk
                chunk_start_et = sp.datetime2et(time_bins[i])
                chunk_end_et = sp.datetime2et(time_bins[i+1])
                mask = (ets > chunk_start_et) & (ets < chunk_end_et)
                masked = np.array(rows)[mask]
                write_str = header_rows + "\n".join(masked) + "\n"

                with open(chunked_save_filepath, "w") as f:
                    f.write(write_str)
                    f.close()


    def _parse_JEDI_file(self, det):
        """
        Reads locally stored JEDI chunk files for a single detector and 
        populates the raw intensity data and native energy bin values

        This method identifies the required chunk files via 
        _get_filepath_list, reads the files (skipping the header rows),
        drops the COUNTS and CPS columns (retaining only the number 
        intensity columns), refines the UTC timestamp using the more 
        precise SPICE ephemeris time, trims to the requested time 
        window, and stores the result in _raw_data

        Parameters
        ----------
        det :str 
            Label for the JEDI detector (see _dets variable)
       """
        start_dt_round = self.start_dt.floor(JEDIData._chunk_res)
        end_dt_round = self.end_dt.ceil(JEDIData._chunk_res)

        # Get all HIERSESP file paths and filter to those for 
        #       this detector
        pds_filepath_list = JEDIData._get_filepath_list(start_dt_round, end_dt_round)
        pds_filepath_list = [filepath for filepath in pds_filepath_list if det in filepath]
        save_filepath_list = [
            JEDIData.JEDI_DIR + pds_filepath.split("/")[-1] for pds_filepath in pds_filepath_list
        ]

        df_list = []
        for save_filepath in save_filepath_list:
            time_bins = pd.date_range(start_dt_round, end_dt_round, freq=JEDIData._chunk_res)
            for i in range(len(time_bins) - 1):
                # Skip chunk windows that do not correspond to the day 
                #       specified by the pds filepath
                if save_filepath.split("_")[-2] != (
                    f"{time_bins[i].year:03}{time_bins[i].dayofyear:03}"
                ):
                    continue

                chunked_save_filepath = JEDIData._get_chunked_save_filepath(
                    save_filepath, time_bins[i], time_bins[i+1]
                )

                try:
                    # Read only the header row to get column names
                    columns_df = pd.read_csv(chunked_save_filepath, nrows=0)
                except:
                    raise Exception(
                        f"{chunked_save_filepath.split("/")[-1]} data not downloaded. "
                        "Use JEDIData.downloadJEDILocal"
                    )

                # Drop raw COUNTS and CPS columns
                # Retain the number intensity (EXF) and metadata columns 
                col_drop = columns_df.columns.str.contains("COUNTS|CPS")
                col_list = columns_df.drop(columns_df.columns[col_drop], axis=1).columns.to_list()

                # Read the data, skipping the subheader
                # skipinitialspace set to True to account for leading 
                #       spaces before data
                df = pd.read_csv(
                    chunked_save_filepath, 
                    skiprows=range(1, 5), 
                    usecols=col_list, 
                    skipinitialspace=True
                )
                
                # Replace the UTC values with a refined value using the 
                #       higher precision version derived from SPICE 
                #       ephemeris time (adds milliseconds to UTC)
                df["UTC"] = sp.et2utc(df["ETLookDirection"], "ISOC", 3)
                df.index = pd.to_datetime(df["UTC"])

                # Trim to desired time window
                df = df[df.index.to_series().between(self.start_dt, self.end_dt)]

                if not df.empty:
                    df_list.append(df)

        df = pd.concat(df_list)
        self._raw_data[det] = df[df.index.to_series().between(self.start_dt, self.end_dt)]

        # Extract native energy bin bounds for this detector
        # Columns 16-446 (every third) hold the per-telescope 
        #       per-channel energy bounds
        # Row 0 is lower bound, Row 1 is upper bound
        en_bins = pd.read_csv(chunked_save_filepath, nrows=2)
        self._native_en_bins[det] = en_bins[en_bins.columns[16:446:3]]

        # Extract metadata from the datafile (contains information about 
        #       coordinate frame and penetrating radiation correction)
        with open(chunked_save_filepath) as file:
            metadata_row = file.readlines()[4].strip()
            self._metadata = dict([keyword.split(":") for keyword in metadata_row.split(",")])


    def _calc_positional_parameters(self):
        """
        Computes Juno's position, radial distance, hemisphere, 
        magnetic field-line direction, and loss cone angle

        Position is calculated via SPICE. The hemisphere is determined 
        from Juno's position above/below the planetocentric equator. The 
        hemisphere and field-line direction labels are set to None if 
        Juno crosses the planetocentric equator during the time window

        The loss cone angle is either calculated approximately from 
        _get_approx_loss_cone_angle, or exactly from 
        _get_exact_loss_cone_angle        
        """
        [state, ltime] = sp.spkezr("Juno", self.et, "IAU_Jupiter", "None", "Jupiter")
        self.position = np.array(state)[:,:3].T
        self.radius_rj = np.sqrt((self.position**2).sum(axis=0)) / 71492

        # Determine hemisphere from the z-component (height above the 
        #       jovicentric equator)
        # The magnetic field points away from Jupiter in the northern 
        #       hemisphere and towards Jupiter in the southern
        z = self.position[2]
        if (z > 0).all():
            self.hemi = "N"
            self.up_direction = "parallel"
            self.down_direction = "antiparallel"
            self.para_direction = "up"
            self.anti_direction = "down"
        elif (z < 0).all():
            self.hemi = "S"
            self.up_direction = "antiparallel"
            self.down_direction = "parallel"
            self.para_direction = "down"
            self.anti_direction = "up"
        else:
            # Trajectory crosses the magnetic equator; labels 
            #       are undefined
            self.hemi = None
            self.up_direction = None
            self.down_direction = None
            self.para_direction = None
            self.anti_direction = None

        if self._use_approx_loss_cone_angle:
            self._get_approx_loss_cone_angle()
        else:
            self._get_exact_loss_cone_angle()


    def _get_approx_loss_cone_angle(self):
        """
        Calculates the loss cone angle from the approximate analytical 
        expression from Mauk et al. (2017; 
        https://doi.org/10.1002/2016GL072286):
                arcsin(R^(-3/2)) 
        where R is the radius in jovian radii
        """
        self.loss_cone_angle = np.degrees(np.arcsin(self.radius_rj**(-3/2)) % (2*np.pi))


    def _get_exact_loss_cone_angle(self):
        """
        Calculates the loss cone angle exactly from the ratio of the 
        magnetic field value at Juno and at the footprint:
                arcsin(sqrt(B_juno / B_foot))
        The magnetic field is calculated via the JRM33 model at both 
        locations. The approximation from _get_approx_loss_cone_angle is 
        often sufficiently accurate and is used by default
        """
        # Calculate the magnetic filed magnitude at Juno
        B_juno = np.array(jm.Internal.Field(*self.position / 71492))
        B_juno_mag = np.sqrt((B_juno**2).sum(axis=0))

        # Calculate the magnetic field magnitude at Juno's footprint
        interp_dict = utils.interpFootprint(self.mean_times)
        B_foot = np.array(jm.Internal.Field(
            interp_dict["interp_x"], 
            interp_dict["interp_y"], 
            interp_dict["interp_z"]
        ))
        B_foot_mag = np.sqrt((B_foot**2).sum(axis=0))

        self.loss_cone_angle = np.degrees(np.arcsin(np.sqrt(B_juno_mag / B_foot_mag)) % (2*np.pi))


    def _make_pitch_angle_bins(self):
        """
        Defines the pitch angle grid used for the datablock

        Bin edges run from 0° to 180° with spacing pitch_angle_res. If 
        the provided pitch_angle_res doesn't divide 180° evenly, it is 
        rounded up to the nearest value that evenly splits the pitch 
        angle range
        """
        # Round up to a pitch angle resolution that divides 180° evenly
        pitch_angle_res_new = 180 / ((180 / self.pitch_angle_res) // 1)
        if pitch_angle_res_new != self.pitch_angle_res:
            print(f"Changing pitch_angle resolution to {pitch_angle_res_new} deg")
            self.pitch_angle_res = pitch_angle_res_new

        self.pitch_angle_bins = np.arange(0, 181, self.pitch_angle_res)
        self.mean_pitch_angles = (self.pitch_angle_bins[1:] + self.pitch_angle_bins[:-1]) / 2
        self.num_pitch_angle_bins = len(self.mean_pitch_angles)


    def _make_energy_bins(self):
        """
        Defines the energy grid used for the datablock.

        Bin edges run from lower_energy to upper_energy in 
        logarithmically-spaced bins. The function sets the mean energy 
        using both the arithmetic and geometric mean. The arithmetic 
        mean is used in the discrete sums over energy. The geometric 
        mean is often useful for plotting bins in log-space
        """

        self.energy_bins = np.logspace(
            np.log10(self.lower_energy), 
            np.log10(self.upper_energy), 
            num=self.num_en_bins + 1
        )
        self.mean_energies = (self.energy_bins[1:] + self.energy_bins[:-1]) / 2
        self.geom_mean_energies = np.sqrt(self.energy_bins[1:]*self.energy_bins[:-1])
        self.delE = self.energy_bins[1:] - self.energy_bins[:-1]


    @staticmethod
    def _calc_bin_ratio(old_lows, old_highs, new_lows, new_highs):
        """
        Calculates the fraction of the native (old) energy bin that 
        overlaps with each custom (new) bin, for use in energy rebinning

        Intensity is assumed to be uniformly distributed in each native 
        bin, so the fraction of intensity that maps to a new bin is the 
        overlap width divided by the native bin width

        Parameters
        ----------
        old_lows : np.ndarray, shape (N_old,)
            Lower bounds of the native energy bins (keV)
        old_highs : np.ndarray, shape (N_old,)
            Upper bounds of the native energy bins (keV)
        new_lows : np.ndarray, shape (N_new,)
            Lower bounds of the custom energy bins (keV)
        new_highs : np.ndarray, shape (N_new,)
            Upper bounds of the custom energy bins (keV)

        Returns
        -------
        bin_fraction : np.ndarray, shape (N_old, N_new)
            bin_fraction[i, j] is the fraction of native bin i that 
            falls within output bin j
        """
        # Use broadcasting ([:,None]) to compute overlaps for all 
        #       (old bin, new bin) pairs simultaneously.
        # overlap_low[i, j] = max(old_lows[i], new_lows[j])
        overlap_low = np.maximum(old_lows[:,None], new_lows)
        # overlap_high[i, j] = max(old_highs[i], new_highs[j])
        overlap_high = np.minimum(old_highs[:,None], new_highs)

        # Calculate width of overlap (setting to 0 when 
        #       overlap_low > overlap_high)
        overlap_width = np.maximum(0, overlap_high - overlap_low)

        # Normalize by the native bin width to get fractional overlap
        bin_fraction = overlap_width / (old_highs - old_lows)[:,None]

        return bin_fraction


    def _create_datablock(self):
        """
        Assembles the 3D intensity datablock by binning all detector and 
        telescope data into the uniform time-energy-pitch angle grid
        
        For each detector in _dets and each of its six telescopes:
          - Raw timestamps are mapped to time bins via np.searchsorted
          - Telescope pitch angle values are mapped to pitch angle bins
          - Native energy channels are rebinned into the custom 
            log-spaced bins using _calc_bin_ratio and matrix 
            multiplication (equivalent to a weighted sum over native 
            bins)
          - Rebinned intensities are accumulated into intensity_sum and 
            a count array number_sum to handle multiple measurements 
            per bin

        The final datablock is the element-wise ratio 
                intensity_sum / number_sum
        with NaN values where ``number_sum == 0``
        """
        # Initilize the intensity and count arrays with the 
        #       time-energy-pitch angle bins
        intensity_sum = np.zeros((self.num_times, self.num_en_bins, self.num_pitch_angle_bins))
        number_sum    = np.zeros((self.num_times, self.num_en_bins, self.num_pitch_angle_bins))

        for det in JEDIData._dets:
            raw = self._raw_data[det]

            # Map timestamps to time bin indices
            # searchsorted returns the insertion index, and subtracting 
            #       1 returns the bin the timestamp falls in (ex: 
            #       bin 0 = [time_bins[0],time_bins[1]])
            # t_bin_idxs shape (num_times_det,)
            t_bin_idxs = np.searchsorted(self.time_bins, raw.index) - 1

            for tel in range(6):
                # Map telescope's pitch angle to pitch angle bin indices
                # pa_bin_idxs shape (num_times_det,)
                pa_time_series = raw[f"Pitch_Angle_T{tel}"]
                pa_bin_idxs = np.searchsorted(self.pitch_angle_bins, pa_time_series) - 1  

                # Extract intensity columns and native energy bounds for
                #       this telescope
                # convert tel_intensities
                # tel_intensities shape (num_times_det, num_native_en_bins)
                tel_intensities = raw.filter(regex=f"T{tel}EXF")

                tel_bins = self._native_en_bins[det].filter(regex=f"T{tel}EXF").T

                # Check column ordering is consistent between intensity 
                #       and energy bins
                # (a sorting step could be added here in the future if 
                #       this check fails)
                if (tel_bins.index != tel_intensities.columns).any():
                    raise Exception(
                        "Flux bin ordering error. Check bin labels for tel_bins and raw data."
                    )

                # Calculate the telescope's fraction overlap array
                # bin_fraction shape (num_native_en_bins, num_en_bins)
                bin_fraction = JEDIData._calc_bin_ratio(
                    tel_bins[0].values,
                    tel_bins[1].values,
                    self.energy_bins[:-1],
                    self.energy_bins[1:]
                )
            
                tel_intensities = tel_intensities.to_numpy()
                # Remove anamalously high intensities
                tel_intensities = np.where(
                    tel_intensities > self.max_intensity, 
                    0, 
                    tel_intensities
                )
                # delE_native shape (num_native_en_bins,)
                delE_native = (tel_bins[1] - tel_bins[0]).to_numpy()

                # Energy Rebinning
                # Multiply intensity by native bin width, project onto 
                #       new bins via the bin_fraction array, then 
                #       normalize by the new bin width
                # Result shape (num_times_det, num_en_bins)
                tel_intensities_new = (tel_intensities * delE_native) @ bin_fraction / self.delE

                # Accumulate rebinned intensities and sample counts into 
                #       output arrays
                # np.add.at is used to correctly handle multiple 
                #       measurements that fall in each time-pitch 
                #       angle bin
                for i in range(self.num_en_bins):
                    np.add.at(
                        intensity_sum[:,i,:], 
                        (t_bin_idxs, pa_bin_idxs), 
                        tel_intensities_new[:,i]
                    )
                    np.add.at(
                        number_sum[:,i,:], 
                        (t_bin_idxs, pa_bin_idxs), 
                        1
                    )

        # Divide accumulated intensity by sample count 
        # Yields NaN if no measurements in the cell
        with np.errstate(invalid="ignore"):
            self.datablock = intensity_sum / number_sum


    def datablock_filtered_LC(self, direction=""):
        """
        Returns a copy of the datablock with pitch angle cells outside 
        the requested loss cone diurection set to NaN

        A valid loss cone can either be specified by the alignment with 
        respect to the background magnetic field ("parallel" or 
        "antiparallel"), or physical directions with respect to the 
        planet ("up" or "down"). To filter for values outside both loss 
        cones, use "outside"

        Parameters
        ----------
        direction : str
            One of "up", "down", "parallel", "antiparallel", or 
            "outside". Default is "" which yields no filtering

        Returns
        -------
        datablock_filtered : np.ndarray, 
                             shape (num_times, num_en_bins, 
                             num_pitch_angle_bins)
            Datablock with pitch angles not in desired direction set as 
            NaN, or the unmodified datablock if no direction given
        """
        if not direction:
            return self.datablock

        # Resolve the physical direction ("up"/"down") into a direction 
        #       relative to the magnetic field 
        #       ("parallel"/"antiparallel")
        if direction == "up":
            direction = self.up_direction.lower()
        elif direction == "down":
            direction = self.down_direction.lower()
        
        # Build a boolean mask of shape (num_times, 1, 
        #       num_pitch_angle_bins) and repeat over the energy axis. 
        #       True when pitch angle is in the desired direction
        if direction == "parallel":
            # Parallel loss cone: pitch angle < loss cone angle
            filter = self.mean_pitch_angles[None,None,:] < self.loss_cone_angle[:,None,None]
        elif direction == "antiparallel":
            # Antiparallel loss cone: pitch angle > 180° - loss cone angle
            filter = self.mean_pitch_angles[None,None,:] > 180 - self.loss_cone_angle[:,None,None]
        elif direction == "outside":
            # Outisde of loss cone: 
            #   loss cone angle < pitch angle < 180° - loss cone angle
            filter = (
                (self.mean_pitch_angles[None,None,:] > self.loss_cone_angle[:,None,None]) 
                & (self.mean_pitch_angles[None,None,:] < 180 - self.loss_cone_angle[:,None,None])
            )
        filter = filter.repeat(self.num_en_bins, axis=1)

        # Set values outside of direction to NaN
        datablock_filtered = np.where(filter, self.datablock, np.nan) 

        return datablock_filtered


    def to_energy_dist(self, **kwargs):
        """
        Averages the datablock over pitch angle bins to obtain the 
        energy distribution (time, energy)

        Parameters
        ----------
        **kwargs
            Optional arguments passed to datablock_filtered_LC 
            (e.g. direction)

        Returns
        -------
        np.ndarray, shape (num_times, num_en_bins)
            Mean differential intensity in each time-energy cell, 
            averaged over the selected pitch angle direction (NaN 
            ignored)
        """
        return np.nanmean(self.datablock_filtered_LC(**kwargs), axis=2)


    def to_pitch_angle_dist(self, unit, **kwargs):
        """
        Integrates the datablock over energy channels to produce a pitch 
        angle distribution (time, pitch angle) in the selected unit

        Conversion from the native keV/s/cm^2/sr to mW/m^2/sr (energy 
        intensity; EI) uses the conversion factors:
                1 keV/s = 10^6 e mW/s
                1/cm^2 = 10^4 1/m^2
                1 keV/s/cm^2 = 10^10 e mW/m^2 = 1.602e-9 mW/m^2
        where e is the value of the elementary charge in Coulombs

        Parameters
        ----------
        unit : str
            Output intensity unit:
            "DNI"
                Differential number intensity (1/cm^2/s/sr/keV); sum of 
                intensity weighted by the width of the energy bin, 
                normalized by the total energy range
            "DEI"
                Differential energy intensity (1/cm^2/s/sr); sum of 
                intensity weighted by the width of the energy bin and 
                the mean energy of the bin, normalized by the total 
                energy range
            "EI"
                Energy intensity (mW/m^2/sr); sum of intensity weighted 
                by the width of the energy bin and the mean energy of 
                the bin, converted to SI units
            DNI, DEI, EI
        **kwargs
            Optional arguments passed to datablock_filtered_LC (e.g. 
            direction)

        Returns
        -------
        np.ndarray, shape (num_times, num_pitch_angle_bins)
            Intensity (in the requested unit) in each time-pitch angle 
            cell (NaN ignored)
        """
        datablock_filtered = self.datablock_filtered_LC(**kwargs)

        if unit == "DNI":
            # Differential number intensity: bin-width-weighted sum 
            #       normalized by total energy range
            return (
                datablock_filtered 
                * self.delE[None, :, None]
            ).sum(axis=1) / self.delE.sum()
        
        elif unit == "DEI":
            # Differential energy intensity: same as DNI but also 
            #       weighted by mean energy
            return (
                datablock_filtered 
                * self.mean_energies[None, :, None] 
                * self.delE[None, :, None]
            ).sum(axis=1) / self.delE.sum()
        
        elif unit == "EI":
            # Energy intensity: same as DEI but not normalized by the 
            #       total energy range AND is converted to SI units
            return 1e10 * e * (
                datablock_filtered 
                * self.mean_energies[None, :, None] 
                * self.delE[None, :, None]
            ).sum(axis=1)
        else:
            Exception("Use valid unit for intensity.")


    def calc_mean_intensity(self, require_full_LC = True, **kwargs):
        """
        Calculates the mean energy intensity in the upward and downward 
        loss cones, and outside the loss cones using a 
        solid-angle-weighted average

        This method is analagous to the method in Mauk+2017, using the 
        average intensity in each loss cone region (i.e. normalizes by 
        the size of the loss cone region)

        Parameters
        ----------
        require_full_LC : bool, optional
            If True (default), calls _mask_by_loss_cone to set intensity 
            values to NaN at times with poorly sampled loss cones
        **kwargs 
            Optional arguments passed to _mask_by_loss_cone (e.g. 
            full_LC_thres)

        Returns
        -------
        intensity_dict : dict
            np.ndarray of shape (num_times,) containing the mean 
            intensity for each of three directions ("up", "down", and 
            "outside")
        """
        lower_pitch_angles = np.radians(self.pitch_angle_bins[:-1])
        upper_pitch_angles = np.radians(self.pitch_angle_bins[1:])

        # cos_term accounts for the sin(theta) term in the spherical 
        #       area differential element
        cos_term = np.cos(lower_pitch_angles) - np.cos(upper_pitch_angles)

        # Calculate pitch angle distribution in each direction
        filtered_up = self.to_pitch_angle_dist(unit="EI", direction="up")
        filtered_down = self.to_pitch_angle_dist(unit="EI", direction="down")
        filtered_outside = self.to_pitch_angle_dist(unit="EI", direction="outside")

        # Calculate the solid angle of each direction by summing 
        #       cos_term only where the data is valid (np.sign is 1 for 
        #       positive values, and 0 for NaN/zero)
        size_up = 2 * np.pi * np.nansum(np.sign(filtered_up) * cos_term[None, :], axis=1)
        size_down = 2 * np.pi * np.nansum(np.sign(filtered_down) * cos_term[None, :], axis=1)
        size_outside = 2 * np.pi * np.nansum(np.sign(filtered_outside) * cos_term[None, :], axis=1)

        # Calculate the solid-angle-weighted mean intensity in each 
        #       direction. Times with an empty up/down loss cone will 
        #       have size_up/down = 0. The divide-by-zero error is 
        #       suppressed using np.errstate and returns a NaN
        with np.errstate(invalid="ignore"):
            intensity_up = 2 * np.pi * (
                np.nansum(filtered_up * cos_term[None, :], axis=1) / size_up
            )
            intensity_down = 2 * np.pi * (
                np.nansum(filtered_down * cos_term[None, :], axis=1) / size_down
            )
            intensity_outside = 2 * np.pi * (
                np.nansum(filtered_outside * cos_term[None, :], axis=1) / size_outside
            )

        # Mask times with undersampled loss cones using 
        #       _mask_by_loss_cone if requested
        if require_full_LC:
            intensity_up = self._mask_by_loss_cone(intensity_up, direction="up", **kwargs)
            intensity_down = self._mask_by_loss_cone(intensity_down, direction="down", **kwargs)

        #replaces mean intensity with NaN if the value is anamalously high
        # df_para = df_para.where(df_para < 1e4, np.nan)
        # df_anti = df_anti.where(df_anti < 1e4, np.nan)
        # df_trap = df_trap.where(df_trap < 1e4, np.nan)

        intensity_dict = {
            "up":intensity_up,
            "down":intensity_down,
            "outside":intensity_outside
        }

        return intensity_dict
    

    def _mask_by_loss_cone(self, a1, direction="", full_LC_thres = 2/3):
        """
        Sets elements of an input array to NaN where the specified loss 
        cone direction is undersampled

        A time step is considered undersampled if the number of valid 
        pitch angle bins within the loss cone is less than:
                full_LC_thres * (loss_cone_angle / pitch_angle_res)
        i.e. that the fraction of sampled pitch angle bins in the loss 
        cone is less than than full_LC_thres

        Parameters
        ----------
        a1 : np.ndarray, shape (num_times,)
            Array of values to mask
        direction : str 
            Loss cone direction to evaluate. Same options as in 
            datablock_filtered_LC. If "" (default), no masking is done.
        full_LC_thres : float, optional
            Minimum fraction of the loss cone bins that must be sampled 
            for a time to be considered valid. Default is 2/3
            
        Returns
        -------
        np.ndarray, shape (num_times,)
            Input array with undersample time steps set to NaN
        """
        # Minimum number of pitch angle bins required to consider the 
        #       loss cone adequately sampled
        min_num_bins = self.loss_cone_angle / self.pitch_angle_res * full_LC_thres

        if not direction:
            # No direction specified: return the unmodified array
            mask = np.full(self.mean_times.shape, True)
        elif direction == "up":
            # Count valid (non-NaN) pitch angle bins up upward loss 
            #       cone and compare to the minimum threshold
            mask = np.nansum(
                np.sign(self.to_pitch_angle_dist(unit="DNI", direction="up")), axis=1
            ) > min_num_bins
        elif direction == "down":
            # Count valid (non-NaN) pitch angle bins up downward loss 
            #       cone and compare to the minimum threshold
            mask = np.nansum(
                np.sign(self.to_pitch_angle_dist(unit="DNI", direction="down")), axis=1
            ) > min_num_bins
        
        return np.where(mask, a1, np.nan)




# ##############################################################
# ## Functions for Saving Processed Time-Pitch Angle Distribution Data ##
# ##############################################################


# def _validate_distribution_args(type, unit, t_res):
#     valid_types = {"AllEnergy", "DownEnergy", "UpEnergy", "PitchAngle"}
#     valid_units = {"DNI", "DEI", "EI"}
#     valid_t_res = {"1s", "5s", "30s"}

#     if type not in valid_types:
#         raise ValueError(f"Type must be one of {valid_types}")
#     if unit not in valid_units:
#         raise ValueError(f"Unit must be one of {valid_units}")
#     if t_res not in valid_t_res:
#         raise ValueError(f"t_res must be one of {valid_t_res}")


# def saveTimeDistribution(df, type, unit, t_res, event):
#     """
#     Saves a dataframe to a local file to be imported later to save processing time

#     Parameters
#     ----------
#         df [pd.DataFrame]: dataframe that is already processed into a product that can be plotted
#                 (such as sub-dictionary of datablock or output of convertDataBlockTo*Dist)
#         type [str]: Energy distribution (with direction) or pitch angle distribution ("AllEnergy", "DownEnergy", "UpEnergy", "PitchAngle")
#         unit [str]: Differential number intensity (1/cm^2/s/sr/keV; DNI) or differential energy intensity (keV/cm^2/s/sr/keV; DEI)
#         t_res [str]: time resolution ("1s", "5s", "30s")
#         event [str]: label use for the event
#     """
#     _validate_distribution_args(type, unit, t_res)
    
#     if event[:2] == "PJ":
#         event = event[2:]

#     df.to_csv(JEDI_DIR + f"PJ{event}_{type}_{unit}_{t_res}.dat", sep=",", index=True, na_rep="nan")


# def importTimeDistribution(type, unit, t_res, event):
#     """
#     Inports a dataframe from a local file created by saveTimeDistribution

#     Parameters
#     ----------
#         type [str]: Energy distribution (with direction) or pitch angle distribution ("AllEnergy", "DownEnergy", "UpEnergy", "PitchAngle")
#         unit [str]: Differential number intensity (1/cm^2/s/sr/keV; DNI) or differential energy intensity (keV/cm^2/s/sr/keV; DEI)
#         t_res [str]: time resolution ("1s", "5s", "30s")
#         event [str]: label use for the event
#     """
#     _validate_distribution_args(type, unit, t_res)
    
#     if event[:2] == "PJ":
#         event = event[2:]

#     df = pd.read_csv(JEDI_DIR + f"PJ{event}_{type}_{unit}_{t_res}.dat", sep=",")
#     df["time"] = pd.to_datetime(df["time"])
#     df = df.set_index("time")
#     df.columns = df.columns.astype("float")

#     return df
