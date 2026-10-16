import numpy as np
import pandas as pd
from scipy.stats import gmean, gstd

from . import jm    #JupiterMag
from . import sp    #SpiceyPy

from . import utils


"""

"""

class AccelIntervals:
    """
    Overview

    Class Attributes
    ----------------


    Instance Attributes
    -------------------

    """
    _good_zone_df = utils._zones_df[utils._zones_df["Confidence"] >= 1].reset_index(drop=True)
    _zone_intervals = pd.Series(
        _good_zone_df["Zone"].to_list(),
        index=pd.IntervalIndex.from_arrays(
            _good_zone_df["Start"], _good_zone_df["End"], closed="left"
        )
    )


    def __init__(self, jedi, pen_min=80, pen_max=160, snr_thres=3):
        """
        
        """
        # Set parameters from input
        self.jedi = jedi
        self._pen_min = pen_min
        self._pen_max = pen_max
        self._snr_thres = snr_thres

        # if self.jedi.t_res != "1s":
        #     raise Exception(
        #         "Provide a JEDIData instance with 1-second time resolution (`t_res=\"1s\"`)."
        #     )
        self.t_res_timedelta = pd.Timedelta(self.jedi.t_res).to_numpy()

        ### Zone identification and zone filters
        # Assign a zone to each time
        idx = AccelIntervals._zone_intervals.index.get_indexer(self.jedi.mean_times)
        self.zone = np.array(
            [AccelIntervals._zone_intervals.iloc[i] if i >= 0 else "None" for i in idx]
        )
        # Create filter for times in ZI and ZII
        self.in_main = (self.zone != "Dif") & (self.zone != "None")
        self.in_zi = (self.zone == "ZI")
        self.in_zii = (self.zone == "ZII")

        # Calculate downward energy distributions
        self.energy_dist_down = self.jedi.to_energy_dist(direction="down")

        ### Signal-to-noise (SNR) calculation
        # Energy-time array of SNR
        self.snr = np.sqrt(self.jedi.datablock_counts.sum(axis=2))
        # Boolean energy-time array of whether SNR exceeds threshold
        self.good_snr_array = self.snr >= self._snr_thres
        # Filter for times where minimum SNR satisfied for all energies
        self.good_snr = self.good_snr_array.all(axis=1)

        ### Creation and application of criteria
        # Create intensity threshold for the event
        self._create_threshold()
        # Apply the set of acceleration criteria to the range of times
        self.criteria = []
        self._snr_criterion()
        self._loss_cone_criterion()
        self._main_aurora_criterion()
        self._broadband_criterion()
        self._energetic_criterion()
        
        self.criteria = np.array(self.criteria)

        # Determine acceleration times and intervals
        self._create_intervals()

        # Merge adjacent intervals (within 1s) and remove intervals 
        #       adjacent to non-sampled time or non-auroral time
        self._merge_nearby_intervals()
        self._remove_near_unsampled()

        # Create summary dataframe for the identified intervals, 
        #       including derived products of the time range
        self._create_summary_dataframe()


    def _create_threshold(self):
        """
        Create intensity threshold from trapped Zone-I & Zone-II 
        distributions. Times must have a minimum signal-to-noise of 3 
        for every energy bin (when summed across all pitch angles).
        """
        # Process trapped energy-time intensity arrays
        self.energy_dist_out = self.jedi.to_energy_dist(direction="outside")
        # Replace zeros with NaNs to avoid errors in logarithms
        self.energy_dist_out = np.where(self.energy_dist_out <= 0, np.nan, self.energy_dist_out)

        # Calculate the geometric mean and geometric standard deviation 
        #       if there are any good SNR times in main aurora
        if (self.in_main & self.good_snr).any():
            self.geom_mean = gmean(
                self.energy_dist_out[self.in_main & self.good_snr], axis=0, nan_policy="omit"
            )
            self.geom_std = gstd(
                self.energy_dist_out[self.in_main & self.good_snr], axis=0, nan_policy="omit"
            )

            self.z = 1
            # Intensity threshold for at least one spectrum per interval
            self.int_thres = self.geom_mean * self.geom_std
            # Intensity threshold for surrounding spectra
            self.lower_int_thres = self.geom_mean * self.geom_std**(0.5)

        # Skip threshold intensity if no good SNR times in main aurora
        else:
            print("No times in main aurora with good SNR")
            nan_arr = np.full(self.jedi.mean_energies.shape, np.nan)
            self.geom_mean = nan_arr
            self.geom_std = nan_arr
            self.int_thres = nan_arr
            self.lower_int_thres = nan_arr
            self.z = 0


    def _snr_criterion(self):
        """
        Criterion 1: Require that the number of counts (summed over all 
        pitch angles) exceeds 9 counts for all energy bins. This 
        corresponds to a signal-to-noise (SNR) of 3 based on Poisson 
        counting statistics.

        This SNR criterion also classifies which times are used to 
        create the intensity threshold.

        This criterion also requires that the loss cone has  been
        """
        self.criteria.append(self.good_snr)


    def _loss_cone_criterion(self):
        """
        Criterion 2: Downward loss cone must be sampled.

        Times with NaNs in all energy bins are largely those where the 
        loss cone is unsampled for that time.
        """
        # Checks that at least one energy bin has been sampled to remove 
        #       times with all NaN values
        self.is_time_sampled = ~np.isnan(self.energy_dist_down).all(axis=1)
        # self.criteria.append(self.energetic_criterion & time_sampled)
        self.criteria.append(self.is_time_sampled)


    def _main_aurora_criterion(self):
        """
        Criterion 3: Measurement must be in main aurora (ZI/ZII)
        """
        self.criteria.append(self.in_main)


    def _broadband_criterion(
            self, 
            exclude_penetrators=True
    ):
        """
        Criterion 4: Time period does not have mono-energetic feature
        Uses criterion as discussed by Mauk et al. (2017, 2023)
            d/dE(I/E) > 0
        """
        # Monoenergetic feature where slope of PSD is greater than 0
        # I/E is proportional to PSD (in non-relativistic regime)
        I_E = self.energy_dist_down / self.jedi.mean_energies
        dI_E_dE = np.gradient(I_E, self.jedi.mean_energies, axis=1)
        # Array of shape energy_dist_down indicating bins identified as 
        #       monoenergetic (Use intensity threshold to minimize false 
        #       identification from low counts)
        self.mono_array = (dI_E_dE > 0) & (self.energy_dist_down > self.int_thres[None,:])

        if exclude_penetrators:
            # Exclude bins affected by penetrating radiation from 
            #       monoenergetic criterion
            non_pen_energies = (
                (self.jedi.mean_energies <= self._pen_min) 
                | (self.jedi.mean_energies >= self._pen_max)
            )
            self.mono_criterion = (self.mono_array[:,non_pen_energies]).any(axis=1)
        else:
            self.mono_criterion = self.mono_array.any(axis=1)
            
        self.mono_times = self.jedi.mean_times[self.mono_criterion]
        self.criteria.append(~self.mono_criterion)


    def _energetic_criterion(self):
        """
        Criterion 5: Measurement must either (a) surpass the higher 
        intensity threshold for all non-NaN energy bins, or (b) surpass 
        the lower intensity threshold for all non-NaN energy bins and be 
        adjacent to another accelerated spectrum (including a spectrum 
        fulfilling criterion 5b).

        At least one of the spectra in a string of spectra must surpass 
        the larger intensity threshold.
        """
        ### Comparison with intensity thresholds
        # Higher intensity threshold
        self.above_int_thres = ~(
            self.energy_dist_down < self.int_thres[None,:]
        ).any(axis=1)
        # Lower intensity threshold for neighboring spectra
        self.above_lower_int_thres = ~(
            self.energy_dist_down < self.lower_int_thres[None,:]
        ).any(axis=1)

        # 1d Boolean arrays indicating times that meet the other 
        #       criteria and the low or high intensity thresholds
        high = np.array(self.criteria).all(axis=0) & self.above_int_thres
        low = np.array(self.criteria).all(axis=0) & self.above_lower_int_thres

        n = len(self.jedi.mean_times)
        # Propagate the energetic criterion to neighboring times
        # Forward propagation: a time is energetic if it surpasses the 
        #       low threshold and the time before it is energetic (or it 
        #       surpasses the high threshold itself)
        # Fills in all neighboring (lower intensity) energetic times 
        #       after a high intensity time
        fwd = np.full(n, False)
        prev = False    # Initializes the previous value
        for i in range(n):
            prev = high[i] | (low[i] & prev)
            fwd[i] = prev

        # Backward propagation: Performs the same process for times 
        #       before a high intensity time
        bwd = np.full(n, False)
        prev = False
        for i in range(n - 1, -1, -1):
            prev = high[i] | (low[i] & prev)
            bwd[i] = prev

        # Energetic criterion is combination of the forward and backward 
        #       propagated times
        self.energetic_criterion = fwd | bwd
        self.criteria.append(self.energetic_criterion)

        # Saves boolean array for times exceeding the low intensity 
        #       threshold but not connected to higher intensity spectrum
        self.non_neighboring_times = self.jedi.mean_times[low & ~self.energetic_criterion]


    def _create_intervals(self):
        """
        Converts the list of times with energetic spectra (accel_times) 
        into an array (shape Nx2) providing the start and end times of a 
        continuous interval.
        """
        intervals = []
        accel_times = self.jedi.mean_times[self.criteria.all(axis=0)]

        if len(accel_times) > 0:
            # Array of indices indicating the last time of each 
            #       continuous interval
            interval_end_idx = np.flatnonzero(np.diff(accel_times) > self.t_res_timedelta)

            # Split accel_times into list of arrays containing 
            #       continuous times
            for interval_arr in np.split(accel_times, interval_end_idx + 1):
                # JEDIData timestamps are for the middle of a time bin, 
                #       so add and subtract half the sample spacing to 
                #       start and end time
                half_spacing = self.t_res_timedelta / 2
                intervals.append(
                    (interval_arr[0] - half_spacing, interval_arr[-1] + half_spacing)
                )

        # Convert list of intervals to an array (force the default shape 
        #       for events with no accelerated intervals)
        self.unmerged_intervals = np.array(intervals).reshape(-1,2).astype(dtype=np.datetime64)


    def _merge_nearby_intervals(self):
        """
        Merges the raw intervals from _create_intervals when they are 
        split by only a second of data.
        """
        merge = np.abs(np.roll(self.unmerged_intervals,shift=1) - self.unmerged_intervals)[:,0] <= np.timedelta64(1, "s")

        merging_intervals = self.unmerged_intervals.copy()

        for i in range(len(merge) - 1, -1, -1):
            if merge[i]:
                merging_intervals[i - 1, 1] = merging_intervals[i, 1]
        self.merged_intervals = merging_intervals[~merge]


    def _remove_near_unsampled(self):
        """
        Removes any merged intervals from _merge_nearby_intervals that 
        have a start or end time adjacent to a time (1) at which the loss 
        cone is entirely unsampled, or (2) outside the main aurora.
        """
        unsampled_times = self.jedi.mean_times[~self.is_time_sampled]
        start_near_nan = np.isin(self.merged_intervals[:,0] - self.t_res_timedelta / 2, unsampled_times)
        end_near_nan = np.isin(self.merged_intervals[:,1] + self.t_res_timedelta / 2, unsampled_times)
        any_near_nan = start_near_nan | end_near_nan

        non_main_times = self.jedi.mean_times[~self.in_main]
        start_near_non_main = np.isin(self.merged_intervals[:,0] - self.t_res_timedelta / 2, non_main_times)
        end_near_non_main = np.isin(self.merged_intervals[:,1] + self.t_res_timedelta / 2, non_main_times)
        any_near_non_main = start_near_non_main | end_near_non_main

        self.rejected_intervals = self.merged_intervals[(any_near_nan | any_near_non_main),:]
        self.accel_intervals = self.merged_intervals[~(any_near_nan | any_near_non_main),:]


    @staticmethod
    def get_perp_length(start_dt, end_dt):
        """
        Calculate the perpendicular (to the background magnetic field) component of the distance between the positions at the start and end times

        Parameters
        ----------
        start_dt : datetime-like
            Start time of the desired interval
        end_dt : datetime-like
            End time of the desired interval
        """
        # Store the start and end time in an array and ensure the datetimes 
        #       have timezone info
        interval = np.array([start_dt, end_dt])
        ets = sp.datetime2et(pd.to_datetime(interval))

        # Calculate position for both times from SpiceyPy
        [pos, ltime] = sp.spkpos("Juno", ets, "IAU_Jupiter", "None", "Jupiter")
        # Displacement between start and end position
        disp = pos[1] - pos[0]
        # Mean position
        mean_pos = pos[0] + disp / 2

        # Magnetic field at mean position 
        Bint = np.array(jm.Internal.Field(*mean_pos / 71492))
        Bext = np.array(jm.Con2020.Field(*mean_pos / 71492))
        Bfield = (Bint + Bext)[:, 0]
        Bhat = Bfield / np.sqrt((Bfield**2).sum())

        # Displacement vector perpendicular to magnetic field
        disp_perp = disp - (disp * Bhat).sum() * Bhat

        return np.sqrt((disp_perp**2).sum())


    def _create_summary_dataframe(self):
        """
        
        """
        if len(self.accel_intervals) > 0:
            # Create dataframe from the start and end times
            intervals_df = pd.DataFrame(self.accel_intervals, columns=["start", "end"])
            # Append the time extent (in s) of the intervals
            intervals_df["dt"] = [dt.seconds for dt in intervals_df["end"] - intervals_df["start"]]
            # Calculate and append the perpendicular scale length
            intervals_df["d_perp"] = intervals_df.apply(
                lambda x: AccelIntervals.get_perp_length(x.start, x.end), axis=1
            )
            # Append the estimated perpendicular wavelength
            intervals_df["k_perp"] = 2 * np.pi / intervals_df["d_perp"]

        else:
            intervals_df = pd.DataFrame(columns=["start","end","dt","d_perp","k_perp"])

        self.intervals_df = intervals_df

        

