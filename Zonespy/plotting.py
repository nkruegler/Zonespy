import os
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.ticker import AutoMinorLocator, FuncFormatter

from . import MODULE_DATA_DIR
from . import utils


"""
Utility functions for creating matplotlib stackplots of auroral zone data.

Includes functions/classes for:
  - Applying a consistent grid across all subplots
  - Formatting datetime x-axes with additional ephemeris rows
  - Appending magnetic footprint coordinates and angle from auroral 
    oval (alpha) as additional ephemeris rows
  - Setting minor tick density to one appropriate to major tick cadence
  - Adding auroral zone intervals as shaded regions on desired subplots
"""

# --- Auroral zone boundaries ---
# Determined from electron intensities and field-aligned currents
# See Mauk et al. (2020) and Sulaiman et al. (2022) for details
# Loaded once at module level and optionally reloaded at call time in 
#       addZoneHighlights to reflect live edits
_zones_df = pd.read_csv(os.path.join(MODULE_DATA_DIR, "zone_boundary.dat"), sep=r"\s+")
_zones_df["Start"] = pd.to_datetime(_zones_df["Start"], format="%Y-%m-%dT%H:%M:%S")
_zones_df["End"] = pd.to_datetime(_zones_df["End"], format="%Y-%m-%dT%H:%M:%S")

# Color map for zone labels; used in addZoneHighlights
zone_c_dict = {"Dif":"k", "ZI":"tab:blue", "ZII":"tab:orange"}

# Hatch pattern to indicate zone confidence (3=highest, 0=lowest)
# Used in addZoneHighlights
hatch_dict = {3:None, 2:"///", 1:"XXX", 0:"***"}


def setGrid(fig):
    """
    Applies a major and minor grid to every subplot in a figure

    Major grid lines are drawn at full opacity; minor grid lines are 
    drawn at reduced opacity. Grid lines are placed below plotted data 
    with set_axisbelow

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure whose axes should have the grid applied
    """
    for axis in fig.get_axes():
        axis.set_axisbelow(True)
        axis.grid(which="major", alpha=0.9)
        axis.grid(which="minor", alpha=0.2)


class MultiLineDateFormatter(ConciseDateFormatter):
    """
    A ConciseDateFormatter subclass that supports appending extra 
    annotation rows below each tick label after the formatter is constructed.

    Each extra row is defined by a callable that maps a matplotlib date 
    float to a value to display. Rows are added via add_line and are 
    rendered in the order they are added

    This is called by the general function setDateFormatter and is used 
    by addLatLonLabel and addAlphaLabel
    """

    def __init__(self, locator, **kwargs):
        super().__init__(locator, **kwargs)
        self._extra_lines: list[callable] = []

    def add_line(self, fn, fmt = ".02f"):
        """
        Append an extra label row to each tick

        Parameters
        ----------
        fn : callable
            Function mapping a matplotlib data float to a numeric value 
            to be displayed on the new row.
        fmt : str, optional
            Python format string for the numeric value. Default is ".02f"

        Returns
        -------
        self : MultiLineDateFormatter
            Returns self to allow method chaining
        """
        self._extra_lines.append((fn, fmt))
        return self

    def format_ticks(self, values):
        """
        Formats all tick labels, appending any extra rows

        Overrides ConciseDateFormatter.format_ticks to concatenate
        the base date string with one additional line per callable, 
        separated by newline characters.

        Parameters
        ----------
        values : list of float
            Matplotlib date floats for each tick position
        
        Returns
        -------
        values_new : list of str
            Formatted tick label strings, one per tick
        """
        base_values = super().format_ticks(values)

        # If no extra rows have been registered, return the base 
        #       labels unchanged
        if not self._extra_lines:
            return base_values

        values_new = []
        for val, base_val in zip(values, base_values):
            # Build a list of strings: base date label + one 
            #       string per extra row
            parts = [base_val] + [f"{fn(val):{fmt}}" for fn, fmt in self._extra_lines]
            values_new.append("\n".join(parts))
        return values_new


def setDateFormatter(axis):
    """
    Applies a MultiLineDateFormatter to the x-axis of the given axis

    Only needs to be called on one axis when ``sharex=True`` is used
    
    The returned formatter can be passed to addLineLabel, 
    addLatLonLabel, or addAlphaLabel to append further rows

    Parameters
    ----------
    axis : matplotlib.axes.Axes 
        Axes with a datetime x-axis to be reformatted

    Returns
    -------
    formatter : MultiLineDateFormatter 
        Formatter instance applied to the axis
    """
    formatter = MultiLineDateFormatter(AutoDateLocator())
    axis.xaxis.set_major_formatter(formatter)

    return formatter


def addLineLabel(formatter, axis, fn, label, **kwargs):
    """
    Adds a new line of data to the x-axis ticks and appends the 
    corresponding row label to the x-axis label string

    Wraps MultiLineDateFormatter.add_line and updates the axis label
    in one call, keeping the tick rows and label rows in sync. Only 
    needs to be called on one axis when ``sharex=True`` is used

    Parameters
    ----------
    formatter : MultiLineDateFormatter 
        Formatter instance managing the tick labels for this axis
    axis : matplotlib.axes.Axes
        Axes whose x-axis label and tick labels should be updated
    fn : callable
        Function mapping a matplotlib data float to a numeric value 
        to be displayed on the new row.
    label : str 
        Text to append to the x-axis label (on a new line)
    **kwargs 
        Optional arguments passed to add_line (e.g. fmt=".03f")
    """
    formatter.add_line(fn, **kwargs)
    axis.set_xlabel(axis.get_xlabel() + "\n" + label)


def addLatLonLabel(formatter, axis):
    """
    Appends footprint latitude and longitude rows to the x-axis labels
    via addLineLabel

    Footprints are calculated at each major tick using 
    utils.interpFootprint. If the spacing between minor ticks is wide 
    enough, footprint coordinates are also added for minor tick labels. 
    Only needs to be called on one axis when ``sharex=True`` is used
    
    Parameters
    ----------
    formatter : MultiLineDateFormatter 
        Formatter instance managing the tick labels for this axis
    axis : matplotlib.axes.Axes
        Axes whose x-axis label and tick labels should be updated
    """
    # Interpolate footprint position at major tick locations
    xticks = axis.get_xticks()
    interp_dict = utils.interpFootprint(mdates.num2date(xticks))

    # Build lookup dictionary to serve as the callable for addLineLabel
    lat_dict = dict(zip(xticks, interp_dict["interp_lats"]))
    lon_dict = dict(zip(xticks, interp_dict["interp_lons"]))

    addLineLabel(formatter, axis, lambda x: lat_dict[x], "Footprint Lat. (deg)", fmt=".01f")
    addLineLabel(formatter, axis, lambda x: lon_dict[x], "Footprint Lon. (deg)", fmt=".01f")

    # --- Minor tick labels ---
    # Add rows for minor ticks if there is display space
    # Threshold of 45 display units was determined to work effectively
    minor_tick_spacing = np.min(np.diff(axis.get_xticks(minor=True)))   # matplotlib date units
    xlim_width = np.diff(axis.get_xlim())[0]    # matplotlib date units
    xaxis_bbox_width = axis.bbox.width      # display units

    if minor_tick_spacing / xlim_width * xaxis_bbox_width > 45:
        xticks_m = axis.get_xticks(minor=True)
        interp_dict_m = utils.interpFootprint(mdates.num2date(xticks_m))
        lat_dict_m = dict(zip(xticks_m, interp_dict_m["interp_lats"]))
        lon_dict_m = dict(zip(xticks_m, interp_dict_m["interp_lons"]))
        minor_formatter = lambda x, pos: f"\n{lat_dict_m[x]:.01f}\n{lon_dict_m[x]:.01f}"
        axis.xaxis.set_minor_formatter(FuncFormatter(minor_formatter))
        axis.xaxis.set_tick_params(which="minor", pad=5)


def addAlphaLabel(formatter, axis):
    """
    Appends angle from the auroral oval (alpha) to the x-axis labels
    via addLineLabel

    Alpha is calculated at each major tick with utils.calcAngleFromOval
    (from the footprints obtained via utils.interpFootprint). If the 
    spacing between minor ticks is wide enough, alpha values are also 
    added for minor tick labels. Only needs to be called on one axis 
    when ``sharex=True`` is used

    If the spacecraft crosses the auroral oval (alpha changes sign) 
    during the plot window, annotations are added which indicate the 
    poleward and equatorward directions.

    Parameters
    ----------
    formatter : MultiLineDateFormatter 
        Formatter instance managing the tick labels for this axis
    axis : matplotlib.axes.Axes
        Axes whose x-axis label and tick labels should be updated
    """
    # Interpolate footprint position and alpha at major tick locations
    xticks = axis.get_xticks()
    interp_dict = utils.interpFootprint(mdates.num2date(xticks))
    alphas = np.vectorize(utils.calcAngleFromOval)(
        interp_dict["interp_lats"], 
        interp_dict["interp_lons"]
    )

    # Build lookup dictionary to serve as the callable for addLineLabel
    alpha_dict = dict(zip(xticks, alphas))

    addLineLabel(formatter, axis, lambda x: alpha_dict[x], "$\\alpha$ (deg)")

    # --- Minor tick labels ---
    # Add rows for minor ticks if there is display space
    # Threshold of 50 display units was determined to work effectively
    minor_tick_spacing = np.min(np.diff(axis.get_xticks(minor=True)))   # matplotlib date units
    xlim_width = np.diff(axis.get_xlim())[0]    # matplotlib date units
    xaxis_bbox_width = axis.bbox.width      # display units

    if minor_tick_spacing / xlim_width * xaxis_bbox_width > 50:
        xticks_m = axis.get_xticks(minor=True)
        interp_dict_m = utils.interpFootprint(mdates.num2date(xticks_m))
        alphas_m = np.vectorize(utils.calcAngleFromOval)(
            interp_dict_m["interp_lats"], 
            interp_dict_m["interp_lons"]
        )
        alpha_dict_m = dict(zip(xticks_m, alphas_m))
        minor_formatter = lambda x, pos: f"\n{alpha_dict_m[x]:.02f}"
        axis.xaxis.set_minor_formatter(FuncFormatter(minor_formatter))
        axis.xaxis.set_tick_params(which="minor", pad=5)

    # --- Poleward/equatorward direction annotations ---
    # If alpha changes sign across the plotted window (i.e. Juno crosses 
    # the auroral oval), add annotations indicating crossing direction
    if alphas[0] * alphas[-1] < 0:
        # Alpha decreasing: Juno moving poleward to equatorward
        if alphas[0] > alphas[-1]:
            axis.text(
                0, 0, r"$\longleftarrow$ Poleward", 
                transform=axis.transAxes, ha="left", va="bottom"
            )
            axis.text(
                1, 0, r"Equatorward $\longrightarrow$", 
                transform=axis.transAxes, ha="right", va="bottom"
            )
        # Alpha increasing: Juno moving equatorward to poleward
        if alphas[0] < alphas[-1]:
            axis.text(
                0, 0, r"$\longleftarrow$ Equatorward", 
                transform=axis.transAxes, ha="left", va="bottom"
            )
            axis.text(
                1, 0, r"Poleward $\longrightarrow$", 
                transform=axis.transAxes, ha="right", va="bottom"
            )


def setMinorTicks(axis):
    """
    Sets the minor tick density on a datetime x-axis to a density 
    appropriate to the major tick cadence

    The major tick cadence is matched against a standard lookup table 
    of time intervals, and the closest match is used to select a minor 
    tick count that subdivides the major interval neatly (e.g. 6 minor 
    ticks per hour when major ticks are every 1 minute). Only needs to 
    be called on one axis when ``sharex=True`` is used

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axes with a datetime x-axis to add minor ticks to
    """
    ticks = axis.get_xticks()
    cadence = ticks[1] - ticks[0]     # Major tick spacing in matplotlib date units (days)

    # Reference cadences in days and their corresponding 
    #       minor tick subdivisions
    cadence_name_list = [
        "15s", "30s", "1min", "5min", "10min", "15min", 
        "30min", "1hr", "2hr", "3hr", "4hr", "6hr", "12hr"
    ]
    cadence_list = np.array([
        0.25/60/24, 0.5/60/24, 1/60/24, 5/60/24, 10/60/24, 15/60/24, 
        30/60/24, 1/24, 2/24, 3/24, 4/24, 6/24, 12/24
    ])
    minor_tick_num = [
        5, 6, 6, 5, 10, 5, 
        6, 6, 8, 6, 8, 6, 12
    ]

    # Find closest matching standard cadence and apply the 
    #       minor tick count
    idx = np.argmin(abs(cadence - cadence_list))
    axis.xaxis.set_minor_locator(AutoMinorLocator(minor_tick_num[idx]))


def addZoneHighlight(axes, update=True):
    """
    Shades auroral zone intervals on all provided axes using colored, 
    hatched patches

    Zone shading color is determined by zone_c_dict and hatch pattern 
    by hatch_dict (keyed by confidence level). All included axes must
    have the same x-axis limits

    Parameters
    ----------
    axes : list of matplotlib.axes.Axes
        Axes on which zone shading should be drawn
    update : bool, optional
        If True, reloads zone_boundary.dat from disk before drawing. 
        This ensures edits to the boundary file are reflected without 
        restarting the Python session
    """
    # Extract start and end times from the axes x-axis limits
    # Use timezone-naive datetimes to enable easier comparison with 
    #       zone boundary datafram
    start_dt = mdates.num2date(axes[1].get_xlim()[0]).replace(tzinfo=None)
    end_dt = mdates.num2date(axes[1].get_xlim()[1]).replace(tzinfo=None)

    # Optionally reload zone boundary data to capture recent edits
    if update:
        _zones_df = pd.read_csv(os.path.join(MODULE_DATA_DIR, "zone_boundary.dat"), sep=r"\s+")
        _zones_df["Start"] = pd.to_datetime(_zones_df["Start"], format="%Y-%m-%dT%H:%M:%S")
        _zones_df["End"] = pd.to_datetime(_zones_df["End"], format="%Y-%m-%dT%H:%M:%S")

    # Filter to zones that overlap the time window of the axes
    in_dt_range = (
        _zones_df["Start"].between(start_dt, end_dt) 
        | _zones_df["End"].between(start_dt, end_dt)
    )
    _zones_df_pj = _zones_df[in_dt_range]

    if _zones_df_pj.empty:
        return
    
    for i in _zones_df_pj.index:
        zone_label = _zones_df_pj["Zone"].loc[i]
        confidence = _zones_df_pj["Confidence"].loc[i]
        hatch = hatch_dict[confidence]

        time_start = _zones_df_pj["Start"].loc[i]
        time_end = _zones_df_pj["End"].loc[i]

        # Apply the shading to every axis in the input array
        for axis in axes[0:]:
            axis.axvspan(
                time_start, time_end, color=zone_c_dict[zone_label], 
                hatch=hatch, alpha=0.1, zorder=10
            )
