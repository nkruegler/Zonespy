import os
import pandas as pd
import datetime as dt

from . import DATA_DIR, MODULE_DATA_DIR
from . import utils

"""
Function to download and compile JADE electron moments data near perijove.

Data is sourced from the Planetary Data System 
    (https://doi.org/10.17189/2fch-6v84)
This submodule accesses this data directly through the PDS file system 
    (see jedi_pds_url variable)
"""

JADE_PDS_URL = "https://pds-ppi.igpp.ucla.edu/data/JNO-J-JAD-5-MOMENTS-V1.0/DATA/"
JADE_DIR = DATA_DIR + "JADE/"

# Perijove table containing labels and time windows for each perijove
_perijove_df = pd.read_csv(os.path.join(MODULE_DATA_DIR, "perijove_list.dat"), sep=r"\s+")


def compileJADEMoments(overwrite=False):
    """
    Donwloads and compiles JADE electron moments data for each perijove 
    and saves the result to a single tab-separated file

    For each perijove in _perijove_df, the function downloads the 
    relevant CSV file from the PDS, trims to the time window where Juno 
    is within 12 Jovian radii, appends magnetic footprint coordinates 
    and the oval offset angle alpha, and concatenates all perijoves 
    into one file

    Perijoves after 2025 are skipped, since no JADE moments data is 
    currently available past this date (as of June 2026)

    Parameters
    ----------
    overwrite : bool, optional
        If True, re-downloads and recompiles files even if the file 
        already exists locally. Default if False
    """

    # Skip download if output file already exists
    if not overwrite:
        if os.path.isfile(JADE_DIR + "JADE_density_pj.dat"):
            print(
                "JADE moments already downloaded. "
                "Use 'overwrite' if new version requested."
            )
            return

    # Column names for the JADE moments data file
    names = [
        "utc", "jade_version", "data_sel", "packet_mode", "acc_time",
        "src_bkgd", "issue", "ev/q_low", "ev/q_hi", "radius", "lat", "lt", "lon",
        "dim", "mass", "charge", "num_look_dir", "n_cc", "n_cc_err",
        "pressure", "pressure_err", "temp", "temp_err", "qual"
    ]

    df_list = []
    for i in range(len(_perijove_df)):
        # Extract time window and label for the perijove
        start = _perijove_df["Time_Before_PJ_at_Radius_12Rj"][i]
        end = _perijove_df["Time_After_PJ_at_Radius_12Rj"][i]
        perijove = _perijove_df["Perijove"][i]
        start_dt = dt.datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
        end_dt = dt.datetime.strptime(end, "%Y-%m-%dT%H:%M:%S")

        # Extract year and day-of-year for constructing file patterns
        year = start_dt.timetuple().tm_year
        doy_start = start_dt.timetuple().tm_yday
        doy_end = end_dt.timetuple().tm_yday
        
        print(f"Getting JADE data for {perijove}")

        # No JADE moments data currently available for dates after 2025
        if year > 2025:
            print(f"No JADE data is available near {perijove}")
            continue

        try:
            # Construct PDS for the specified CSV and read it
            # Directory naming convention: YYYY/YYYYDDD
            # File naming convention: 
            #   JAD_L50_HLS_ELC_MOM_ISO_2D_ELECTRONS_YYYYDDD_V02.CSV
            df = pd.read_csv(
                JADE_PDS_URL 
                + f"{year}/{year}{doy_start:03}/"
                f"JAD_L50_HLS_ELC_MOM_ISO_2D_ELECTRONS_{year}{doy_start:03}_V02.CSV",
                names=names
            )

            # If time window spans midnight, read following day's file
            if doy_end != doy_start:
                try:
                    df_end = pd.read_csv(
                        JADE_PDS_URL 
                        + f"{year}/{year}{doy_end:03}/"
                        f"JAD_L50_HLS_ELC_MOM_ISO_2D_ELECTRONS_{year}{doy_end:03}_V02.CSV",
                        names=names
                    )
                    df = pd.concat([df, df_end], ignore_index=True)
                except:
                    # If the second file is unavailable, proceed with 
                    #       first day only
                    pass

            # Parse the timestamps and trim to the perijove time window
            df["datetime"] = pd.to_datetime(df["utc"], format="%Y-%jT%H:%M:%S.%f")
            df_near = df[df["datetime"].between(start_dt, end_dt)]

            # Add the perijove label as the first column
            df_near.insert(0, "event", perijove)

            if df_near.empty:
                print(f"No JADE data is available near {perijove}")
            else:
                df_list.append(df_near)
                print("Success")

        except:
            print(f"No JADE data is available near {perijove}")

    # Concatenate all perijoves into one DataFrame
    print("Compiling data sets and calculating footprints...")
    jade_df = pd.concat(df_list, ignore_index=True)

    # Interpolate magnetic footprint at each timestamp
    interp_dict = utils.interpFootprint(jade_df["datetime"], UV=True)
    jade_df["footlats"] = interp_dict["interp_lats"]
    jade_df["footlons"] = interp_dict["interp_lons"]

    # Comput angle from the auroral oval for each timestamp
    jade_df["alpha"] = jade_df.apply(
        lambda x: utils.calcAngleFromOval(x["footlats"], x["footlons"]), 
        axis=1
    )

    # Drop the datetime column and save to file
    jade_df = jade_df.drop(columns="datetime")
    jade_df.to_csv(JADE_DIR + "JADE_density_pj.dat", sep="\t", index=False)
    print("Done")
