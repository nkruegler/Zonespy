import glob
import os
import tomllib

import JupiterMag as jm
import spiceypy as sp


MODULE_DIR = os.path.dirname(__file__)

CONFIG_DIR = os.path.join(MODULE_DIR, "config.toml")
try:
    with open(CONFIG_DIR, "rb") as f:
        config = tomllib.load(f)
except:
    raise Exception("File config.toml not present. " \
    "Edit config.example.toml to include SPICE and Instrument Data Destination Directory. " \
    "Then rename to config.toml.")

MODULE_DATA_DIR = os.path.join(MODULE_DIR, "ModuleData/")

if not config["paths"]["use_external"]:
    DATA_DIR = os.path.join(MODULE_DIR, "ExampleInstrumData/")
else:
    DATA_DIR = config["paths"]["external_data_dir"]

SPICE_DIR = config["paths"]["spice_dir"]

# --- Importing SPICE kernels ---
sp.furnsh(SPICE_DIR + "naif0012.tls") # leapseconds
sp.furnsh(SPICE_DIR + "pck00010.tpc") # planetary constants
sp.furnsh(SPICE_DIR + "juno_v12.tf") # Juno frame details
# Load all available Juno empherides
for file in glob.glob(SPICE_DIR + r"juno_rec_*.bsp"):
    sp.furnsh(file)

# --- Initilizing JupiterMag Models ---
jm.Internal.Config(Model="jrm33", Degree=18)
jm.Con2020.Config(equation_type="analytic")