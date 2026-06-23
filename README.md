# Zonespy

A Python module for processing and visualising Juno auroral zone data, including magnetic field perturbations, field-aligned current estimates, UV brightness maps, energetic electron measurements, and electron density measurements.

## Dependencies

Zonespy requires the following Python packages:

- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [SciPy](https://scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [astropy](https://www.astropy.org/)
- [SpiceyPy](https://spiceypy.readthedocs.io/)
- [JupiterMag](https://github.com/mattkjames7/JupiterMag)
- [requests](https://requests.readthedocs.io/)

Standard library dependencies: `os`, `glob`, `datetime`, `tomllib`.

Zonespy was built and tested using Windows.

## Installation

Zonespy can be installed directly from GitHub using pip:

```bash
pip install git+https://github.com/nkruegler/Zonespy.git
```

After installation and before importing the package, the Zonespy config file must be set up to include (i) the directory where SPICE kernels (see below) are stored locally, and (ii) the destination directory for the storage of downloaded data files created by the package. An example config is found in `config.example.toml`. The example file can be modified to include the relevant paths and renamed to `config.toml`.

## SPICE Kernels

Zonespy uses [SpiceyPy](https://spiceypy.readthedocs.io/) to compute Juno ephemeris and perform time conversions. The following SPICE kernel types are required:

- Leap-second (LSK): `naif0012.tls`
- Planetary constants (PCK): `pck00010.tpc`
- Juno reference frame specifications (FK): `juno_v12.tf`
- Spacecraft ephemeris (SPK): `juno_rec_*.bsp`

Kernels are available from the NAIF PDS archive:
https://naif.jpl.nasa.gov/pub/naif/pds/data/jno-j_e_ss-spice-6-v1.0/jnosp_1000/

Kernels are loaded upon import of Zonespy in Python.

## Data Sources

Zonespy uses data from four Juno instruments. MAG, JEDI, and JADE moments data are downloaded via Zonespy and saved locally from the NASA Planetary Data System (PDS) file system. UVS data must be manually downloaded and placed in the appropriate local directory before use.

| Instrument | Data product | DOI | Download/Storage |
|---|---|---|---|
| MAG | Calibrated magnetic field (FGM-CAL) | [10.17189/1519711](https://doi.org/10.17189/1519711) | via Zonespy |
| JEDI | High energy resolution electron spectra (HIERSESP) | [10.17189/1519713](https://doi.org/10.17189/1519713) | via Zonespy |
| JADE | Electron moments | [10.17189/2fch-6v84](https://doi.org/10.17189/2fch-6v84) | via Zonespy |
| UVS | UV brightness profiles | [10.58119/ULG/SRUBKL](https://doi.org/10.58119/ULG/SRUBKL) | Manual |

Zonespy also uses the auroral electron density dataset from Kruegler et al. (2025), available at [Zenodo](https://zenodo.org/records/14713895), and the UVS reference auroral oval contours from Head et al. (2024), available at [Zenodo](https://zenodo.org/records/12067692). Both are downloaded and cached automatically on first use.

## Workflow

See the `examples/` directory for a complete worked example in `ZonespyExample.ipynb`. A typical Zonespy workflow consists of the following steps:

### 1. Define a time window

```python
start_dt, end_dt = utils.get_start_end_time(source="omode", event="PJ01")
```

### 2. Download data (first use only)

MAG and JEDI data must be downloaded before they can be loaded. This only needs to be done once per time window.

```python
MAGData.downloadMAGLocal(start_dt, end_dt)
JEDIData.downloadJEDILocal(start_dt, end_dt)
```

Note: JEDI downloads are large (~2–3 GB per day of data) and may take a significant amount of time.

UVS data must be downloaded manually from the DOI above and placed in the `UVS/` subdirectory of your data directory.

### 3. Load and process data

```python
mag = MAGData(start_dt, end_dt)
jedi = JEDIData(start_dt, end_dt)
uvs = UVSData(start_dt, end_dt)
```

On instantiation, each class automatically runs its full processing pipeline:

- `MAGData` computes the Juno ephemeris, subtracts the JRM33+Con2020 model field, and estimates field-aligned current density via the Lühr method.
- `JEDIData` rebins the native JEDI energy channels onto a log-spaced grid, bins measurements by pitch angle, and assembles a 3-D intensity datablock (time × energy × pitch angle).
- `UVSData` reads the relevant FITS file and unpacks UV brightness profiles along the Juno trajectory and four offset tracks.

### 4. Compile JADE density data (first use only; if needed)

JADE electron density data is compiled from the PDS into a single cached file:

```python
from Zonespy.jade_dens import compileJADEMoments
compileJADEMoments()
```

### 5. Plot

Uses `plotting` submodule. See `ZonespyExample.ipynb` for a specific example for each dataset.

## Module Reference

### `utils`
Utility functions used across the module, including footprint interpolation (`interpFootprint`), oval offset angle calculation (`calcAngleFromOval`), and time window retrieval (`get_start_end_time`).

### `mag`
Defines `MAGData`: loads and processes Juno/MAG 1-second calibrated magnetic field data. Computes magnetic perturbations and field-aligned current density using the Lühr single-spacecraft method.

### `jedi`
Defines `JEDIData`: loads and processes Juno/JEDI HIERSESP electron spectra. Assembles a 3-D intensity datablock and provides methods for computing energy distributions, pitch angle distributions, and loss-cone mean intensities.

### `uvs`
Defines `UVSData`: reads locally stored Juno/UVS FITS files and unpacks UV brightness image maps and the profiles along the spacecraft trajectory.

### `jade_dens`
Provides `compileJADEMoments`: downloads JADE electron density moments from the PDS for each perijove pass, appends magnetic footprint coordinates and oval offset angles, and saves the result to a single cached file.

### `plotting`
Utility functions for creating stackplots, including consistent grid styling, multi-row datetime axis formatting with footprint coordinates and alpha values, minor tick placement, and auroral zone interval highlighting.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

This module incorporates code adapted from work by Bertrand Bonfond and Linus Head (University of Liège), licensed under the BSD 3-Clause License.

## Citation

If you use Zonespy in your research, please cite the relevant data sources listed above. A citation for the package itself will be added upon publication.