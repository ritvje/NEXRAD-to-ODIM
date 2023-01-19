# NEXRAD to ODIM HDF5 files

A short script for transforming NEXRAD Level 2 files to ODIM HDF5 file format.
The output file should be more-or-less v2.2 compliant, but nothing is guaranteed.
The output filename is the same as the input filename, but with the extension changed to `.h5`.

The code for saving the data to ODIM HDF5 is modified from the [xradar.io](https://github.com/openradar/xradar) module.

Requirements:

- xarray
- pyart
- xradar
- dask
- h5py
- arrow

## Usage

```bash

python nexrad_to_odim.py <input-folder-path> --nworkers <number-of-parallel-workers>

```
