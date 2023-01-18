"""NEXRAD Level 2 to ODIM HDF5 files.

Transforms a NEXRAD Level 2 file to an ODIM HDF5 file. The output file should be v2.2.
compliant, but nothing is guaranteed. The output filename is the same as
the input filename, but with the extension changed to .h5.

The code for saving the data to ODIM HDF5 is modified from the xradar.io module.

Requirements:
- xarray
- pyart
- xradar
- dask
- h5py

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""
import argparse
from pathlib import Path
import pyart
import xarray as xr
import xradar as xd

import dask

import datetime as dt

import h5py
import numpy as np

from xradar.model import required_sweep_metadata_vars

# Ignore runtime warnings
import warnings

warnings.filterwarnings("ignore")


ODIM_TO_PYART_FIELDS = {
    "DBZH": "reflectivity",
    "HCLASS": "radar_echo_classification",
    "KDP": "specific_differential_phase",
    "PHIDP": "differential_phase",
    "RHOHV": "cross_correlation_ratio",
    "SQI": "normalized_coherent_power",
    "TH": "total_power_horizontal",
    "VRAD": "velocity",
    "VRADH": "velocity_horizontal",
    "WRAD": "spectrum_width",
    "ZDR": "differential_reflectivity",
    "SNR": "signal_to_noise_ratio",
    "LOG": "log_signal_to_noise_ratio",
}

PYART_TO_ODIM_FIELDS = {v: k for k, v in ODIM_TO_PYART_FIELDS.items()}


def _write_odim(source, destination):
    """Writes ODIM_H5 Attributes.

    Function from xradar.io module.

    Parameters
    ----------
    source : dict
        Attributes to write
    destination : handle
        h5py-group handle
    """
    for key, value in source.items():
        if key in destination.attrs:
            continue
        if isinstance(value, str):
            tid = h5py.h5t.C_S1.copy()
            tid.set_size(len(value) + 1)
            H5T_C_S1_NEW = h5py.Datatype(tid)
            destination.attrs.create(key, value, dtype=H5T_C_S1_NEW)
        else:
            destination.attrs[key] = value


def _write_odim_dataspace(source, destination):
    """Write ODIM_H5 Dataspaces.

    Function from xradar.io module.

    Parameters
    ----------
    source : dict
        Moments to write
    destination : handle
        h5py-group handle
    """
    # for now assume all variables as valid
    # keys = [key for key in source if key in sweep_vars_mapping]
    # but not metadata variables
    keys = [key for key in source if key not in required_sweep_metadata_vars]
    data_list = [f"data{i + 1}" for i in range(len(keys))]
    data_idx = np.argsort(data_list)
    for idx in data_idx:
        value = source[keys[idx]]
        h5_data = destination.create_group(data_list[idx])
        enc = value.encoding

        # p. 21 ff
        h5_what = h5_data.create_group("what")
        try:
            undetect = float(value._Undetect)
        except AttributeError:
            undetect = np.finfo(np.float_).max

        # set some defaults, if not available
        # Calculate scale and offset
        N_BITS = 16
        add_offset = value.valid_min
        scale_factor = (value.valid_max - value.valid_min) / (2 ** N_BITS - 2)
        # scale_factor = float(enc.get("scale_factor", 1.0))
        # add_offset = float(enc.get("add_offset", 0.0))
        _fillvalue = float(enc.get("_FillValue", undetect))
        dtype = enc.get("dtype", value.dtype)
        nodata = N_BITS ** 2 - 1
        undetect = 0.0

        what = {
            "quantity": PYART_TO_ODIM_FIELDS[value.name],
            "gain": scale_factor,
            "offset": add_offset,
            "nodata": nodata,
            "undetect": undetect,
        }
        _write_odim(what, h5_what)

        # moments handling
        # todo: check bottom-up/top-down rhi
        dim0 = "elevation" if source.sweep_mode == "rhi" else "azimuth"
        val = value.sortby(dim0).values
        fillval = _fillvalue * scale_factor
        fillval += add_offset
        val = (val - add_offset) / scale_factor
        val[np.isnan(val)] = undetect
        # if np.issubdtype(dtype, np.integer):
        val = np.rint(val).astype(np.uint16)
        # todo: compression is chosen totally arbitrary here
        #  maybe parameterizing it?
        ds = h5_data.create_dataset(
            "data",
            data=val,
            dtype=f"uint{N_BITS}",
            compression="gzip",
            compression_opts=9,
            fillvalue=_fillvalue,
        )
        # if enc["dtype"] == "uint8":
        image = "IMAGE"
        version = "1.2"
        tid1 = h5py.h5t.C_S1.copy()
        tid1.set_size(len(image) + 1)
        H5T_C_S1_IMG = h5py.Datatype(tid1)
        tid2 = h5py.h5t.C_S1.copy()
        tid2.set_size(len(version) + 1)
        H5T_C_S1_VER = h5py.Datatype(tid2)
        ds.attrs.create("CLASS", image, dtype=H5T_C_S1_IMG)
        ds.attrs.create("IMAGE_VERSION", version, dtype=H5T_C_S1_VER)


def to_odim(dtree, filename):
    """Save DataTree to ODIM_H5/V2_2 compliant file.

    Function mostly from xradar.io module, with modifications:

    - change pyart field names to ODIM field names
    - always add IMAGE_VERSION attribute to data datasetss

    Parameters
    ----------
    dtree : :class:`datatree.DataTree`
    filename : str
        output filename
    """
    root = dtree["/"]

    h5 = h5py.File(filename, "w")

    # root group, only Conventions for ODIM_H5
    _write_odim({"Conventions": "ODIM_H5/V2_2"}, h5)

    # how group
    how = {}
    how.update({"_modification_program": "xradar"})

    h5_how = h5.create_group("how")
    _write_odim(how, h5_how)

    grps = [g for g in dtree.groups if "sweep" in g]

    # what group, object, version, date, time, source, mandatory
    # p. 10 f
    what = {}
    if len(grps) > 1:
        what["object"] = "PVOL"
    else:
        what["object"] = "SCAN"
    # todo: parameterize version
    what["version"] = "H5rad 2.2"
    what["date"] = str(root["time_coverage_start"].values)[:10].replace("-", "")
    what["time"] = str(root["time_coverage_end"].values)[11:19].replace(":", "")
    what["source"] = root.attrs["instrument_name"]

    h5_what = h5.create_group("what")
    _write_odim(what, h5_what)

    # where group, lon, lat, height, mandatory
    where = {
        "lon": root["longitude"].values,
        "lat": root["latitude"].values,
        "height": root["altitude"].values,
    }
    h5_where = h5.create_group("where")
    _write_odim(where, h5_where)

    # datasets
    ds_list = [f"dataset{i + 1}" for i in range(len(grps))]
    for idx in range(len(ds_list)):
        ds = dtree[grps[idx]].ds
        dim0 = "elevation" if ds.sweep_mode == "rhi" else "azimuth"

        # datasetN group
        h5_dataset = h5.create_group(ds_list[idx])

        # what group p. 21 ff.
        h5_ds_what = h5_dataset.create_group("what")
        ds_what = {}
        # skip NaT values
        valid_times = ~np.isnat(ds.time.values)
        t = sorted(ds.time.values[valid_times])
        start = dt.datetime.utcfromtimestamp(np.rint(t[0].astype("O") / 1e9))
        end = dt.datetime.utcfromtimestamp(np.rint(t[-1].astype("O") / 1e9))
        ds_what["product"] = "SCAN"
        ds_what["startdate"] = start.strftime("%Y%m%d")
        ds_what["starttime"] = start.strftime("%H%M%S")
        ds_what["enddate"] = end.strftime("%Y%m%d")
        ds_what["endtime"] = end.strftime("%H%M%S")
        _write_odim(ds_what, h5_ds_what)

        # where group, p. 11 ff. mandatory
        h5_ds_where = h5_dataset.create_group("where")
        rscale = ds.range.values[1] / 1.0 - ds.range.values[0]
        rstart = (ds.range.values[0] - rscale / 2.0) / 1000.0
        a1gate = np.argsort(ds.sortby(dim0).time.values)[0]
        fixed_angle = ds["sweep_fixed_angle"].values
        ds_where = {
            "elangle": fixed_angle,
            "nbins": ds.range.shape[0],
            "rstart": rstart,
            "rscale": rscale,
            "nrays": ds.azimuth.shape[0],
            "a1gate": a1gate,
        }
        _write_odim(ds_where, h5_ds_where)

        # how group, p. 14 ff.
        h5_ds_how = h5_dataset.create_group("how")
        tout = [tx.astype("O") / 1e9 for tx in ds.sortby(dim0).time.values]
        tout_sorted = sorted(tout)

        # handle non-uniform times (eg. only second-resolution)
        if np.count_nonzero(np.diff(tout_sorted)) < (len(tout_sorted) - 1):
            tout = np.roll(
                np.linspace(tout_sorted[0], tout_sorted[-1], len(tout)), a1gate
            )
            tout_sorted = sorted(tout)

        difft = np.diff(tout_sorted) / 2.0
        difft = np.insert(difft, 0, difft[0])
        azout = ds.sortby(dim0).azimuth
        diffa = np.diff(azout) / 2.0
        diffa = np.insert(diffa, 0, diffa[0])
        elout = ds.sortby(dim0).elevation
        diffe = np.diff(elout) / 2.0
        diffe = np.insert(diffe, 0, diffe[0])

        # ODIM_H5 datasetN numbers are 1-based
        sweep_number = ds.sweep_number + 1
        ds_how = {
            "scan_index": sweep_number,
            "scan_count": len(grps),
            "startazT": tout - difft,
            "stopazT": tout + difft,
            "startazA": azout - diffa,
            "stopazA": azout + diffa,
            "startelA": elout - diffe,
            "stopelA": elout + diffe,
        }
        _write_odim(ds_how, h5_ds_how)

        # write moments
        _write_odim_dataspace(ds, h5_dataset)

    h5.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "input_dir",
        type=str,
        help="Input directory path, all files ending with '_V06' will be processed",
    )
    argparser.add_argument(
        "--nworkers",
        type=int,
        default=1,
        help="Number of workers",
    )
    args = argparser.parse_args()

    @dask.delayed
    def transform_file(inputpath):
        # Read NEXRAD Level 2 file
        radar = pyart.io.read_nexrad_archive(inputpath)

        # Convert to netCDF Cf/Radial
        pyart.io.write_cfradial(inputpath.with_suffix(".nc"), radar)

        # Read with xradar
        dtree = xd.io.open_cfradial1_datatree(
            inputpath.with_suffix(".nc"), first_dim="time", optional=False
        )

        # Write ODIM_H5
        to_odim(dtree, inputpath.with_suffix(".h5"))

        # Remove netcdf file
        inputpath.with_suffix(".nc").unlink()

    res = []
    for inputpath in Path(args.input_dir).glob("*_V06"):
        res.append(transform_file(inputpath))

    scheduler = "processes" if args.nworkers > 1 else "single-threaded"
    dask.compute(*res, num_workers=args.nworkers, scheduler=scheduler)
