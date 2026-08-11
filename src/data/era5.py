from __future__ import annotations

import zipfile
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
import shutil


def build_date_parts(start_date, end_date):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if start.year != end.year or start.month != end.month:
        raise ValueError(
            "build_date_parts must receive a date range within one calendar month."
        )

    days = [
        f"{day:02d}"
        for day in range(start.day, end.day + 1)
    ]

    times = [f"{hour:02d}:00" for hour in range(24)]

    return (
        f"{start.year:04d}",
        f"{start.month:02d}",
        days,
        times,
    )


def unzip_file(zip_path, extract_dir):
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)

    # Remove previous extracted files to avoid reading stale outputs
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
        names = zf.namelist()

    return [extract_dir / name for name in names]

def iter_month_ranges(start_date, end_date):
    """
    Split a date range into calendar-month chunks.

    Example:
        2026-03-03 to 2026-05-10
    becomes:
        2026-03-03 to 2026-03-31
        2026-04-01 to 2026-04-30
        2026-05-01 to 2026-05-10
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if start > end:
        raise ValueError(
            f"start_date ({start_date}) must not be after end_date ({end_date})."
        )

    current = start

    while current <= end:
        current_month_end = current + pd.offsets.MonthEnd(0)
        chunk_end = min(current_month_end, end)

        yield current, chunk_end

        current = chunk_end + pd.Timedelta(days=1)


def first_existing(paths, suffixes):
    for path in paths:
        if path.suffix.lower() in suffixes:
            return path

    raise FileNotFoundError(f"No file found with suffixes: {suffixes}")


def process_single_csv(csv_path):
    df = pd.read_csv(csv_path)

    #print("Single raw columns:", df.columns.tolist())

    if "valid_time" not in df.columns:
        raise KeyError(
            f"'valid_time' not found in single-level CSV columns: {df.columns.tolist()}"
        )

    df["time"] = pd.to_datetime(df["valid_time"]).dt.tz_localize(None)

    rename_map = {
        "2m_temperature": "T2M",
        "2m_dewpoint_temperature": "D2M",
        "10m_u_component_of_wind": "U10",
        "10m_v_component_of_wind": "V10",
        "surface_pressure": "SP",
        "total_precipitation": "TP",
        "boundary_layer_height": "BLH",
        "total_cloud_cover": "TCC",
        "cloud_base_height": "CBH",
        "t2m": "T2M",
        "d2m": "D2M",
        "u10": "U10",
        "v10": "V10",
        "sp": "SP",
        "tp": "TP",
        "blh": "BLH",
        "tcc": "TCC",
        "cbh": "CBH",
    }

    df = df.rename(columns=rename_map)

    if {"T2M", "D2M"}.issubset(df.columns):
        t_c = df["T2M"] - 273.15
        td_c = df["D2M"] - 273.15

        df["RH"] = 100.0 * np.exp(
            (17.625 * td_c) / (243.04 + td_c)
            - (17.625 * t_c) / (243.04 + t_c)
        )

        df["RH"] = df["RH"].clip(lower=0, upper=100)

    if {"U10", "V10"}.issubset(df.columns):
        df["WS10"] = np.sqrt(df["U10"] ** 2 + df["V10"] ** 2)

    drop_cols = [
        c for c in ["valid_time", "latitude", "longitude"]
        if c in df.columns
    ]

    df = df.drop(columns=drop_cols)

    keep_cols = [
        "time",
        "T2M",
        "D2M",
        "RH",
        "U10",
        "V10",
        "SP",
        "TP",
        "BLH",
        "TCC",
        "CBH",
        "WS10",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    out = df[keep_cols].sort_values("time").reset_index(drop=True)

    #print("Single processed columns:", out.columns.tolist())
    
    #print("Single processed shape:", out.shape)

    return out


def process_pressure_netcdf(nc_path):
    ds = xr.open_dataset(nc_path)

    df = ds.to_dataframe().reset_index()

    required = {"valid_time", "pressure_level"}

    if not required.issubset(df.columns):
        raise KeyError(
            f"Pressure data missing required columns. Found: {df.columns.tolist()}"
        )

    df["time"] = pd.to_datetime(df["valid_time"]).dt.tz_localize(None)

    df["pressure_level"] = (
        df["pressure_level"]
        .astype(int)
        .astype(str)
    )

    value_cols = [c for c in ["z", "t", "u", "v"] if c in df.columns]

    if len(value_cols) == 0:
        raise ValueError(
            f"No pressure-level variables found. Available columns: {df.columns.tolist()}"
        )

    df_agg = (
        df.groupby(["time", "pressure_level"], as_index=False)[value_cols]
        .mean()
    )

    var_map = {
        "z": "GP",
        "t": "T",
        "u": "U",
        "v": "V",
    }

    wide_parts = []

    for raw_var, short_var in var_map.items():
        if raw_var not in df_agg.columns:
            print(f"Warning: {raw_var} not found in pressure data.")
            continue

        temp = (
            df_agg[["time", "pressure_level", raw_var]]
            .pivot(index="time", columns="pressure_level", values=raw_var)
            .rename(columns=lambda level: f"{short_var}_{level}")
            .reset_index()
        )

        wide_parts.append(temp)

    if len(wide_parts) == 0:
        raise ValueError("No pressure-level variables available after processing.")

    out = wide_parts[0]

    for part in wide_parts[1:]:
        out = out.merge(part, on="time", how="outer")

    out = out.sort_values("time").reset_index(drop=True)

    #print("Pressure processed columns:", out.columns.tolist())
    #print("Pressure processed shape:", out.shape)

    return out

def download_era5_month(
    client,
    lat,
    lon,
    chunk_start,
    chunk_end,
    work_dir,
    single_vars,
    pressure_vars,
    pressure_levels,
    pressure_area,
    overwrite=False,
):
    """
    Download and process one calendar-month chunk of ERA5 data.
    """
    chunk_start = pd.Timestamp(chunk_start)
    chunk_end = pd.Timestamp(chunk_end)

    chunk_label = (
        f"{chunk_start.strftime('%Y%m%d')}_"
        f"{chunk_end.strftime('%Y%m%d')}"
    )

    chunk_dir = Path(work_dir) / chunk_label
    chunk_dir.mkdir(parents=True, exist_ok=True)

    single_zip = chunk_dir / f"single_levels_{chunk_label}.zip"
    pressure_zip = chunk_dir / f"pressure_levels_{chunk_label}.zip"

    single_extract = chunk_dir / "single_extracted"
    pressure_extract = chunk_dir / "pressure_extracted"

    monthly_output = chunk_dir / f"era5_merged_{chunk_label}.csv"

    # Reuse an already completed monthly result
    if monthly_output.exists() and not overwrite:
        print(f"Using existing ERA5 chunk: {monthly_output.name}")

        monthly_df = pd.read_csv(monthly_output)
        monthly_df["time"] = pd.to_datetime(monthly_df["time"])

        return monthly_df

    start_str = chunk_start.strftime("%Y-%m-%d")
    end_str = chunk_end.strftime("%Y-%m-%d")

    print(f"Downloading ERA5 single levels: {start_str} to {end_str}")

    single_request = {
        "variable": single_vars,
        "location": {
            "latitude": lat,
            "longitude": lon,
        },
        "date": [f"{start_str}/{end_str}"],
        "data_format": "csv",
    }

    client.retrieve(
        "reanalysis-era5-single-levels-timeseries",
        single_request,
        str(single_zip),
    )

    year, month, days, times = build_date_parts(
        start_str,
        end_str,
    )

    print(f"Downloading ERA5 pressure levels: {start_str} to {end_str}")

    pressure_request = {
        "product_type": ["reanalysis"],
        "variable": pressure_vars,
        "pressure_level": pressure_levels,
        "year": [year],
        "month": [month],
        "day": days,
        "time": times,
        "area": pressure_area,
        "data_format": "netcdf",
        "download_format": "zip",
    }

    client.retrieve(
        "reanalysis-era5-pressure-levels",
        pressure_request,
        str(pressure_zip),
    )

    single_files = unzip_file(
        single_zip,
        single_extract,
    )

    pressure_files = unzip_file(
        pressure_zip,
        pressure_extract,
    )

    single_csv = first_existing(
        single_files,
        (".csv",),
    )

    pressure_nc = first_existing(
        pressure_files,
        (".nc", ".netcdf"),
    )

    single_df = process_single_csv(single_csv)
    pressure_df = process_pressure_netcdf(pressure_nc)

    monthly_df = pd.merge(
        single_df,
        pressure_df,
        on="time",
        how="inner",
        validate="one_to_one",
    )

    monthly_df = (
        monthly_df
        .sort_values("time")
        .drop_duplicates(subset="time")
        .reset_index(drop=True)
    )

    monthly_df.to_csv(
        monthly_output,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

    print(
        f"Completed ERA5 chunk: {chunk_label}, "
        f"{len(monthly_df)} rows"
    )

    return monthly_df

def download_era5_data(
    lat,
    lon,
    start_date,
    end_date,
    work_dir,
    output_file,
    single_vars=None,
    pressure_vars=None,
    pressure_levels=None,
    pressure_area=None,
    overwrite=False,
):
    """
    Download ERA5 single-level and pressure-level data in monthly chunks,
    process each chunk, and merge all chunks into one final CSV.
    """
    work_dir = Path(work_dir)
    output_file = Path(output_file)

    work_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if single_vars is None:
        single_vars = [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
            "total_precipitation",
            "boundary_layer_height",
            "total_cloud_cover",
            "cloud_base_height",
        ]

    if pressure_vars is None:
        pressure_vars = [
            "geopotential",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
        ]

    if pressure_levels is None:
        pressure_levels = ["500", "850"]

    if pressure_area is None:
        pressure_area = [
            45.50,  # North
            9.00,   # West
            45.25,  # South
            9.25,   # East
        ]

    client = cdsapi.Client()

    monthly_dfs = []

    for chunk_start, chunk_end in iter_month_ranges(
        start_date,
        end_date,
    ):
        monthly_df = download_era5_month(
            client=client,
            lat=lat,
            lon=lon,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            work_dir=work_dir,
            single_vars=single_vars,
            pressure_vars=pressure_vars,
            pressure_levels=pressure_levels,
            pressure_area=pressure_area,
            overwrite=overwrite,
        )

        monthly_dfs.append(monthly_df)

    if not monthly_dfs:
        raise RuntimeError("No ERA5 monthly data were downloaded or processed.")

    final_df = pd.concat(
        monthly_dfs,
        ignore_index=True,
    )

    final_df["time"] = pd.to_datetime(
        final_df["time"],
        errors="raise",
    )

    final_df = (
        final_df
        .drop_duplicates(subset="time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    expected_start = pd.Timestamp(start_date)
    expected_end = pd.Timestamp(end_date) + pd.Timedelta(hours=23)

    final_df = final_df.loc[
        final_df["time"].between(
            expected_start,
            expected_end,
            inclusive="both",
        )
    ].copy()

    # Validate hourly temporal continuity
    expected_index = pd.date_range(
        expected_start,
        expected_end,
        freq="h",
    )

    actual_index = pd.DatetimeIndex(final_df["time"])

    missing_times = expected_index.difference(actual_index)
    unexpected_times = actual_index.difference(expected_index)

    print("\nERA5 final quality check")
    print("------------------------")
    print("Expected rows:", len(expected_index))
    print("Actual rows:", len(final_df))
    print("Missing timestamps:", len(missing_times))
    print("Unexpected timestamps:", len(unexpected_times))

    if len(missing_times) > 0:
        print("First missing timestamps:")
        print(missing_times[:10].tolist())

    if len(unexpected_times) > 0:
        print("First unexpected timestamps:")
        print(unexpected_times[:10].tolist())

    output_df = final_df.copy()

    output_df["time"] = output_df["time"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    output_df.to_csv(
        output_file,
        index=False,
    )

    print("\nFinal ERA5 dataset saved")
    print("Path:", output_file)
    print("Shape:", output_df.shape)
    print(
        "Time range:",
        output_df["time"].min(),
        "to",
        output_df["time"].max(),
    )

    return output_df