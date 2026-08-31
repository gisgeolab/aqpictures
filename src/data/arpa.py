from functools import reduce
from pathlib import Path

import pandas as pd


def get_variable_name(sensor_id, value_col):
    value_col = value_col.strip()

    if sensor_id == 6048:
        return "wind_direction_mean"

    elif sensor_id == 5911:
        return "temperature_mean"

    elif sensor_id == 6597:
        return "relative_humidity_mean"

    elif sensor_id == 19019 and value_col == "Medio":
        return "wind_speed_mean"

    elif sensor_id == 19019 and value_col == "Massimo":
        return "wind_gust_max"

    else:
        return f"unknown_{sensor_id}_{value_col}"


def merge_arpa_tables(
    files,
    output_file,
    time_column="Data-Ora",
    sensor_column="Id Sensore",
    utc_offset_hours=1,
    missing_value=-999,
):
    dfs = []

    for file in files:
        file = Path(file)

        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        sensor_id = int(df[sensor_column].iloc[0])

        value_col = [
            c for c in df.columns
            if c not in [sensor_column, time_column]
        ][0]

        new_col = get_variable_name(sensor_id, value_col)

        df = df[[time_column, value_col]].rename(
            columns={value_col: new_col}
        )

        dfs.append(df)

    if not dfs:
        raise ValueError(f"No ARPA CSV files were found: {files}")

    combined = pd.concat(dfs, ignore_index=True)

    combined[time_column] = pd.to_datetime(
        combined[time_column],
        errors="coerce",
    )

    # ARPA exports all timestamps in fixed Italian standard time (UTC+1),
    # including dates that fall within the daylight-saving period. Convert
    # them to UTC by subtracting the provider-specified fixed offset, then
    # explicitly mark the resulting timestamps as UTC.
    combined[time_column] = (
        combined[time_column] - pd.Timedelta(hours=utc_offset_hours)
    )
    combined[time_column] = combined[time_column].dt.tz_localize("UTC")

    combined = combined.sort_values(time_column)

    merged = (
        combined.groupby(time_column, as_index=False)
        .first()
        .sort_values(time_column)
        .reset_index(drop=True)
    )

    merged.replace(missing_value, pd.NA, inplace=True)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(output_file, index=False)

    return merged
