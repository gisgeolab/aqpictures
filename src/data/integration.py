from functools import reduce
from pathlib import Path

import pandas as pd


def merge_all_datasets(
    arpa_file,
    image_file,
    pm25_file,
    era5_file,
    output_file,
    arpa_time_column="Data-Ora",
    pm25_time_column="Start",
    pm25_value_column="Value",
    pm25_unit_column="Unit",
    image_time_column="datetime",
):
    arpa = pd.read_csv(arpa_file)
    image_features = pd.read_csv(image_file)
    pm25 = pd.read_csv(pm25_file)
    era5 = pd.read_csv(era5_file)

    for df in [arpa, image_features, pm25, era5]:
        df.columns = df.columns.str.strip()

    pm25[pm25_time_column] = pd.to_datetime(pm25[pm25_time_column])

    pm25_selected = pm25[
        [
            pm25_time_column,
            pm25_value_column,
            pm25_unit_column,
        ]
    ].copy()

    pm25_selected = pm25_selected.rename(
        columns={
            pm25_time_column: "time",
            pm25_value_column: "PM25",
            pm25_unit_column: "Unit",
        }
    )

    era5["time"] = pd.to_datetime(era5["time"])

    era5_pm25 = pd.merge(
        era5,
        pm25_selected,
        on="time",
        how="left",
    )

    arpa = arpa.rename(columns={arpa_time_column: "time"})
    arpa["time"] = pd.to_datetime(arpa["time"])

    image_features = image_features.rename(columns={image_time_column: "time"})
    image_features["time"] = pd.to_datetime(image_features["time"])
    image_features = image_features.sort_values("time").reset_index(drop=True)

    merged_all = pd.merge(
        image_features,
        era5_pm25,
        on="time",
        how="left",
    )
    merged_all = pd.merge(
        merged_all,
        arpa,
        on="time",
        how="left",
    )

    merged_all = merged_all.sort_values("time").reset_index(drop=True)

    column_order = [
    # Time
    "time",

    # Image features: color and brightness
    "R_roi",
    "G_roi",
    "B_roi",
    "R_std",
    "G_std",
    "B_std",
    "S_mean",
    "V_mean",
    "colorfulness",
    "sky_brightness",

    # Image features: contrast and texture
    "contrast",
    "gray_entropy",
    "laplacian_variance",
    "mean_gradient_magnitude",
    "local_contrast",

    # Image features: haze and visibility
    "B_R_ratio",
    "dark_pixel_ratio",
    "sky_luminance_gradient",

    # Image path
    "image_path",

    # PM2.5
    "PM25",
    "Unit",

    # ERA5 single-level variables
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

    # ERA5 pressure-level variables
    "GP_500",
    "GP_850",
    "T_500",
    "T_850",
    "U_500",
    "U_850",
    "V_500",
    "V_850",

    # ARPA meteorological variables
    "temperature_mean",
    "wind_direction_mean",
    "relative_humidity_mean",
    "wind_speed_mean",
    "wind_gust_max",
]

    existing_cols = [c for c in column_order if c in merged_all.columns]
    other_cols = [c for c in merged_all.columns if c not in existing_cols]

    merged_all = merged_all[existing_cols + other_cols]

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    merged_all.to_csv(output_file, index=False)

    #print(f"Saved: {output_file}")
    #print("Final rows:", len(merged_all))
    #print("Final shape:", merged_all.shape)
    #print(merged_all.head())

    return merged_all
