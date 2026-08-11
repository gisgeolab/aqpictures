"""
Feature group definitions for PM2.5 estimation.
"""

# Target
TARGET = "PM25"

# -------------------------------------------------
# Image features
# -------------------------------------------------

COLOR_FEATURES = [
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
]

CONTRAST_TEXTURE_FEATURES = [
    "contrast",
    "gray_entropy",
    "laplacian_variance",
    "mean_gradient_magnitude",
    "local_contrast",
]

HAZE_VISIBILITY_FEATURES = [
    "B_R_ratio",
    "dark_pixel_ratio",
    "sky_luminance_gradient",
]

IMAGE_FEATURES = (
    COLOR_FEATURES
    + CONTRAST_TEXTURE_FEATURES
    + HAZE_VISIBILITY_FEATURES
)

# -------------------------------------------------
# ERA5
# -------------------------------------------------

ERA5_FEATURES = [
    "T2M",
    "D2M",
    "RH",
    "U10",
    "V10",
    "SP",
    "TP",
    "BLH",
    "TCC",
    "WS10",
    "GP_500",
    "GP_850",
    "T_500",
    "T_850",
    "U_500",
    "U_850",
    "V_500",
    "V_850",
]

# -------------------------------------------------
# ARPA
# -------------------------------------------------

ARPA_FEATURES = [
    "temperature_mean",
    "wind_direction_mean",
    "relative_humidity_mean",
    "wind_speed_mean",
    "wind_gust_max",
    "wind_dir_sin",
    "wind_dir_cos",
]

# -------------------------------------------------
# Temporal
# -------------------------------------------------

TEMPORAL_FEATURES = [
    "hour",
    "day",
    "weekday",
    "month",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_daytime",
]

# =====================================================
# Feature groups
# =====================================================

FEATURE_GROUPS = {
    "Image": IMAGE_FEATURES,
    "ERA5": ERA5_FEATURES,
    "ARPA": ARPA_FEATURES,
    "Temporal": TEMPORAL_FEATURES,
}

# All features
ALL_FEATURES = (
    IMAGE_FEATURES
    + ERA5_FEATURES
    + ARPA_FEATURES
    + TEMPORAL_FEATURES
)

FUSION_FEATURE_SETS = {
    "Image": IMAGE_FEATURES,
    "Image + ERA5": IMAGE_FEATURES + ERA5_FEATURES,
    "Image + ERA5 + ARPA": IMAGE_FEATURES + ERA5_FEATURES + ARPA_FEATURES,
    "All Features": ALL_FEATURES,
}
