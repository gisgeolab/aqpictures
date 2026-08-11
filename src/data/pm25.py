import shutil
import time
from pathlib import Path

import pandas as pd
import requests


API_BASE = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/"
URLS_ENDPOINT = "ParquetFile/urls"


def extract_urls(text):
    urls = []

    for raw in text.splitlines():
        line = raw.strip().strip('"')

        if line.startswith("http://") or line.startswith("https://"):
            urls.append(line)

    return urls


def download_file(url, out_path):
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_pm25_data(
    api_start,
    api_end,
    station_prefix,
    temp_dir,
    output_file,
    countries=None,
    cities=None,
    pollutants=None,
    dataset=1,
    aggregation="hour",
    remove_temp=True,
):
    if countries is None:
        countries = ["IT"]

    if cities is None:
        cities = ["Milano (greater city)"]

    if pollutants is None:
        pollutants = ["PM2.5"]

    temp_dir = Path(temp_dir)
    output_file = Path(output_file)

    temp_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    body = {
        "countries": countries,
        "cities": cities,
        "pollutants": pollutants,
        "dataset": dataset,
        "dateTimeStart": api_start,
        "dateTimeEnd": api_end,
        "aggregationType": aggregation,
        "source": "Jupyter notebook",
    }

   # print("Request:", body)

    response = requests.post(
        API_BASE + URLS_ENDPOINT,
        json=body,
        timeout=180,
    )
    response.raise_for_status()

    urls = extract_urls(response.text)
    #print("Found parquet files:", len(urls))

    if len(urls) == 0:
        raise ValueError("No parquet file URLs found from the API response.")

    for i, url in enumerate(urls, start=1):
        file_name = url.split("/")[-1]
        out_path = temp_dir / file_name

        # print(f"[{i}/{len(urls)}] Downloading {file_name}")

        download_file(url, out_path)
        time.sleep(0.2)

    parquet_files = list(temp_dir.glob("*.parquet"))

    if len(parquet_files) == 0:
        raise FileNotFoundError(f"No parquet files found in: {temp_dir}")

    dfs = []

    for file in parquet_files:
        #print("Reading", file.name)

        df = pd.read_parquet(file)
        df["source_file"] = file.name
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    #print("Merged shape:", merged.shape)
    #print("Columns:", merged.columns.tolist())

    filtered = merged[
        merged["Samplingpoint"]
        .astype(str)
        .str.startswith(station_prefix, na=False)
    ].copy()

    #print("After station filter:", filtered.shape)

    filtered["Start"] = pd.to_datetime(filtered["Start"], utc=True)
    filtered["End"] = pd.to_datetime(filtered["End"], utc=True)

    start_ts = pd.to_datetime(api_start, utc=True)
    end_ts = pd.to_datetime(api_end, utc=True)

    filtered = filtered[
        (filtered["Start"] >= start_ts)
        & (filtered["Start"] <= end_ts)
    ].copy()

    filtered["Start"] = filtered["Start"].dt.tz_localize(None)
    filtered["End"] = filtered["End"].dt.tz_localize(None)

    filtered = filtered.sort_values("Start").reset_index(drop=True)

    #print("After time filter:", filtered.shape)

    target_columns = [
        "Samplingpoint",
        "Pollutant",
        "Start",
        "End",
        "Value",
        "Unit",
        "AggType",
        "Validity",
        "Verification",
        "ResultTime",
        "DataCapture",
        "FkObservationLog",
    ]

    existing = [c for c in target_columns if c in filtered.columns]

    final_df = filtered[existing].copy()

    final_df.to_csv(output_file, index=False)

    #print(f"Saved CSV: {output_file}")
    #print("Final shape:", final_df.shape)
    #print(final_df.head())

    if remove_temp and temp_dir.exists():
        shutil.rmtree(temp_dir)
        #print("Temporary PM2.5 files removed.")

    return final_df