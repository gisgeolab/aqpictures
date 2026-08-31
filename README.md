# A Benchmark for Webcam-Based PM2.5 Estimation

### A reproducible benchmark for image features, multi-source data fusion, conventional machine-learning models, and pretrained CNN representations

This repository contains an end-to-end research workflow for webcam-based PM2.5 estimation using ground-level imagery and multi-source environmental data. It covers data collection, preprocessing, ROI-based image feature extraction, feature and data-source evaluation, machine-learning benchmarking, and a pretrained CNN extension.

The project integrates four complementary data sources:

- webcam imagery and image-derived visual features;
- PM2.5 observations from EEA monitoring stations;
- ERA5 meteorological reanalysis variables; and
- ARPA Lombardia ground meteorological observations.

All model comparisons use controlled data inputs and date-aware evaluation procedures so that differences in performance can be interpreted consistently.

---

# Overall Workflow

```mermaid
flowchart LR

    subgraph S1[Data Sources]
        A[📷 Webcam Images]
        B[🌫️ EEA PM2.5]
        C[☁️ ERA5 Meteorology]
        D[🌦️ ARPA Meteorology]
    end

    subgraph S2[Dataset Construction · Notebooks 01–02]
        E[ROI Selection]
        F[Handcrafted Image Features]
        G[Multi-Source Integration]
        H[Cleaning & Feature Engineering]
        I[Analysis-Ready Dataset]
    end

    subgraph S3[Feature & Data-Source Evaluation · Notebook 03]
        J[RQ1 · Image Feature Evaluation]
        K[RQ2 · Progressive Data Fusion]
    end

    subgraph S4[Engineered-Feature Benchmark · Notebook 04]
        L[Common Date-Aware Protocol]
        M[RQ3 · Five-Model Benchmark]
        N[Benchmark Results Export]
    end

    subgraph S5[Pretrained CNN Extension · Notebook 05]
        O[Frozen EfficientNet-B0 Embeddings]
        P[RQ4 · Image-Only Evaluation]
        Q[RQ5 · Multimodal Fusion]
        R[Comparison with Benchmark]
    end

    A --> E --> F --> G
    B --> G
    C --> G
    D --> G
    G --> H --> I

    I --> J --> K
    I --> L --> M --> N

    A --> O --> P
    O --> Q
    I --> Q
    P --> R
    Q --> R
    N --> R

    classDef source fill:#EEF4FF,stroke:#4A78C2,color:#1B2A4A,stroke-width:1.5px;
    classDef process fill:#F4FFF4,stroke:#4C9A5A,color:#1F3A28,stroke-width:1.5px;
    classDef research fill:#FFF7E6,stroke:#D97706,color:#5B3A00,stroke-width:1.5px;
    classDef benchmark fill:#F3EEFF,stroke:#7950B2,color:#35205C,stroke-width:1.5px;
    classDef output fill:#FDECEC,stroke:#C0392B,color:#6B1E18,stroke-width:1.5px;

    class A,B,C,D source;
    class E,F,G,H process;
    class J,K,P,Q research;
    class L,M,O benchmark;
    class I,N,R output;
```

The workflow separates dataset construction from evaluation. Notebook 03 evaluates handcrafted features and data-source contributions, Notebook 04 benchmarks engineered-feature models, and Notebook 05 provides a focused pretrained-CNN extension whose final comparison uses the benchmark export from Notebook 04.

---

# Research Questions

- **RQ1:** Can handcrafted ROI image features effectively represent PM2.5 variations?
- **RQ2:** Does combining image, environmental, monitoring, and temporal predictors improve PM2.5 estimation compared with using image features alone?
- **RQ3:** How accurately do conventional machine-learning models and a feature-based neural network estimate PM2.5 under a common benchmark protocol?
- **RQ4:** Can pretrained CNN representations from raw ROI images achieve competitive PM2.5 estimation performance compared with handcrafted image features?
- **RQ5:** Does combining learned image representations with environmental and temporal predictors improve performance over image representations alone?

---

# Repository Structure

```text
webcam-pm25-benchmark/
├── .github/
│   └── workflows/
│       └── webcam.yml
├── config/
│   └── roi.json
├── data/
│   ├── raw/                  # Local raw data; full content is not committed
│   ├── interim/              # Complete intermediate datasets
│   └── processed/            # Final analysis-ready dataset
├── notebooks/
│   ├── 01_webcam_pm25_dataset_pipeline.ipynb
│   ├── 02_dataset_cleaning_preprocessing.ipynb
│   ├── 03_image_features_and_multisource_data_evaluation.ipynb
│   ├── 04_machine_learning_model_benchmark.ipynb
│   └── 05_deep_learning_model.ipynb
├── results/
│   ├── feature_evaluation/
│   ├── model_benchmark/
│   └── deep_learning/
├── src/
│   ├── data/
│   │   ├── arpa.py
│   │   ├── era5.py
│   │   ├── integration.py
│   │   ├── pm25.py
│   │   └── webcam.py
│   ├── vision/
│   │   ├── handcrafted.py
│   │   └── roi.py
│   ├── features/
│   │   ├── evaluation.py
│   │   └── plots.py
│   ├── models/
│   │   ├── benchmark.py
│   │   └── plots.py
│   ├── deep_learning/
│   │   ├── embeddings.py
│   │   ├── models.py
│   │   └── training.py
│   └── config.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

# Data Availability

The repository includes the complete intermediate and processed datasets required for the feature evaluation and conventional model benchmark:

```text
data/interim/
data/processed/
```

The complete raw webcam image archive is not committed to the repository because of its size. It is stored separately on Google Drive and is publicly accessible here:

[Webcam Image Archive (Google Drive)](https://drive.google.com/drive/folders/1sUjlPimVWVhaKdo9CWsO0uQVuJK8V2J6?usp=drive_link)

After downloading, place the images in:

```text
data/raw/images/
```

The expected image filenames use timestamps such as:

```text
20250327-1200.jpg
```

The raw webcam images are used for ROI inspection, handcrafted image-feature extraction, dataset construction, and pretrained CNN representation learning.

Other raw data sources are not fully committed to the repository. PM2.5 and ERA5 data can be retrieved through the corresponding data-access procedures implemented in the project, while ARPA Lombardia meteorological source files must be supplied separately.

Availability by entry point:

| Starting point | Repository data sufficient? | Additional requirements |
| -------------- | --------------------------: | ----------------------- |
| Notebook 01 | No | Raw webcam images, ARPA source files, network access, and CDS API configuration |
| Notebook 02 | Yes | `data/interim/merged_dataset.csv` |
| Notebook 03 | Yes | `data/processed/final_dataset.csv` |
| Notebook 04 | Yes | `data/processed/final_dataset.csv` |
| Notebook 05 | No | Raw webcam images and the benchmark export from Notebook 04 |

---

# Installation

Clone the repository:

```bash
git clone https://github.com/QingxuanTuo/webcam-pm25-benchmark.git
cd webcam-pm25-benchmark
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```powershell
.venv\Scripts\activate
```

Install the dependencies and start JupyterLab:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

Run Jupyter from either the project root or `notebooks/`. The notebooks automatically locate the nearest parent directory containing both `src/` and `notebooks/`. Ensure that the selected Jupyter kernel uses the project `.venv` environment.

## Google Colab

The notebooks are Colab-compatible when the repository code and required data are cloned or mounted in the Colab environment. Notebook 01 mounts Google Drive and expects the full webcam image directory at:

```text
/content/drive/MyDrive/webcam_images
```

The repository itself must still be available in the current directory or one of its parent directories so that the notebooks can locate `src/`.

## ERA5 CDS API Configuration

ERA5 downloading requires a Copernicus Climate Data Store account. Follow the official [CDS API setup instructions](https://cds.climate.copernicus.eu/how-to-api), accept the applicable dataset Terms of Use, and create a `.cdsapirc` file in your home directory.

Windows:

```text
C:\Users\YOUR_USERNAME\.cdsapirc
```

Linux/macOS:

```text
~/.cdsapirc
```

Current configuration format:

```text
url: https://cds.climate.copernicus.eu/api
key: <PERSONAL-ACCESS-TOKEN>
```

---

# Automated Webcam Collection

Webcam images are collected from the Milan webcam platform using the scheduled GitHub Actions workflow defined in:

```text
.github/workflows/webcam.yml
```

The download script is located at:

```text
src/data/webcam.py
```

The workflow downloads timestamped webcam images and uploads them to Google Drive using the repository's configured `RCLONE_CONF` secret. It can also be triggered manually from:

```text
Actions -> webcam download -> Run workflow
```

The collection workflow will be revalidated in the new repository before its scheduled operation is relied upon.

---

# Quick Start

The complete research workflow follows this order:

```text
Notebook 01 -> Notebook 02 -> Notebook 03 -> Notebook 04 -> Notebook 05
```

Because complete interim and processed datasets are provided, users interested only in evaluation and benchmarking may start from Notebook 03. Notebook 05 still requires the complete image archive and the output produced by Notebook 04.

## 1. Webcam PM2.5 Dataset Pipeline

```text
notebooks/01_webcam_pm25_dataset_pipeline.ipynb
```

Performs:

- ROI selection and visual inspection;
- handcrafted image feature extraction;
- PM2.5 downloading and processing;
- ERA5 downloading and processing;
- ARPA Lombardia data merging; and
- multi-source dataset integration.

Main outputs:

```text
data/interim/image_features.csv
data/interim/PM25_MI_hourly.csv
data/interim/era5_all_merged.csv
data/interim/arpa_merged.csv
data/interim/merged_dataset.csv
```

## 2. Dataset Cleaning and Preprocessing

```text
notebooks/02_dataset_cleaning_preprocessing.ipynb
```

Performs data profiling, missing-value handling, cleaning, feature engineering, temporal validation, and final dataset generation.

Main output:

```text
data/processed/final_dataset.csv
```

## 3. Image Features and Multi-Source Data Evaluation

```text
notebooks/03_image_features_and_multisource_data_evaluation.ipynb
```

Evaluates handcrafted ROI image features and the incremental contribution of ERA5, ARPA, and temporal predictors. The models in this notebook are analytical tools for feature and data-source evaluation.

Main output directory:

```text
results/feature_evaluation/
```

## 4. Machine-Learning Model Benchmark

```text
notebooks/04_machine_learning_model_benchmark.ipynb
```

Benchmarks five engineered-feature models under a common, date-aware evaluation protocol and exports the comparison used by Notebook 05.

Required output for Notebook 05:

```text
results/model_benchmark/traditional_model_benchmark_results.csv
```

## 5. Pretrained CNN Extension

```text
notebooks/05_deep_learning_model.ipynb
```

Evaluates two prespecified configurations: Frozen EfficientNet Image Only and Frozen EfficientNet Fusion. EfficientNet-B0 is used as a frozen feature extractor rather than as part of a broad CNN architecture benchmark.

Required inputs:

```text
data/processed/final_dataset.csv
data/interim/image_features.csv
config/roi.json
data/raw/images/
results/model_benchmark/traditional_model_benchmark_results.csv
```

Main output directory:

```text
results/deep_learning/
```

Embedding caches are generated automatically when absent. Generated `.npz` embedding caches and `.pt` model files are excluded from Git, while the result tables and figures remain available.

---

# Generated Datasets

| Dataset | Description |
|---|---|
| `data/interim/image_features.csv` | Timestamped ROI-based handcrafted image features and image paths |
| `data/interim/PM25_MI_hourly.csv` | Hourly PM2.5 observations from the EEA monitoring station |
| `data/interim/era5_all_merged.csv` | Processed ERA5 meteorological variables |
| `data/interim/arpa_merged.csv` | Processed ARPA Lombardia ground observations |
| `data/interim/merged_dataset.csv` | Integrated multi-source environmental dataset |
| `data/processed/final_dataset.csv` | Cleaned, feature-engineered, analysis-ready dataset |

---

# Repository Outputs

The repository retains the principal CSV tables and PNG figures generated by the evaluation notebooks:

```text
results/
├── feature_evaluation/   # Notebook 03
├── model_benchmark/      # Notebook 04
└── deep_learning/        # Notebook 05
```

These outputs include feature correlations, feature-group and fusion evaluations, cross-validation and hold-out metrics, temporal robustness analyses, prediction diagnostics, and CNN fusion comparisons.

---

# License

The source code in this repository is licensed under the [MIT License](LICENSE).

Third-party environmental data and webcam imagery are not covered by the MIT License and remain subject to the terms and licences of their respective providers.

