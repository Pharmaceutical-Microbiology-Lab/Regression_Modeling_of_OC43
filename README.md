# A HEK293T-derived explainable CatBoost signature for estimating HCoV-OC43 viral burden from host transcriptomes

This repository contains the code, analysis pipeline, and pre-trained models for the publication:
**"A HEK293T-derived explainable CatBoost signature for estimating HCoV-OC43 viral burden from host transcriptomes"** by Haesung Jeon and Choongho Lee.

## Overview

We developed a CatBoost regression model using single-cell RNA sequencing data to predict HCoV-OC43 viral burden at single-cell resolution. By leveraging CatBoost's native feature importance and SHAP (SHapley Additive exPlanations) values, we identified a core signature of 10 host genes (**TPI1, PPIA, HMGN2, FTL, RPS29, TSC22D3, SNHG7, HSPA8, ATP5MF, and IFRD1**) that are most predictive of infection intensity. This compact 10-gene model retains strong predictive performance and successfully generalizes to independent bulk RNA-seq datasets (e.g., MRC-5 lung fibroblasts). Furthermore, when tested on a non-viral cellular stress dataset (tunicamycin-treated K562 cells), the model effectively distinguishes generic stress from high-burden viral states while properly representing the shared biological overlap at lower levels of infection.

## Project Structure

- `notebooks/`: Jupyter notebooks covering the entire workflow (data preparation, model training, validation, and visualization).
- `scripts/`: Python and R scripts containing utility functions and plotting routines used across the notebooks.
- `envs/`: Conda environment configuration files (`ml_env.yml` and `sc_env.yml`).
- `Data/`: Directory to store the required raw data files.

## Environment Setup

This project uses two Conda environments:

- **sc_env**: Used for single-cell preprocessing (Scanpy, etc.).
- **ml_env**: Used for machine learning, SHAP analysis, and downstream validation tasks (CatBoost, PyCaret, etc.).

You can recreate these environments using the provided YAML files:

```bash
conda env create -f envs/sc_env.yml
conda env create -f envs/ml_env.yml
```

## Data Requirements

Before running the notebooks, please download the following raw datasets from NCBI GEO and place them in the correct directories:

1. **GSE278059 (OC43 scRNA-seq Data in HEK293T)**
   - Download `GSM8538852_20221110_OC43_HEK_5p_2_cellIDs.csv` from [NCBI GSE278059](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278059) and place it in the `./Data/` directory.
   - Download the Raw matrix file from the same link and place it in the `./Data/Single-cell_raw_data/` directory.

2. **GSM2406677 (K562 Tunicamycin ER Stress Data)**
   - Download the `barcodes.tsv.gz`, `genes.tsv.gz`, and `matrix.mtx.gz` files from [NCBI GSM2406677](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2406677).
   - Place these three downloaded files inside the `./Data/GSM2406677/` directory.
   - _Note: Our pipeline specifically analyzes cells treated with tunicamycin (4 μg/mL for 6 hours; identified via the gemgroup 1 barcode suffix '-1') to test the model's specificity against non-viral ER stress._

_(Note: The processed feature count matrices for the MRC-5 dataset (GSE252692) are already provided in the repository to facilitate study reproducibility.)_

## Workflow & Notebooks

The analysis workflow is divided into sequentially numbered Jupyter notebooks located in the `notebooks/` directory:

- **00 - 01**: Initial setup, quality control, and selection of training data (`00_Initial.ipynb`, `01_Train_data_selection.ipynb`).
- **02 - 03**: Training the full regression model and tuning CatBoost hyperparameters to derive the 10-gene signature (`02_Full_model_train.ipynb`, `03_Catboost_tunning.ipynb`).
- **04**: Preparing and validating the model against external MRC-5 bulk RNA-seq data (`04-02_MRC-5_prepare.ipynb`, `04-03_MRC-5_Bulk-seq_data_validation.ipynb`, `04_01_MRC-5_DESeq2.ipynb`).
  - _Note: When applying the model to bulk data, the Z-score scaler derived from single-cell training was intentionally omitted due to the fundamental differences in noise characteristics (e.g., zero-inflation/dropouts) between single-cell and bulk RNA-seq modalities._
- **05**: Visualization of the main predictive results, feature importances, and SHAP values (`05_Visualize.ipynb`).
- **06**: Pseudobulk preparation, PCA, DESeq2 analysis, and GO term biological characterization of infection states (`06-01` to `06-04`).
- **07**: External validation using K562 tunicamycin ER stress data and uninfected (healthy) baseline comparison (`07_01` to `07_03`).
