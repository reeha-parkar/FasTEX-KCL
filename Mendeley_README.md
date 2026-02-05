# FasTEX Dataset - Repository Structure

This repository contains FT-IR spectroscopy data for textile fiber identification and classification. The data is organised into six main directories, each representing a different stage of data processing.

## Folder Structure

### 1. `01_raw_spectra/`
**Contents:** 160 SP files  
**Description:** Contains original instrument output files in PerkinElmer `.sp` format. These files can be opened using software such as PerkinElmer Spectrum, Spectragryph, or other FT-IR data viewers supporting `.sp` files.

### 2. `02_feature_matrix_raw/`
**Contents:** 1 CSV file  
**Description:** Tabular feature matrix of unprocessed transmission spectra. Each row corresponds to a measurement.

### 3. `03_feature_matrix_baseline_corrected/`
**Contents:** 1 CSV file  
**Description:** Tabular feature matrix of baseline-corrected spectra generated using the asymmetric least-squares (ALS) method.

### 4. `04_feature_matrix_baseline_corrected_average/`
**Contents:** 1 CSV file  
**Description:** Tabular feature matrix of average baseline-corrected spectra grouped by fiber subtype (e.g., wool, nylon 6).

### 5. `05_feature_matrix_preprocessed/`
**Contents:** 1 CSV file  
**Description:** Tabular feature matrices of fully pre-processed spectra prepared using three alternative preprocessing pipelines:

- **(i)** Conversion to absorbance, ALS baseline correction, and SNV scatter correction
- **(ii)** Conversion to absorbance, ALS correction, SNV, and Savitzky–Golay smoothing (15-point window, first derivative)
- **(iii)** Conversion to absorbance, ALS correction, SNV, and Savitzky–Golay smoothing (15-point window, second derivative)

### 6. `06_other_files/`
**Contents:** 1 CSV file, 1 Python script  
**Description:** Contains:
- **Metadata table** providing sample-level information (e.g., sample ID, fiber type, origin, collection details)
- **Python companion script** enabling users to apply the same preprocessing workflow to their own spectra

---

## Usage Notes

- Raw `.sp` files in `01_raw_spectra/` require PerkinElmer Spectrum software for viewing. Alternatively, since the spectral data in these files corresponds to the same information available in the CSV format, users may visualise the spectra using any spreadsheet or data visualisation software (e.g., Microsoft Excel, Python with matplotlib/pandas, R) by accessing the tabular data in `02_feature_matrix_raw/`.
- CSV files contain tabular data that can be opened with any spreadsheet software or data analysis tools (Excel, Python pandas, R, etc.)
- The Python preprocessing script in `06_other_files/` can be used to replicate the preprocessing pipeline on new data

## Data Processing Pipeline

The folder structure reflects the sequential data processing stages:

```
Raw Spectra (.sp) 
    ↓
Feature Matrix (Raw Transmission)
    ↓
Baseline Correction (ALS)
    ↓
Fully Preprocessed (Multiple Pipelines)
```

---

## Citation

If you use this dataset in your research, please cite the associated publication.

## Contact

For questions or issues regarding this dataset, please contact the corresponding author.
