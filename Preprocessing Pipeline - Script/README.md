# FTIR Spectra Preprocessing Pipeline

## Quick Start

1. Install dependencies: `pip install pandas numpy scipy`
2. Place raw FTIR CSV files in `exported_csvs/`
3. Ensure `metadata/all_samples_metadata.csv` exists
4. Run: `python process_spectra_pipeline.py`

---

## Directory Structure

```
├── process_spectra_pipeline.py     # Main script
├── exported_csvs/                  # INPUT: Raw FTIR CSV files
├── metadata/
│   └── all_samples_metadata.csv    # INPUT: Required metadata
└── ml_datasets/                    # OUTPUT: Generated datasets
```

---

## Input Requirements

### Raw CSV Files (`exported_csvs/`)
- **Filename format:** `GenericFibreName - ScanNumber.csv` (e.g., `Cotton - 1.csv`)
- **Header line 1 (Row 1 Column 2 should contain):** `[Source],[Binder_ID]`  (e.g., `Forensic,1`, `Bio-Couture,4.4`)
- **Data:** Wavenumber and transmittance pairs

### Metadata File (`metadata/all_samples_metadata.csv`)
Must contain columns: `Sample_ID`, `Source`, `Binder_ID`, `Origin`, `Type`, `Subtype`, `Details`

---

## Output Files (in `ml_datasets/`)

1. `feature_matrix_raw_transmittance.csv` - Raw %T data
2. `feature_matrix_baseline_corrected.csv` - ALS corrected
3. `feature_matrix_preprocessed_als_snv.csv` - Pipeline 1 (ALS+SNV)
4. `feature_matrix_preprocessed_als_snv_d1.csv` - Pipeline 2 (ALS+SNV+D1) 
5. `feature_matrix_preprocessed_als_snv_d2.csv` - Pipeline 3 (ALS+SNV+D2)
6. `scanned_samples_metadata.csv` - Unique sample metadata of all scanned samples

---

## For Data Collection

- Use consistent naming: `FiberType - Number.csv`
- Scan each sample 1-3 times (replicas tracked automatically)
- Export with full header including Source and corresponding Binder_ID
- Keep the `all_samples_metadata.csv` updated with all new/unscanned samples so `scanned_samples_metadata.csv` can be generated

---

## Troubleshooting

**"all_samples_metadata.csv not found"** → Create metadata file in `metadata/` folder

**"Sample_ID not found"** → Check filename matches metadata, verify Binder_ID in CSV header

**"Unknown fiber type"** → Add fiber to `FIBRE_ORIGIN` dictionary in script (line ~50)

**Files skipped** → Check filename format, CSV header structure, file integrity

---

## Contact

**Author:** Reeha Karim Parkar  
**Email:** reeha_karim.parkar@kcl.ac.uk  
**GitHub:** @reeha-parkar

---

**Version:** 1.0 | **Last Updated:** December 16, 2025
