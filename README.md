# FTIR-Based Textile Fiber Classification

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Machine learning classification of textile fibers using FTIR spectroscopy and chemometric preprocessing**

---

## Overview

This repository contains the complete workflow for automated textile fiber classification using Fourier-Transform Infrared (FTIR) spectroscopy. The project implements multiple preprocessing pipelines and evaluates classification performance using Support Vector Machines (SVM) and Random Forest algorithms.

**Key Features:**
- Chemometric preprocessing (ALS baseline correction, SNV normalization, Savitzky-Golay derivatives)
- Multi-pipeline comparison (P1: ALS+SNV, P2: ALS+SNV+D1, P3: ALS+SNV+D2)
- Binary classification (Natural vs Man-made) and multiclass classification (fiber types/subtypes)
- Automated batch processing script for new samples
- Comprehensive visualization and analysis notebooks

---

## Repository Structure

```
Experiments/
├── initial_analysis.ipynb                    # Exploratory data analysis and experimentation
├── initial_preprocessing_experiments.ipynb   # Preprocessing method comparison
├── pretreatments.ipynb                       # Final preprocessing pipeline evaluation
├── initial_ml_classification.ipynb           # ML model development and testing
├── data_analysis_and_figures.ipynb           # Results visualization for publication
├── spectral_visualisation.ipynb              # Spectral plotting utilities
│
├── Preprocessing Pipeline - Script/          # Standalone batch processing tool
│   ├── process_spectra_pipeline.py           # Main script
│   ├── README.md                             # Script documentation
│   ├── exported_csvs/                        # Raw FTIR CSV input
│   ├── metadata/                             # Sample metadata
│   └── ml_datasets/                          # Generated feature matrices
│
├── raw_csv_data/                             # Raw FTIR spectra (CSV format) from PerkinElmer Spectrum
├── ml_datasets/                              # Preprocessed datasets for ML
├── images/                                   # Output figures
└── requirements.txt                          # Python dependencies
```

---

## Datasets

**Fiber Collections:**
- Microtrace Forensic Fiber Reference Collection (synthetic fibers)
- Arbidar Natural Fibre Collection (natural fibers)

**Sample Coverage:**
- Natural fibers: Cotton, Linen, Jute, Silk, Wool, Rayon
- Man-made fibers: Polyester, Nylon, Acrylic, Modacrylic

**Spectral Data:**
- Spectral range: 4000-400 cm⁻¹
- Resolution: ~1 cm⁻¹
- Multiple replicas per sample (3-5 scans)

---

## Preprocessing Pipelines

| Pipeline | Methods | Output Format | Use Case |
|----------|---------|---------------|----------|
| **P1** | ALS + SNV | Absorbance | Baseline classification |
| **P2** | ALS + SNV + D1 | 1st derivative | **Recommended** (best performance) |
| **P3** | ALS + SNV + D2 | 2nd derivative | Comparative analysis |

**Preprocessing Details:**
- **ALS (Asymmetric Least Squares):** Baseline correction (λ=1e6, p=0.001)
- **SNV (Standard Normal Variate):** Scatter normalization
- **Savitzky-Golay:** Derivative computation (window=15, polynomial=3)

---

## Classification Performance

**Binary Classification (Natural vs Man-made):**
- Cross-validation accuracy: 98.5% ± 1.2%
- Test accuracy: 100% (15/15 samples)
- Algorithm: SVM with RBF kernel

**Multiclass Classification (12 fiber subtypes):**
- Cross-validation accuracy: 95.8% ± 2.1%
- Test accuracy: 93.3% (14/15 samples)
- Algorithm: Random Forest (200 trees)

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/reeha-parkar/ftir-fiber-classification.git
cd ftir-fiber-classification/Experiments

# Install dependencies
pip install -r requirements.txt
```

### Workflow

1. **Exploratory Analysis:** Run `initial_analysis.ipynb`
2. **Preprocessing Evaluation:** Run `pretreatments.ipynb`
3. **Classification:** Run `initial_ml_classification.ipynb`
4. **Results Visualization:** Run `data_analysis_and_figures.ipynb`

### Batch Processing New Samples

```bash
cd "Preprocessing Pipeline - Script"
python process_spectra_pipeline.py
```

See [`Preprocessing Pipeline - Script/README.md`](Preprocessing%20Pipeline%20-%20Script/README.md) for detailed instructions.

---

## Requirements

- Python 3.12+
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn
- scikit-learn

See [`requirements.txt`](requirements.txt) for complete list with versions.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{parkar2025ftir,
  author = {Parkar, Reeha Karim},
  title = {FTIR-Based Textile Fiber Classification using Machine Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/reeha-parkar/ftir-fiber-classification}
}
```

---

## Author

**Reeha Karim Parkar**  
Department of Forensic Science, King's College London  
📧 reeha_karim.parkar@kcl.ac.uk | reehaparkar@gmail.com  
🔗 [GitHub](https://github.com/reeha-parkar)

**Supervisor:** Dr. Matteo Gallidabino  
**Funding:** IMPACT+

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- Microtrace LLC for the Forensic and Natural Fibre Collection
- IMPACT+ for 
- King's College London, Department of Forensic Science

---

**Last Updated:** December 16, 2025
