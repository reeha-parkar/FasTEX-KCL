# FTIR-Based Textile Fiber Classification

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

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
├── initial_ml_classification.ipynb           # ML model development and testing
├── data_analysis_and_figures.ipynb           # Final dataset analysis checks and visualisations
├── pretreatments_and_classification.ipynb    # Final preprocessing pipeline evaluation
├── spectral_visualisation.ipynb              # Spectral plotting utilities
│
├── Preprocessing Pipeline - Script/          # Standalone batch processing tool
│   ├── process_spectra_pipeline.py           # Main script
│   ├── README.md                             # Script documentation and instructions
│   ├── exported_csvs/                        # Raw FTIR CSV input
│   ├── metadata/                             # Sample metadata
│   └── ml_datasets/                          # Generated feature matrices
│
├── raw_csv_data/                             # Raw FTIR spectra (CSV format) from PerkinElmer Spectrum exports
├── ml_datasets/                              # Preprocessed datasets for ML
├── images/                                   # Output figures
└── requirements.txt                          # Python dependencies
```

---

## Datasets

**Fiber Collections:**
- Microtrace Forensic Fiber Reference Collection (synthetic fibers)
- Microtrace Arbidar Natural Fibre Collection (natural fibers)
- Bio-Couture & UNUSUWUL, associated with IMPACT+ (assorted fibers)

**Sample Coverage:**
- Natural fibers: Cotton, Linen, Jute, Silk, Wool
- Man-made fibers: Polyester, Nylon, Acrylic, Modacrylic, Regenerated Cellulose

**Spectral Data:**
- Spectral range: 4000-400 cm⁻¹
- Resolution: 4 cm⁻¹
- Multiple replicas per sample (1-3 scans)

---

## Preprocessing Pipelines

| Pipeline | Methods | Output Format | Use Case |
|----------|---------|---------------|----------|
| **P1** | ALS + SNV | Absorbance | Baseline classification |
| **P2** | ALS + SNV + D1 | 1st derivative | SOTA |
| **P3** | ALS + SNV + D2 | 2nd derivative | SOTA, Comparative analysis |

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
git clone https://github.com/reeha-parkar/FasTEX-KCL.git
cd FasTEX-KCL/

# Install dependencies
pip install -r requirements.txt
```

### Workflow

1. **Exploratory Analysis:** Run `initial_analysis.ipynb` (For experimentations on different types of chemometric preprocessing techniques)
2. **Preprocessing Evaluation and Classification:** Run `pretreatments_and_classification.ipynb`
3. **Results Visualization:** Run `data_analysis_and_figures.ipynb`

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

**Dataset:**
```bibtex
@dataset{parkar2025dataset,
  author = {Parkar, Reeha and Jain, Angelica and Prendergast-Miller, Miranda and Stanton, Thomas and Sheridan, Kelly and Gallidabino, Matteo},
  title = {A dataset of infrared (ATR-FTIR) spectra for textile fibres of natural and man-made origin},
  year = {2025},
  publisher = {Mendeley Data},
  version = {V1},
  doi = {10.17632/rx3fjgz96x.1},
  url = {https://doi.org/10.17632/rx3fjgz96x.1}
}
```

> **Note:** A data descriptor paper for this dataset is currently under review at *Data in Brief* journal. This citation will be updated upon publication.

**Code Repository:**
```bibtex
@misc{parkar2025ftir,
  author = {Parkar, Reeha},
  title = {FTIR-Based Textile Fiber Classification using Machine Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/reeha-parkar/ftir-fiber-classification}
}
```

---

## Author

**Reeha Karim Parkar**  
ML Research Inern, King's College London  
📧 reeha_karim.parkar@kcl.ac.uk | reehaparkar@gmail.com  
🔗 [GitHub](https://github.com/reeha-parkar)

**Supervisor/PI:** Dr. Matteo Gallidabino, Department of Forensic Science, King's College London

**Funding:** [IMPACT+](https://hosting.northumbria.ac.uk/impactplusnetwork/fastex/)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- Microtrace LLC for the Forensic and Natural Fibre Collection
- IMPACT+
- King's College London, Department of Forensic Science

---

**Last Updated:** December 16, 2025
