"""
FTIR Spectra Processing Pipeline: ALS + SNV + D1
=================================================

This script processes raw FTIR spectral CSV files through a complete preprocessing pipeline:
1. Load raw transmittance (%T) data
2. Apply ALS baseline correction
3. Apply SNV (Standard Normal Variate) normalization
4. Apply 1st derivative (Savitzky-Golay)

Input:
- Raw CSV files in 'exported_csvs/' folder
- Existing metadata file: 'metadata/all_samples_metadata.csv'

Output:
- feature_matrix_raw_transmittance.csv
- feature_matrix_baseline_corrected.csv
- feature_matrix_preprocessed.csv (ALS+SNV+D1)
- scanned_samples_metadata.csv

Author: Reeha Parkar
Date: 15th December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')

###############################################################################
# FIBER CLASSIFICATION MAPPINGS
###############################################################################

# Fiber origin mapping (Natural vs Man-made)
FIBRE_ORIGIN = {
    # Natural - Plant-based (Cellulose)
    'Cotton': 'Natural',
    'Linen': 'Natural',
    'Jute': 'Natural',
    'Denim': 'Natural',
    'Flax': 'Natural',
    'Hemp': 'Natural',
    'Kapok': 'Natural',
    'Kenaf': 'Natural',
    'Rafia': 'Natural',
    'Ramie': 'Natural',
    'Sisal': 'Natural',
    'Cacao': 'Natural',
    'Coir': 'Natural',
    
    # Natural - Animal-based (Protein)
    'Silk': 'Natural',
    'Wool': 'Natural',
    'Alpaca': 'Natural',
    'Rabbit': 'Natural',
    'Camel': 'Natural',
    'Goat': 'Natural',
    'Goose': 'Natural',
    'Llama': 'Natural',
    'Musk Ox': 'Natural',
    'Yak': 'Natural',
    
    # Man-made
    'Polyester': 'Man-made',
    'Nylon': 'Man-made',
    'Acrylic': 'Man-made',
    'Modacrylic': 'Man-made',
    'Aramid': 'Man-made',
    'Carbon': 'Man-made',
    'Chlorofiber': 'Man-made',
    'Olefin': 'Man-made',
    'PBI': 'Man-made',
    'Polyacrylate': 'Man-made',
    'Spandex': 'Man-made',
    'Sulfar': 'Man-made',
    'Rayon': 'Man-made',
    'Viscose': 'Man-made',
    'Lyocell': 'Man-made',
    'Modal': 'Man-made',
    'Acetate': 'Man-made'
    
    # ADD MORE FIBER TYPES HERE AS NEEDED
    # Format: 'FiberName': 'Natural' or 'Man-made'
}

# Natural fiber classification - Plant-based (Cellulose)
CELLULOSE_FIBRES = ['Cotton', 'Linen', 'Jute', 'Bamboo', 'Denim', 'Flax', 'Hemp', 
                    'Kapok', 'Kenaf', 'Rafia', 'Ramie', 'Sisal', 'Cacao', 'Coir']

# Natural fiber classification - Animal-based (Protein)
PROTEIN_FIBRES = ['Silk', 'Wool', 'Alpaca', 'Rabbit', 'Camel', 'Goat', 
                  'Goose', 'Llama', 'Musk Ox', 'Yak']

# Regenerated cellulose fibers
REGENERATED_CELLULOSE = ['Rayon', 'Viscose', 'Lyocell', 'Modal']

# Acetate fibers
ACETATE_FIBRES = ['Acetate']

# Fiber name abbreviations for Sample_ID generation
FIBRE_ABBREVIATIONS = {
    # Natural - Cellulose
    'Cotton': 'COT',
    'Linen': 'LIN',
    'Jute': 'JUT',
    'Bamboo': 'BAM',
    'Denim': 'DEN',
    'Flax': 'FLX',
    'Hemp': 'HMP',
    'Kapok': 'KPK',
    'Kenaf': 'KNF',
    'Rafia': 'RAF',
    'Ramie': 'RAM',
    'Sisal': 'SIS',
    'Cacao': 'CAC',
    'Coir': 'COI',
    'Unknown': 'UNK',
    
    # Natural - Protein
    'Silk': 'SIL',
    'Wool': 'WOL',
    'Alpaca': 'ALP',
    'Rabbit': 'RAB',
    'Camel': 'CAM',
    'Goat': 'GOT',
    'Goose': 'GOS',
    'Llama': 'LLM',
    'Musk Ox': 'MUX',
    'Yak': 'YAK',
    
    # Man-made
    'Polyester': 'PES',
    'Nylon': 'NYL',
    'Acrylic': 'ACR',
    'Modacrylic': 'MAC',
    'Aramid': 'ARM',
    'Carbon': 'CAR',
    'Chlorofiber': 'CLF',
    'Olefin': 'OLE',
    'PBI': 'PBI',
    'Polyacrylate': 'PAC',
    'Spandex': 'SPX',
    'Sulfar': 'SUL',
    'Rayon': 'RAY',
    'Viscose': 'VIS',
    'Lyocell': 'LYO',
    'Modal': 'MOD',
    'Acetate': 'ACT'
    
    # ADD MORE ABBREVIATIONS HERE AS NEEDED
}

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def extract_fibre_type_from_filename(filename):
    """
    Extract fiber type from filename.
    
    Examples:
    - "Cotton - 1.csv" -> "Cotton"
    - "Acrylic 1.csv" -> "Acrylic"
    - "Silk - Test 1.csv" -> "Silk"
    
    Parameters:
    -----------
    filename : str
        Filename of the raw CSV file
    
    Returns:
    --------
    str
        Extracted fiber type
    """
    # Remove file extension
    name = Path(filename).stem
    
    # Match alphabetic characters at the start (may include spaces)
    match = re.match(r'^([a-zA-Z\s]+)', name)
    
    if match:
        return match.group(1).strip()
    
    return "Unknown"


def extract_header_info(csv_file):
    """
    Extract Source and Binder Location from CSV header.
    
    Expected header format:
    Created as New Dataset,Forensic,1
    (Column 1 = instrument metadata [ignored], Column 2 = Source, Column 3 = Binder ID)
    
    Parameters:
    -----------
    csv_file : Path
        Path to CSV file
    
    Returns:
    --------
    tuple : (source, binder_id)
    """
    with open(csv_file, 'r') as f:
        first_line = f.readline().strip()
    
    # Parse header: "Instrument Info,Source Name,Binder Location"
    # Skip first column (instrument metadata), read columns 2 and 3
    header_parts = first_line.split(',')
    
    source = header_parts[1].strip() if len(header_parts) > 1 else ''
    binder_id = header_parts[2].strip() if len(header_parts) > 2 else ''
    
    return source, binder_id


def determine_hierarchical_type(fibre_name):
    """
    Determine hierarchical Type based on fiber name.
    
    Parameters:
    -----------
    fibre_name : str
        Fiber name (e.g., 'Cotton', 'Nylon', 'Acrylic')
    
    Returns:
    --------
    str
        Hierarchical type
    """
    if fibre_name in CELLULOSE_FIBRES:
        return 'Cellulose'
    elif fibre_name in PROTEIN_FIBRES:
        return 'Protein'
    elif fibre_name in REGENERATED_CELLULOSE:
        return 'Regenerated Cellulose'
    elif fibre_name in ACETATE_FIBRES:
        return 'Cellulose Acetate'
    elif fibre_name == 'Nylon':
        return 'Polyamide'
    elif fibre_name == 'Acrylic':
        return 'Acrylic'
    elif fibre_name == 'Modacrylic':
        return 'Modacrylic'
    elif fibre_name == 'Olefin':
        return 'Polyolefin'
    elif fibre_name == 'Aramid':
        return 'Aramid'
    elif fibre_name == 'Spandex':
        return 'Elastane'
    else:
        # For other fibers (Polyester, Carbon, etc.), use fiber name as type
        return fibre_name


def extract_subtype_from_remarks(remarks, fibre_name):
    """
    Extract fiber subtype from remarks/details.
    
    Uses standardized subtypes based on polymer composition and industry terminology.
    
    Parameters:
    -----------
    remarks : str
        Remarks or details text
    fibre_name : str
        Name of the fiber (e.g., 'Cotton', 'Rayon', 'Polyester')
    
    Returns:
    --------
    str
        Standardized subtype
    """
    remarks_lower = str(remarks).lower() if not pd.isna(remarks) else ''
    
    # For natural fibers (Cellulose or Protein), the subtype is the fiber name itself
    if fibre_name in CELLULOSE_FIBRES or fibre_name in PROTEIN_FIBRES:
        return fibre_name
    
    # Polyester subtypes
    if fibre_name == 'Polyester':
        return 'PET'
    
    # Nylon (Polyamide) subtypes
    if fibre_name == 'Nylon':
        if 'pa 6,6' in remarks_lower or 'pa 66' in remarks_lower or 'pa-66' in remarks_lower:
            return 'PA 6,6'
        elif 'pa 6' in remarks_lower or 'pa-6' in remarks_lower:
            return 'PA 6'
        else:
            return 'Unspecified PA Copolymer'
    
    # Acrylic subtypes (≥ 85% acrylonitrile)
    if fibre_name == 'Acrylic':
        has_pan = 'pan' in remarks_lower
        has_mma = 'mma' in remarks_lower
        has_ma = ' ma' in remarks_lower or 'ma ' in remarks_lower or '+ ma' in remarks_lower or 'ma+' in remarks_lower or 'ma(' in remarks_lower
        has_aa = ' aa' in remarks_lower or 'aa ' in remarks_lower or '+ aa' in remarks_lower or 'aa+' in remarks_lower or 'aa(' in remarks_lower
        has_va = False
        if 'va' in remarks_lower and 'vdc' not in remarks_lower and 'vc' not in remarks_lower:
            if ' va' in remarks_lower or 'va ' in remarks_lower or '+ va' in remarks_lower or 'va+' in remarks_lower:
                has_va = True
        
        # Determine specific copolymer type
        if has_pan and has_aa and has_ma:
            return 'PAN/AA/MA'
        elif has_pan and has_mma and has_ma:
            return 'PAN/MMA/MA'
        elif has_pan and has_mma:
            return 'PAN/MMA'
        elif has_pan and has_ma and not has_mma:
            return 'PAN/MA'
        elif has_pan and has_aa:
            return 'PAN/AA'
        elif has_pan and has_va:
            return 'PAN/VA'
        else:
            return 'Unspecified PAN (Acrylic) Copolymer'
    
    # Modacrylic subtypes (35-85% acrylonitrile)
    if fibre_name == 'Modacrylic':
        has_pan = 'pan' in remarks_lower
        has_vdc = 'vdc' in remarks_lower
        has_va = False
        has_vc = False
        has_vbr = 'vbr' in remarks_lower
        
        if 'va' in remarks_lower and 'vdc' not in remarks_lower:
            if ' va' in remarks_lower or 'va ' in remarks_lower or '+ va' in remarks_lower or 'va+' in remarks_lower:
                has_va = True
        
        if ' vc' in remarks_lower or 'vc ' in remarks_lower or '+ vc' in remarks_lower or 'vc(' in remarks_lower or 'vc.' in remarks_lower:
            has_vc = True
        
        # Determine specific copolymer type
        if has_pan and has_va and has_vc:
            return 'PAN/VA/VC'
        elif has_pan and has_vdc:
            return 'PAN/VDC'
        elif has_pan and has_vc:
            return 'PAN/VC'
        elif has_pan and has_vbr:
            return 'PAN/VBr'
        elif has_vdc:
            return 'PAN/VDC'
        else:
            return 'Unspecified PAN (Modacrylic) Copolymer'
    
    # Olefin (Polyolefin) subtypes
    if fibre_name == 'Olefin':
        if 'polypropylene' in remarks_lower or 'pp' in remarks_lower:
            return 'PP'
        elif 'polyethylene' in remarks_lower or 'pe' in remarks_lower:
            return 'PE'
        else:
            return 'Unspecified Polyolefin Type'
    
    # Aramid subtypes
    if fibre_name == 'Aramid':
        if 'kevlar' in remarks_lower:
            return 'Para-Aramid'
        elif 'nomex' in remarks_lower:
            return 'Meta-Aramid'
        else:
            return 'Unspecified Aramid Type'
    
    # Acetate subtypes
    if fibre_name == 'Acetate':
        if 'triacetate' in remarks_lower:
            return 'Cellulose Triacetate'
        elif 'secondary' in remarks_lower or 'diacetate' in remarks_lower:
            return 'Cellulose Diacetate'
        else:
            return 'Unspecified Cellulose Acetate Type'
    
    # Regenerated cellulose subtypes
    if fibre_name in REGENERATED_CELLULOSE:
        if fibre_name == 'Viscose':
            return 'Viscose'
        elif fibre_name == 'Lyocell':
            return 'Lyocell'
        elif fibre_name == 'Modal':
            return 'Modal'
        elif fibre_name == 'Rayon':
            if 'viscose' in remarks_lower:
                return 'Viscose'
            elif 'lyocell' in remarks_lower:
                return 'Lyocell'
            elif 'modal' in remarks_lower:
                return 'Modal'
            else:
                return 'Unspecified Regenerated Cellulose Polymer'
        else:
            return fibre_name
    
    # Spandex (Elastane) subtypes
    if fibre_name == 'Spandex':
        return 'PU'
    
    # Other man-made fiber types - return fiber name as subtype
    if fibre_name in ['Carbon', 'Chlorofiber', 'PBI', 'Polyacrylate', 'Sulfar']:
        return fibre_name
    
    # Default: return fiber name
    return fibre_name


def generate_sample_id(source, fibre_type, binder_id):
    """
    Generate Sample_ID with structured format: CCC_TTT_NNN
    
    Parameters:
    -----------
    source : str
        Source name (used to determine collection code)
    fibre_type : str
        Fiber type name (e.g., 'Cotton', 'Nylon', 'Acrylic')
    binder_id : str
        Binder/physical label ID
    
    Returns:
    --------
    str
        Generated Sample_ID in format CCC_TTT_NNN
    """
    # Determine collection code (CCC) based on source
    # ADD MORE SOURCE MAPPINGS HERE AS NEEDED
    source_lower = source.lower()
    
    if 'microtrace' in source_lower or 'forensic' in source_lower:
        collection_code = 'MTF'  # Microtrace Forensic Fiber Reference Collection
    elif 'arbidar' in source_lower or 'natural' in source_lower:
        collection_code = 'MTA'  # Arbidar Natural Fiber Collection
    elif 'bio-couture' in source_lower or 'biocouture' in source_lower:
        collection_code = 'BIO'  # Bio-Couture
    elif 'unusuwul' in source_lower:
        collection_code = 'UNU'  # UNUSUWUL
    else:
        collection_code = 'INT'  # Internal/Other collection
    
    # Get material code (TTT)
    material_code = FIBRE_ABBREVIATIONS.get(fibre_type, 'UNK')
    
    # Generate numerical code (NNN)
    # Convert binder_id to 3-digit format
    # "1.1" -> "011", "12.1" -> "121", "5" -> "005", "119" -> "119"
    numerical_part = binder_id.replace('.', '')
    numerical_code = numerical_part.zfill(3)
    
    # Construct Sample_ID
    return f"{collection_code}_{material_code}_{numerical_code}"


###############################################################################
# PREPROCESSING FUNCTIONS
###############################################################################

def als_baseline_correction(y, lam=1e6, p=0.001, max_iter=10):
    """
    Asymmetric Least Squares baseline correction.
    
    Reference: Eilers & Boelens (2005). Baseline Correction with Asymmetric Least Squares Smoothing.
    
    Parameters:
    -----------
    y : array-like
        Input spectrum intensities
    lam : float
        Smoothness parameter (typical range: 10^2 to 10^9)
        Higher values = smoother baseline
    p : float
        Asymmetry parameter (typical range: 0.001 to 0.1)
        Lower values = baseline fits valleys better
    max_iter : int
        Maximum iterations
        
    Returns:
    --------
    baseline : ndarray
        Estimated baseline
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L-2, L))
    w = np.ones(L)
    
    for i in range(max_iter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.T.dot(D)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
        
    return z


def apply_als_baseline_correction_transmittance(transmittance):
    """
    Apply ALS baseline correction to transmittance spectrum.
    
    Workflow: %T → Absorbance → ALS correction → back to %T
    
    Parameters:
    -----------
    transmittance : ndarray
        Transmittance spectrum (%T)
    
    Returns:
    --------
    corrected_transmittance : ndarray
        Baseline-corrected transmittance (%T)
    """
    # Convert %T to Absorbance: A = -log10(T/100)
    absorbance = -np.log10((transmittance + 1e-10) / 100.0)
    
    # Apply ALS baseline correction on absorbance
    baseline_abs = als_baseline_correction(absorbance, lam=1e6, p=0.001, max_iter=10)
    
    # Subtract baseline from absorbance
    corrected_abs = absorbance - baseline_abs
    
    # Convert corrected absorbance back to %T: T = 100 * 10^(-A)
    corrected_transmittance = 100.0 * np.power(10, -corrected_abs)
    
    return corrected_transmittance


def apply_snv_normalization(spectrum):
    """
    Apply Standard Normal Variate (SNV) normalization.
    
    SNV centers the spectrum to zero mean and scales to unit variance.
    
    Parameters:
    -----------
    spectrum : ndarray
        Input spectrum (baseline-corrected transmittance)
    
    Returns:
    --------
    snv_spectrum : ndarray
        SNV-normalized spectrum
    """
    # Convert to absorbance first for SNV
    absorbance = -np.log10((spectrum + 1e-10) / 100.0)
    
    # Apply SNV: (x - mean) / std
    mean = np.mean(absorbance)
    std = np.std(absorbance)
    
    if std > 0:
        snv_absorbance = (absorbance - mean) / std
    else:
        snv_absorbance = absorbance - mean
    
    # Convert back to transmittance-like format
    # Note: SNV-normalized data is no longer true %T but normalized intensities
    snv_spectrum = 100.0 * np.power(10, -snv_absorbance)
    
    return snv_spectrum


def apply_first_derivative(spectrum, window_length=15, polyorder=3):
    """
    Apply Savitzky-Golay 1st derivative.
    
    Parameters:
    -----------
    spectrum : ndarray
        Input spectrum (SNV-normalized)
    window_length : int
        Window size for Savitzky-Golay filter (must be odd)
    polyorder : int
        Polynomial order for Savitzky-Golay filter
    
    Returns:
    --------
    derivative_spectrum : ndarray
        1st derivative spectrum
    """
    # Convert to absorbance
    absorbance = -np.log10((spectrum + 1e-10) / 100.0)
    
    # Apply Savitzky-Golay 1st derivative
    derivative = savgol_filter(absorbance, window_length=window_length, 
                               polyorder=polyorder, deriv=1)
    
    # Return derivative values (not converted back to %T format)
    # Derivative values can be positive or negative
    return derivative


###############################################################################
# MAIN PROCESSING PIPELINE
###############################################################################

def main():
       
    # Define paths
    input_folder = Path('exported_csvs')
    metadata_folder = Path('metadata')
    output_folder = Path('ml_datasets')
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(exist_ok=True)
    
    # Load existing all_samples_metadata.csv
    all_samples_metadata_path = metadata_folder / 'all_samples_metadata.csv'
    if not all_samples_metadata_path.exists():
        print(f"ERROR: {all_samples_metadata_path} not found!")
        print("Please ensure the metadata file exists before running this script.")
        return
    
    all_samples_df = pd.read_csv(all_samples_metadata_path)
    print(f"-- Loaded all_samples_metadata.csv: {len(all_samples_df)} samples")
    print()
    
    # Get all CSV files from input folder
    csv_files = sorted(list(input_folder.glob("*.csv")))
    print(f"Found {len(csv_files)} CSV files in {input_folder}/")
    print()
    
    # Storage for datasets
    ml_records = []
    metadata_records = []
    seen_samples = {}
    
    # Counters
    spectrum_counter = 1
    sample_counters = {}
    
    # Reference wavenumbers
    wavenumbers_ref = None
    
    print("Processing raw spectral files...")
    print("-" * 80)
    
    for csv_file in csv_files:
        try:
            # Extract fiber type from filename
            fibre_type = extract_fibre_type_from_filename(csv_file.name)
            
            # Skip if fiber type not in mapping
            if fibre_type not in FIBRE_ORIGIN:
                print(f"SKIP: {csv_file.name} - Unknown fiber type: {fibre_type}")
                continue
            
            # Extract source and binder_id from header
            source, binder_id = extract_header_info(csv_file)
            
            if not binder_id:
                print(f"SKIP: {csv_file.name} - No Binder ID found in header")
                continue
            
            # Determine origin
            origin = FIBRE_ORIGIN[fibre_type]
            
            # Determine hierarchical Type
            fibre_type_hierarchical = determine_hierarchical_type(fibre_type)
            
            # Read spectral data (skip first row which is header)
            df = pd.read_csv(csv_file, skiprows=1, header=0)
            
            # Extract wavenumbers and transmittance
            wavenumbers = df['cm-1'].values
            transmittance = df['%T'].values
            
            # Set reference wavenumbers from first file
            if wavenumbers_ref is None:
                wavenumbers_ref = wavenumbers
            
            # Generate Sample_ID
            sample_id = generate_sample_id(source, fibre_type, binder_id)
            
            # Match with existing metadata from all_samples_metadata.csv
            match = all_samples_df[all_samples_df['Sample_ID'] == sample_id]
            
            if match.empty:
                print(f"WARNING: {csv_file.name} - Sample_ID {sample_id} not found in all_samples_metadata.csv")
                # Still process but with minimal metadata
                details = ''
                fibre_subtype = extract_subtype_from_remarks('', fibre_type)
            else:
                # Use metadata from existing record
                details = str(match.iloc[0]['Details']).strip()
                fibre_subtype = str(match.iloc[0]['Subtype']).strip()
            
            # Track replicas for this sample
            if sample_id not in sample_counters:
                sample_counters[sample_id] = 0
            sample_counters[sample_id] += 1
            replica = sample_counters[sample_id]
            
            # Create Spectrum_ID
            spectrum_id = f"{spectrum_counter:04d}"
            spectrum_counter += 1
            
            # Build ML record
            ml_record = {
                'Spectrum_ID': spectrum_id,
                'Sample_ID': sample_id,
                'Replica': replica,
                'Origin': origin,
                'Type': fibre_type_hierarchical,
                'Subtype': fibre_subtype
            }
            
            # Add transmittance values for each wavenumber
            for wn, trans in zip(wavenumbers, transmittance):
                ml_record[f"{wn:.1f}"] = trans
            
            ml_records.append(ml_record)
            
            # Add to metadata (only once per unique sample)
            if sample_id not in seen_samples:
                metadata_record = {
                    'Sample_ID': sample_id,
                    'Source': source,
                    'Binder_ID': binder_id,
                    'Origin': origin,
                    'Type': fibre_type_hierarchical,
                    'Subtype': fibre_subtype,
                    'Details': details if details else 'N/A'
                }
                metadata_records.append(metadata_record)
                seen_samples[sample_id] = metadata_record
            
            print(f"OK: {csv_file.name:30s} -> {sample_id:20s} (Replica {replica})")
            
        except Exception as e:
            print(f"ERROR: {csv_file.name} - {str(e)}")
            continue
    
    print("=" * 80)
    print(f"Processing complete!")
    print(f"Total spectra processed: {len(ml_records)}")
    print(f"Unique samples: {len(metadata_records)}")
    print()
    
    # Create DataFrames
    ml_dataset = pd.DataFrame(ml_records)
    metadata_dataset = pd.DataFrame(metadata_records)
    
    # Get spectral columns
    spectral_columns = [col for col in ml_dataset.columns if col not in 
                       ['Spectrum_ID', 'Sample_ID', 'Replica', 'Origin', 'Type', 'Subtype']]
    
    print("=" * 80)
    print("STEP 1: Exporting Raw Transmittance Dataset")
    print("=" * 80)
    
    # Export raw transmittance dataset
    raw_output_path = output_folder / 'feature_matrix_raw_transmittance.csv'
    ml_dataset.to_csv(raw_output_path, index=False)
    print(f"Exported: {raw_output_path}")
    print(f"  Shape: {ml_dataset.shape}")
    print(f"  Size: {raw_output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    
    # Export scanned samples metadata
    metadata_output_path = output_folder / 'scanned_samples_metadata.csv'
    metadata_dataset.to_csv(metadata_output_path, index=False)
    print(f"Exported: {metadata_output_path}")
    print(f"  Shape: {metadata_dataset.shape}")
    print(f"  Size: {metadata_output_path.stat().st_size / 1024:.2f} KB")
    print()
    
    print("=" * 80)
    print("STEP 2: Applying ALS Baseline Correction")
    print("=" * 80)
    
    # Apply ALS baseline correction
    baseline_corrected_spectra = []
    
    for idx, row in ml_dataset.iterrows():
        transmittance = row[spectral_columns].values.astype(np.float64)
        corrected = apply_als_baseline_correction_transmittance(transmittance)
        baseline_corrected_spectra.append(corrected)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(ml_dataset)} spectra...")
    
    # Create baseline-corrected dataset
    ml_dataset_baseline = ml_dataset[['Spectrum_ID', 'Sample_ID', 'Replica', 
                                      'Origin', 'Type', 'Subtype']].copy()
    
    for i, col in enumerate(spectral_columns):
        ml_dataset_baseline[col] = [spec[i] for spec in baseline_corrected_spectra]
    
    # Export baseline-corrected dataset
    baseline_output_path = output_folder / 'feature_matrix_baseline_corrected.csv'
    ml_dataset_baseline.to_csv(baseline_output_path, index=False)
    print(f"\nExported: {baseline_output_path}")
    print(f"  Shape: {ml_dataset_baseline.shape}")
    print(f"  Size: {baseline_output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    
    print("=" * 80)
    print("STEP 3: Applying ALS + SNV + D1 Pipeline")
    print("=" * 80)
    
    # Apply full preprocessing pipeline (ALS + SNV + D1)
    preprocessed_spectra = []
    
    for idx, row in ml_dataset.iterrows():
        # Start with raw transmittance
        transmittance = row[spectral_columns].values.astype(np.float64)
        
        # Step 1: ALS baseline correction
        baseline_corrected = apply_als_baseline_correction_transmittance(transmittance)
        
        # Step 2: SNV normalization
        snv_normalized = apply_snv_normalization(baseline_corrected)
        
        # Step 3: 1st derivative (Savitzky-Golay)
        derivative = apply_first_derivative(snv_normalized, window_length=15, polyorder=3)
        
        preprocessed_spectra.append(derivative)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(ml_dataset)} spectra...")
    
    # Create preprocessed dataset
    ml_dataset_preprocessed = ml_dataset[['Spectrum_ID', 'Sample_ID', 'Replica', 
                                         'Origin', 'Type', 'Subtype']].copy()
    
    for i, col in enumerate(spectral_columns):
        ml_dataset_preprocessed[col] = [spec[i] for spec in preprocessed_spectra]
    
    # Export preprocessed dataset
    preprocessed_output_path = output_folder / 'feature_matrix_preprocessed.csv'
    ml_dataset_preprocessed.to_csv(preprocessed_output_path, index=False)
    print(f"\nExported: {preprocessed_output_path}")
    print(f"  Shape: {ml_dataset_preprocessed.shape}")
    print(f"  Size: {preprocessed_output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    
    print("=" * 80)
    print("PIPELINE COMPLETE!")
    print("-" * 80)
    print("\nGenerated Files:")
    print(f"1. {raw_output_path.name}")
    print(f"   - Raw transmittance data (%T)")
    print(f"2. {baseline_output_path.name}")
    print(f"   - ALS baseline-corrected data (%T)")
    print(f"3. {preprocessed_output_path.name}")
    print(f"   - Fully preprocessed data (ALS + SNV + D1)")
    print(f"4. {metadata_output_path.name}")
    print(f"   - Metadata for scanned samples")
    print()
    print("Summary Statistics:")
    print(f"  Total spectra: {len(ml_dataset)}")
    print(f"  Unique samples: {len(metadata_dataset)}")
    print(f"  Natural fibers: {len(ml_dataset[ml_dataset['Origin'] == 'Natural'])}")
    print(f"  Man-made fibers: {len(ml_dataset[ml_dataset['Origin'] == 'Man-made'])}")
    print()
    print("Type Distribution:")
    print(ml_dataset['Type'].value_counts())
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
