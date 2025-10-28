# MIMIC-sepsis

This repository contains code for analyzing and predicting sepsis using the MIMIC (Medical Information Mart for Intensive Care) dataset. The project implements various machine learning models to predict sepsis onset and analyze patient trajectories.

## Citation

If you use this dataset or code in your research, please cite our paper:

```bibtex
@inproceedings{huang2025mimic,
  title={MIMIC-Sepsis: A Curated Benchmark for Modeling and Learning from Sepsis Trajectories in the ICU},
  author={Huang, Yong and Yang, Zhongqi and Rahmani, Amir M},
  booktitle={IEEE-EMBS International Conference on Biomedical and Health Informatics 2025},
  year={2025}
}
```


## Project Overview

The project focuses on:
- Processing and analyzing MIMIC-III/IV data for sepsis prediction
- Implementing multiple machine learning models for time series prediction
- Analyzing patient trajectories and treatment effects
- Benchmarking different model architectures

## Features

- Multiple model implementations:
  - Linear models
  - LSTM-based models
  - Transformer-based models
- Comprehensive data processing pipeline
- Treatment effect analysis
- Various clinical measurements tracking:
  - Vital signs
  - Laboratory results
  - Medications
  - Fluid balance
  - Mechanical ventilation
  - Microbiology results

## Repository Structure

```
src/
├── benchmark.py           # Main benchmarking script
├── linear_model.py        # Linear time series model implementation
├── transformer_model.py   # Transformer-based model implementation
├── lstm_model.py          # LSTM-based model implementation
├── data_processor.py      # Data processing utilities
├── format_traj.py         # Patient trajectory formatting
├── metrics.py             # Evaluation metrics
├── init_preprocess.py     # Initial data preprocessing
└── Various clinical data processing modules:
    ├── demog.py           # Demographic data processing
    ├── lab_ce.py          # Lab results processing
    ├── mechvent.py        # Mechanical ventilation data
    ├── vaso.py            # Vasopressor data
    ├── uo.py              # Urine output data
    ├── fluid.py           # Fluid balance data
    └── abx.py             # Antibiotic data
```

## Requirements

The project requires Python 3.x and the following main dependencies:
- pandas
- numpy
- torch
- scikit-learn
- matplotlib
- seaborn

## Usage

The data processing pipeline consists of three main steps that must be executed in sequence:

1. Run SQL queries to extract raw data from MIMIC database:
```bash
# Execute SQL queries to extract necessary tables from MIMIC database
# This step requires access to the MIMIC database and appropriate credentials
python src/demog.py
python src/ce.py
python src/lab_ce.py
python src/mechvent.py
python src/vaso.py
python src/uo.py
python src/fluid.py
python src/abx.py

```

2. Initial Data Preprocessing:
```bash
python src/init_preprocess.py
```
This script processes the raw data and creates intermediate files for further processing.

3. Format Patient Trajectories:
```bash
python src/format_traj.py
```
This script creates the final formatted trajectories used for model training and evaluation.

### Output Files

After running the complete pipeline, you will have access to the following key files:

1. Raw data files (from SQL queries):
   - Patient demographics
   - ICU stays information
   - Vital signs
   - Laboratory measurements
   - Medication administrations
   - Fluid balance records
   - Microbiology results

2. Processed data files (from init_preprocess.py):
   - Cleaned and normalized clinical measurements
   - Merged patient records
   - Time-aligned data points

3. Final trajectory files (from format_traj.py):
   - Formatted patient trajectories ready for model input
   - Treatment sequences
   - Outcome labels
   - Feature matrices

## Running Benchmarks

The benchmark script supports various configurations:
- Different model types (linear, LSTM, transformer)
- Treatment effect inclusion/exclusion
- Custom prediction horizons
- Regularization options

```python
python src/benchmark.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MIMIC-III/IV database for providing the clinical data
- Contributors and researchers who have worked on this project 