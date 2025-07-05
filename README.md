# PhantomStroke

A machine learning-based system for keystroke inference from gyroscope sensor data using ultrasonic signals. This project implements the attack scenario described in the research where ultrasonic signals (18-22 kHz) emitted by laptop speakers can be detected by smartphone gyroscopes and used to infer typed keystrokes.

## Overview

PhantomStroke uses machine learning models to classify gyroscope vibration patterns into corresponding keystrokes. The system includes:

- **Data Simulation**: Generates realistic gyroscope data for different keystrokes
- **Feature Extraction**: Extracts time-domain, frequency-domain, and wavelet features
- **ML Models**: Implements CNN, SVM, and Random Forest classifiers
- **Ensemble Learning**: Combines multiple models for improved accuracy
- **Real-time Inference**: Predicts keystrokes from new gyroscope data

## Architecture

The system follows the attack flow described in `flowchart.md`:

1. **Signal Generation**: Keystrokes trigger ultrasonic signals (18-22 kHz)
2. **Sensor Reading**: Smartphone gyroscope detects MEMS vibrations
3. **Feature Extraction**: Process gyroscope data (FFT, statistical features)
4. **ML Inference**: Trained models predict keystrokes
5. **Output**: Inferred keystrokes are returned

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Soorya005/PhantomStroke.git
cd PhantomStroke
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

Train all ML models on simulated gyroscope data:

```bash
python train_models.py
```

Options:
- `--output_dir`: Output directory for results (default: 'output')
- `--n_samples`: Number of samples per keystroke (default: 100)
- `--keystrokes`: Comma-separated keystrokes to train on (default: 'a,b,c,d,e,f,g,h,i,j,space,enter')
- `--no_plots`: Skip generating visualizations

Example:
```bash
python train_models.py --n_samples 150 --keystrokes "a,e,i,o,u,space"
```

### Inference

Use trained models to predict keystrokes:

```bash
# Test a specific keystroke
python inference.py --keystroke a

# Interactive mode
python inference.py --interactive

# Use specific model
python inference.py --keystroke a --model random_forest
```

### Custom Data Generation

Generate custom gyroscope datasets:

```python
from src.phantom_stroke.data.simulator import GyroscopeDataSimulator

# Create simulator
simulator = GyroscopeDataSimulator(sampling_rate=100, window_duration=0.2)

# Generate single keystroke
signal = simulator.generate_keystroke_signal('a', noise_level=0.1)

# Generate dataset
keystrokes = ['a', 'e', 'i', 'space']
features, labels = simulator.generate_dataset(keystrokes, samples_per_key=50)
```

### Feature Extraction

Extract features from gyroscope data:

```python
from src.phantom_stroke.features.extractor import GyroscopeFeatureExtractor

# Create extractor
extractor = GyroscopeFeatureExtractor(sampling_rate=100)

# Extract features
features = extractor.extract_all_features(gyroscope_data)
```

## Project Structure

```
PhantomStroke/
├── src/
│   └── phantom_stroke/
│       ├── data/
│       │   └── simulator.py          # Gyroscope data simulation
│       ├── features/
│       │   └── extractor.py          # Feature extraction
│       ├── models/
│       │   └── classifiers.py        # ML models (CNN, SVM, RF)
│       └── utils/
│           └── visualization.py      # Plotting and utilities
├── train_models.py                   # Main training script
├── inference.py                      # Inference script
├── requirements.txt                  # Python dependencies
├── flowchart.md                      # Attack flow documentation
├── flow.md                           # System overview
└── README.md                         # This file
```

## ML Models

### Random Forest Classifier
- **Features**: Time/frequency domain features from gyroscope data
- **Advantages**: Fast training, feature importance analysis
- **Use case**: Baseline model with good interpretability

### Support Vector Machine (SVM)
- **Features**: Scaled feature vectors
- **Kernel**: RBF kernel with optimized parameters
- **Advantages**: Good generalization, effective in high dimensions

### Convolutional Neural Network (CNN)
- **Input**: Raw gyroscope time series (window_length, 3)
- **Architecture**: 1D convolutions + global pooling + dense layers
- **Advantages**: Automatic feature learning, high accuracy

### Ensemble Model
Combines predictions from multiple models using weighted voting for improved accuracy.

## Features

The system extracts comprehensive features from gyroscope data:

**Time Domain:**
- Statistical moments (mean, std, skewness, kurtosis)
- Peak detection and analysis
- Zero-crossing rate
- Cross-axis correlations

**Frequency Domain:**
- FFT-based spectral features
- Ultrasonic frequency band analysis (18-22 kHz)
- Spectral centroid, bandwidth, rolloff
- Frequency band power ratios

**Wavelet Domain (optional):**
- Wavelet decomposition coefficients
- Multi-resolution analysis

## Results

After training, the system generates:

- **Model Performance**: Accuracy, precision, recall, F1-score
- **Visualizations**: Confusion matrices, feature importance, training curves
- **Saved Models**: Trained models for inference
- **Metrics**: Detailed performance analysis in JSON format

Example output structure:
```
output/
├── models/                 # Trained model files
├── plots/                  # Visualizations
├── metrics/               # Performance metrics
└── data/                  # Generated datasets
```

## Research Context

This implementation is based on research into acoustic side-channel attacks where:

1. Malware on a laptop emits ultrasonic signals (18-22 kHz) for each keystroke
2. A nearby smartphone with a malicious webpage open captures these signals via gyroscope
3. Machine learning models classify the gyroscope vibration patterns into keystrokes

**Note**: This project is for academic and research purposes only. See LICENSE for usage restrictions.

## Dependencies

- Python 3.7+
- NumPy, SciPy (signal processing)
- Scikit-learn (traditional ML models)
- TensorFlow/Keras (neural networks)
- Matplotlib, Seaborn (visualization)
- Pandas (data manipulation)

## License

Copyright (c) 2025 Soorya A P, Vyshnav Pradeep, Sooraj. All rights reserved.
This software is for academic and research purposes only. See LICENSE for details.