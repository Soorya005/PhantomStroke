"""
Feature extraction module for PhantomStroke.
Extracts meaningful features from gyroscope data for ML model training.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
import pandas as pd


class GyroscopeFeatureExtractor:
    """Extracts features from gyroscope time series data."""
    
    def __init__(self, sampling_rate: int = 100):
        """
        Initialize the feature extractor.
        
        Args:
            sampling_rate: Sampling rate of the gyroscope data in Hz
        """
        self.sampling_rate = sampling_rate
        
    def extract_time_domain_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract time domain statistical features.
        
        Args:
            data: Gyroscope data of shape (n_samples, 3) for X, Y, Z axes
            
        Returns:
            Dictionary of time domain features
        """
        features = {}
        
        for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
            axis_data = data[:, axis_idx]
            
            # Basic statistical features
            features[f'{axis_name}_mean'] = np.mean(axis_data)
            features[f'{axis_name}_std'] = np.std(axis_data)
            features[f'{axis_name}_var'] = np.var(axis_data)
            features[f'{axis_name}_min'] = np.min(axis_data)
            features[f'{axis_name}_max'] = np.max(axis_data)
            features[f'{axis_name}_range'] = np.max(axis_data) - np.min(axis_data)
            features[f'{axis_name}_median'] = np.median(axis_data)
            
            # Percentiles
            features[f'{axis_name}_q25'] = np.percentile(axis_data, 25)
            features[f'{axis_name}_q75'] = np.percentile(axis_data, 75)
            features[f'{axis_name}_iqr'] = features[f'{axis_name}_q75'] - features[f'{axis_name}_q25']
            
            # Higher order moments
            features[f'{axis_name}_skewness'] = self._skewness(axis_data)
            features[f'{axis_name}_kurtosis'] = self._kurtosis(axis_data)
            
            # Energy and power
            features[f'{axis_name}_energy'] = np.sum(axis_data ** 2)
            features[f'{axis_name}_rms'] = np.sqrt(np.mean(axis_data ** 2))
            
            # Zero crossing rate
            features[f'{axis_name}_zcr'] = self._zero_crossing_rate(axis_data)
            
            # Peak detection
            peaks, _ = signal.find_peaks(np.abs(axis_data))
            features[f'{axis_name}_num_peaks'] = len(peaks)
            if len(peaks) > 0:
                features[f'{axis_name}_peak_mean'] = np.mean(np.abs(axis_data[peaks]))
                features[f'{axis_name}_peak_std'] = np.std(np.abs(axis_data[peaks]))
            else:
                features[f'{axis_name}_peak_mean'] = 0
                features[f'{axis_name}_peak_std'] = 0
        
        # Cross-axis correlations
        features['xy_correlation'] = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
        features['xz_correlation'] = np.corrcoef(data[:, 0], data[:, 2])[0, 1]
        features['yz_correlation'] = np.corrcoef(data[:, 1], data[:, 2])[0, 1]
        
        # Overall magnitude features
        magnitude = np.sqrt(np.sum(data ** 2, axis=1))
        features['magnitude_mean'] = np.mean(magnitude)
        features['magnitude_std'] = np.std(magnitude)
        features['magnitude_max'] = np.max(magnitude)
        
        return features
    
    def extract_frequency_domain_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency domain features using FFT.
        
        Args:
            data: Gyroscope data of shape (n_samples, 3) for X, Y, Z axes
            
        Returns:
            Dictionary of frequency domain features
        """
        features = {}
        
        for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
            axis_data = data[:, axis_idx]
            
            # Compute FFT
            fft_values = fft(axis_data)
            fft_freqs = fftfreq(len(axis_data), 1/self.sampling_rate)
            
            # Take only positive frequencies
            positive_freq_idx = fft_freqs > 0
            fft_magnitude = np.abs(fft_values[positive_freq_idx])
            fft_freqs_pos = fft_freqs[positive_freq_idx]
            
            # Spectral features
            features[f'{axis_name}_spectral_centroid'] = np.sum(fft_freqs_pos * fft_magnitude) / np.sum(fft_magnitude)
            features[f'{axis_name}_spectral_bandwidth'] = np.sqrt(
                np.sum(((fft_freqs_pos - features[f'{axis_name}_spectral_centroid']) ** 2) * fft_magnitude) / 
                np.sum(fft_magnitude)
            )
            features[f'{axis_name}_spectral_rolloff'] = self._spectral_rolloff(fft_freqs_pos, fft_magnitude)
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(fft_magnitude)
            features[f'{axis_name}_dominant_freq'] = fft_freqs_pos[dominant_freq_idx]
            features[f'{axis_name}_dominant_freq_magnitude'] = fft_magnitude[dominant_freq_idx]
            
            # Frequency bands (relevant to ultrasonic range 18-22 kHz)
            ultrasonic_mask = (fft_freqs_pos >= 18) & (fft_freqs_pos <= 22)
            if np.any(ultrasonic_mask):
                features[f'{axis_name}_ultrasonic_power'] = np.sum(fft_magnitude[ultrasonic_mask] ** 2)
                features[f'{axis_name}_ultrasonic_peak'] = np.max(fft_magnitude[ultrasonic_mask])
            else:
                features[f'{axis_name}_ultrasonic_power'] = 0
                features[f'{axis_name}_ultrasonic_peak'] = 0
            
            # Low frequency power (0-10 Hz)
            low_freq_mask = (fft_freqs_pos >= 0) & (fft_freqs_pos <= 10)
            features[f'{axis_name}_low_freq_power'] = np.sum(fft_magnitude[low_freq_mask] ** 2)
            
            # Mid frequency power (10-25 Hz)
            mid_freq_mask = (fft_freqs_pos > 10) & (fft_freqs_pos <= 25)
            features[f'{axis_name}_mid_freq_power'] = np.sum(fft_magnitude[mid_freq_mask] ** 2)
            
            # High frequency power (25+ Hz)
            high_freq_mask = fft_freqs_pos > 25
            features[f'{axis_name}_high_freq_power'] = np.sum(fft_magnitude[high_freq_mask] ** 2)
            
            # Spectral entropy
            features[f'{axis_name}_spectral_entropy'] = self._spectral_entropy(fft_magnitude)
        
        return features
    
    def extract_wavelet_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract wavelet-based features.
        
        Args:
            data: Gyroscope data of shape (n_samples, 3) for X, Y, Z axes
            
        Returns:
            Dictionary of wavelet features
        """
        features = {}
        
        try:
            import pywt
            
            for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
                axis_data = data[:, axis_idx]
                
                # Wavelet decomposition
                coeffs = pywt.wavedec(axis_data, 'db4', level=4)
                
                # Features from approximation coefficients
                features[f'{axis_name}_wavelet_approx_energy'] = np.sum(coeffs[0] ** 2)
                features[f'{axis_name}_wavelet_approx_std'] = np.std(coeffs[0])
                
                # Features from detail coefficients
                for i, detail_coeffs in enumerate(coeffs[1:], 1):
                    features[f'{axis_name}_wavelet_detail_{i}_energy'] = np.sum(detail_coeffs ** 2)
                    features[f'{axis_name}_wavelet_detail_{i}_std'] = np.std(detail_coeffs)
                    
        except ImportError:
            # PyWavelets not available, skip wavelet features
            pass
        
        return features
    
    def extract_all_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract all available features from gyroscope data.
        
        Args:
            data: Gyroscope data of shape (n_samples, 3) for X, Y, Z axes
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # Time domain features
        features.update(self.extract_time_domain_features(data))
        
        # Frequency domain features
        features.update(self.extract_frequency_domain_features(data))
        
        # Wavelet features (if available)
        features.update(self.extract_wavelet_features(data))
        
        return features
    
    def extract_features_batch(self, data_batch: np.ndarray) -> np.ndarray:
        """
        Extract features from a batch of gyroscope data samples.
        
        Args:
            data_batch: Batch of gyroscope data of shape (n_samples, window_length, 3)
            
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        n_samples = data_batch.shape[0]
        
        # Extract features from first sample to determine feature count
        sample_features = self.extract_all_features(data_batch[0])
        feature_names = list(sample_features.keys())
        n_features = len(feature_names)
        
        # Initialize feature matrix
        features_matrix = np.zeros((n_samples, n_features))
        
        # Extract features for all samples
        for i in range(n_samples):
            sample_features = self.extract_all_features(data_batch[i])
            for j, feature_name in enumerate(feature_names):
                features_matrix[i, j] = sample_features.get(feature_name, 0)
        
        return features_matrix, feature_names
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if np.std(data) == 0:
            return 0
        return np.mean(((data - np.mean(data)) / np.std(data)) ** 3)
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if np.std(data) == 0:
            return 0
        return np.mean(((data - np.mean(data)) / np.std(data)) ** 4) - 3
    
    def _zero_crossing_rate(self, data: np.ndarray) -> float:
        """Calculate zero crossing rate."""
        zero_crossings = np.where(np.diff(np.sign(data)))[0]
        return len(zero_crossings) / len(data)
    
    def _spectral_rolloff(self, freqs: np.ndarray, magnitude: np.ndarray, rolloff_percent: float = 0.95) -> float:
        """Calculate spectral rolloff frequency."""
        cumulative_magnitude = np.cumsum(magnitude)
        total_magnitude = cumulative_magnitude[-1]
        rolloff_idx = np.where(cumulative_magnitude >= rolloff_percent * total_magnitude)[0]
        
        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        return freqs[-1]
    
    def _spectral_entropy(self, magnitude: np.ndarray) -> float:
        """Calculate spectral entropy."""
        # Normalize to get probability distribution
        magnitude_norm = magnitude / np.sum(magnitude)
        magnitude_norm = magnitude_norm[magnitude_norm > 0]  # Remove zeros
        
        if len(magnitude_norm) == 0:
            return 0
        
        return -np.sum(magnitude_norm * np.log2(magnitude_norm))


def create_feature_dataset(raw_data: np.ndarray, labels: np.ndarray, 
                         sampling_rate: int = 100) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create a feature dataset from raw gyroscope data.
    
    Args:
        raw_data: Raw gyroscope data of shape (n_samples, window_length, 3)
        labels: Labels for each sample
        sampling_rate: Sampling rate of the data
        
    Returns:
        Tuple of (feature_matrix, labels, feature_names)
    """
    extractor = GyroscopeFeatureExtractor(sampling_rate)
    feature_matrix, feature_names = extractor.extract_features_batch(raw_data)
    
    return feature_matrix, labels, feature_names


if __name__ == "__main__":
    # Demo usage
    from ..data.simulator import GyroscopeDataSimulator
    
    # Generate sample data
    simulator = GyroscopeDataSimulator()
    signal = simulator.generate_keystroke_signal('a')
    
    # Extract features
    extractor = GyroscopeFeatureExtractor()
    features = extractor.extract_all_features(signal)
    
    print(f"Extracted {len(features)} features:")
    for name, value in list(features.items())[:10]:  # Show first 10 features
        print(f"  {name}: {value:.4f}")