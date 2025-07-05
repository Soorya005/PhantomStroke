"""
Data simulation module for PhantomStroke.
Simulates gyroscope data corresponding to different keystrokes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import random


class GyroscopeDataSimulator:
    """Simulates gyroscope data for different keystrokes based on ultrasonic resonance patterns."""
    
    def __init__(self, sampling_rate: int = 100, window_duration: float = 0.2):
        """
        Initialize the gyroscope data simulator.
        
        Args:
            sampling_rate: Samples per second (Hz)
            window_duration: Duration of each keystroke window in seconds
        """
        self.sampling_rate = sampling_rate
        self.window_duration = window_duration
        self.window_samples = int(sampling_rate * window_duration)
        
        # Define keystroke patterns - each key has unique resonance characteristics
        # These represent different resonance patterns for common keys
        self.keystroke_patterns = {
            'a': {'freq': [19.2, 20.1], 'amplitude': [0.8, 1.2], 'phase': [0.1, 0.3]},
            'b': {'freq': [19.5, 20.3], 'amplitude': [0.9, 1.1], 'phase': [0.2, 0.4]},
            'c': {'freq': [19.8, 20.5], 'amplitude': [0.7, 1.3], 'phase': [0.0, 0.2]},
            'd': {'freq': [19.1, 19.9], 'amplitude': [0.8, 1.0], 'phase': [0.3, 0.5]},
            'e': {'freq': [20.0, 20.8], 'amplitude': [1.0, 1.4], 'phase': [0.1, 0.4]},
            'f': {'freq': [19.3, 20.2], 'amplitude': [0.6, 1.1], 'phase': [0.2, 0.3]},
            'g': {'freq': [19.7, 20.4], 'amplitude': [0.9, 1.2], 'phase': [0.0, 0.1]},
            'h': {'freq': [19.4, 20.0], 'amplitude': [0.8, 1.1], 'phase': [0.4, 0.6]},
            'i': {'freq': [20.1, 20.9], 'amplitude': [0.7, 1.0], 'phase': [0.1, 0.2]},
            'j': {'freq': [19.6, 20.3], 'amplitude': [0.9, 1.3], 'phase': [0.3, 0.5]},
            'space': {'freq': [18.8, 19.5], 'amplitude': [1.2, 1.8], 'phase': [0.0, 0.1]},
            'enter': {'freq': [20.5, 21.0], 'amplitude': [1.5, 2.0], 'phase': [0.2, 0.4]},
        }
        
    def generate_keystroke_signal(self, keystroke: str, noise_level: float = 0.1) -> np.ndarray:
        """
        Generate simulated gyroscope data for a specific keystroke.
        
        Args:
            keystroke: The keystroke character
            noise_level: Amount of noise to add (0.0 to 1.0)
            
        Returns:
            Array of shape (window_samples, 3) representing X, Y, Z gyroscope data
        """
        if keystroke not in self.keystroke_patterns:
            keystroke = 'a'  # Default pattern
            
        pattern = self.keystroke_patterns[keystroke]
        t = np.linspace(0, self.window_duration, self.window_samples)
        
        # Generate base signal with multiple frequency components
        signal = np.zeros((self.window_samples, 3))
        
        for axis in range(3):
            # Primary frequency component
            freq1 = np.random.uniform(*pattern['freq'])
            amp1 = np.random.uniform(*pattern['amplitude'])
            phase1 = np.random.uniform(*pattern['phase'])
            
            # Secondary frequency component (harmonic)
            freq2 = freq1 * 1.5 + np.random.normal(0, 0.5)
            amp2 = amp1 * 0.3
            phase2 = phase1 + np.random.uniform(0, np.pi)
            
            # Combine frequency components
            signal[:, axis] = (
                amp1 * np.sin(2 * np.pi * freq1 * t + phase1) +
                amp2 * np.sin(2 * np.pi * freq2 * t + phase2)
            )
            
            # Add decay envelope
            decay = np.exp(-3 * t / self.window_duration)
            signal[:, axis] *= decay
            
            # Add axis-specific characteristics
            if axis == 0:  # X-axis - more sensitive to certain key positions
                signal[:, axis] *= (1.0 + 0.2 * np.sin(2 * np.pi * 5 * t))
            elif axis == 1:  # Y-axis - different resonance pattern
                signal[:, axis] *= (1.0 + 0.15 * np.cos(2 * np.pi * 7 * t))
            else:  # Z-axis - vertical component
                signal[:, axis] *= (1.0 + 0.1 * np.sin(2 * np.pi * 3 * t))
        
        # Add noise
        noise = np.random.normal(0, noise_level, signal.shape)
        signal += noise
        
        return signal
    
    def generate_dataset(self, keystrokes: List[str], samples_per_key: int = 100, 
                        noise_range: Tuple[float, float] = (0.05, 0.2)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a complete dataset of gyroscope signals and labels.
        
        Args:
            keystrokes: List of keystrokes to generate data for
            samples_per_key: Number of samples to generate per keystroke
            noise_range: Range of noise levels to use for variation
            
        Returns:
            Tuple of (features, labels) where features has shape (n_samples, window_samples, 3)
        """
        n_samples = len(keystrokes) * samples_per_key
        features = np.zeros((n_samples, self.window_samples, 3))
        labels = []
        
        sample_idx = 0
        for keystroke in keystrokes:
            for _ in range(samples_per_key):
                noise_level = np.random.uniform(*noise_range)
                features[sample_idx] = self.generate_keystroke_signal(keystroke, noise_level)
                labels.append(keystroke)
                sample_idx += 1
        
        return features, np.array(labels)
    
    def generate_mixed_sequence(self, sequence: str, noise_level: float = 0.1) -> np.ndarray:
        """
        Generate gyroscope data for a sequence of keystrokes.
        
        Args:
            sequence: String of keystrokes
            noise_level: Amount of noise to add
            
        Returns:
            Array of shape (len(sequence) * window_samples, 3)
        """
        total_samples = len(sequence) * self.window_samples
        signal = np.zeros((total_samples, 3))
        
        for i, keystroke in enumerate(sequence):
            start_idx = i * self.window_samples
            end_idx = start_idx + self.window_samples
            signal[start_idx:end_idx] = self.generate_keystroke_signal(keystroke, noise_level)
        
        return signal


def create_default_dataset(output_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create a default dataset with common keystrokes.
    
    Args:
        output_dir: Optional directory to save the dataset
        
    Returns:
        Tuple of (features, labels, keystroke_list)
    """
    # Common keystrokes for training
    keystrokes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'space', 'enter']
    
    simulator = GyroscopeDataSimulator()
    features, labels = simulator.generate_dataset(keystrokes, samples_per_key=50)
    
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'features.npy'), features)
        np.save(os.path.join(output_dir, 'labels.npy'), labels)
        
        # Save keystroke mapping
        keystroke_mapping = {i: key for i, key in enumerate(keystrokes)}
        np.save(os.path.join(output_dir, 'keystroke_mapping.npy'), keystroke_mapping)
    
    return features, labels, keystrokes


if __name__ == "__main__":
    # Demo usage
    simulator = GyroscopeDataSimulator()
    
    # Generate single keystroke
    signal = simulator.generate_keystroke_signal('a')
    print(f"Generated signal shape: {signal.shape}")
    
    # Generate dataset
    keystrokes = ['a', 'e', 'i', 'space']
    features, labels = simulator.generate_dataset(keystrokes, samples_per_key=10)
    print(f"Dataset shape: features={features.shape}, labels={labels.shape}")