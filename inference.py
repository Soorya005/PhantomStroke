"""
Inference script for PhantomStroke.
Uses trained models to predict keystrokes from gyroscope data.
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from phantom_stroke.data.simulator import GyroscopeDataSimulator
from phantom_stroke.features.extractor import GyroscopeFeatureExtractor
from phantom_stroke.models.classifiers import (
    SVMKeystrokeClassifier, 
    RandomForestKeystrokeClassifier,
    CNNKeystrokeClassifier
)


class PhantomStrokeInference:
    """Main inference class for PhantomStroke keystroke prediction."""
    
    def __init__(self, models_dir: str):
        """
        Initialize the inference system.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        self.models = {}
        self.feature_extractor = GyroscopeFeatureExtractor(sampling_rate=100)
        self.load_models()
    
    def load_models(self):
        """Load all available trained models."""
        print("Loading trained models...")
        
        # Load Random Forest
        rf_path = os.path.join(self.models_dir, 'random_forest_model.pkl')
        if os.path.exists(rf_path):
            try:
                rf_model = RandomForestKeystrokeClassifier()
                rf_model.load(rf_path)
                self.models['random_forest'] = rf_model
                print("✓ Random Forest model loaded")
            except Exception as e:
                print(f"✗ Failed to load Random Forest model: {e}")
        
        # Load SVM
        svm_path = os.path.join(self.models_dir, 'svm_model.pkl')
        if os.path.exists(svm_path):
            try:
                svm_model = SVMKeystrokeClassifier()
                svm_model.load(svm_path)
                self.models['svm'] = svm_model
                print("✓ SVM model loaded")
            except Exception as e:
                print(f"✗ Failed to load SVM model: {e}")
        
        # Load CNN
        cnn_model_path = os.path.join(self.models_dir, 'cnn_model_model.h5')
        cnn_meta_path = os.path.join(self.models_dir, 'cnn_model_metadata.pkl')
        if os.path.exists(cnn_model_path) and os.path.exists(cnn_meta_path):
            try:
                # Need to determine input shape - load from metadata or use default
                cnn_model = CNNKeystrokeClassifier(input_shape=(20, 3), n_classes=10)  # Default values
                cnn_model.load(os.path.join(self.models_dir, 'cnn_model'))
                self.models['cnn'] = cnn_model
                print("✓ CNN model loaded")
            except Exception as e:
                print(f"✗ Failed to load CNN model: {e}")
        
        if not self.models:
            raise ValueError("No trained models found in the specified directory")
        
        print(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def predict_single_keystroke(self, gyroscope_data: np.ndarray, 
                                model_name: Optional[str] = None) -> Dict[str, any]:
        """
        Predict keystroke from a single gyroscope data sample.
        
        Args:
            gyroscope_data: Gyroscope data of shape (window_length, 3)
            model_name: Specific model to use, or None for ensemble
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        if model_name and model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available. Available: {list(self.models.keys())}")
        
        results = {}
        
        # Extract features for non-CNN models
        features = None
        if any(name in self.models for name in ['random_forest', 'svm']):
            features_dict = self.feature_extractor.extract_all_features(gyroscope_data)
            features = np.array([list(features_dict.values())]).reshape(1, -1)
        
        # Get predictions from specified model or all models
        models_to_use = [model_name] if model_name else self.models.keys()
        
        for name in models_to_use:
            if name in self.models:
                model = self.models[name]
                
                try:
                    if isinstance(model, CNNKeystrokeClassifier):
                        # CNN uses raw data
                        raw_data = gyroscope_data.reshape(1, gyroscope_data.shape[0], gyroscope_data.shape[1])
                        prediction = model.predict(raw_data)[0]
                        probabilities = model.predict_proba(raw_data)[0]
                    else:
                        # SVM and RF use features
                        prediction = model.predict(features)[0]
                        probabilities = model.predict_proba(features)[0]
                    
                    results[name] = {
                        'prediction': prediction,
                        'probabilities': probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities,
                        'confidence': float(np.max(probabilities)) if hasattr(probabilities, 'max') else 0.0
                    }
                    
                except Exception as e:
                    results[name] = {'error': str(e)}
        
        # Ensemble prediction if multiple models
        if len(results) > 1 and not model_name:
            ensemble_prediction = self._ensemble_predict(results)
            results['ensemble'] = ensemble_prediction
        
        return results
    
    def predict_sequence(self, gyroscope_sequence: np.ndarray, 
                        window_size: int = 20, stride: int = 10,
                        model_name: Optional[str] = None) -> List[Dict[str, any]]:
        """
        Predict keystrokes from a sequence of gyroscope data using sliding window.
        
        Args:
            gyroscope_sequence: Gyroscope data of shape (sequence_length, 3)
            window_size: Size of sliding window
            stride: Stride for sliding window
            model_name: Specific model to use
            
        Returns:
            List of predictions for each window
        """
        predictions = []
        
        for i in range(0, len(gyroscope_sequence) - window_size + 1, stride):
            window_data = gyroscope_sequence[i:i + window_size]
            prediction = self.predict_single_keystroke(window_data, model_name)
            prediction['window_start'] = i
            prediction['window_end'] = i + window_size
            predictions.append(prediction)
        
        return predictions
    
    def _ensemble_predict(self, model_results: Dict[str, Dict]) -> Dict[str, any]:
        """
        Create ensemble prediction from multiple model results.
        
        Args:
            model_results: Results from individual models
            
        Returns:
            Ensemble prediction result
        """
        valid_results = {k: v for k, v in model_results.items() if 'error' not in v}
        
        if not valid_results:
            return {'error': 'No valid model predictions'}
        
        # Simple voting ensemble
        votes = {}
        total_confidence = 0
        
        for model_name, result in valid_results.items():
            prediction = result['prediction']
            confidence = result['confidence']
            
            votes[prediction] = votes.get(prediction, 0) + confidence
            total_confidence += confidence
        
        # Get prediction with highest weighted vote
        ensemble_prediction = max(votes.keys(), key=lambda k: votes[k])
        ensemble_confidence = votes[ensemble_prediction] / total_confidence if total_confidence > 0 else 0
        
        return {
            'prediction': ensemble_prediction,
            'confidence': ensemble_confidence,
            'votes': votes,
            'num_models': len(valid_results)
        }
    
    def simulate_and_predict(self, keystroke: str, noise_level: float = 0.1) -> Dict[str, any]:
        """
        Simulate a keystroke and predict it (for testing).
        
        Args:
            keystroke: Keystroke to simulate
            noise_level: Amount of noise to add
            
        Returns:
            Prediction results
        """
        # Generate simulated data
        simulator = GyroscopeDataSimulator()
        simulated_data = simulator.generate_keystroke_signal(keystroke, noise_level)
        
        # Predict
        results = self.predict_single_keystroke(simulated_data)
        results['true_keystroke'] = keystroke
        results['simulated'] = True
        
        return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='PhantomStroke Keystroke Inference')
    parser.add_argument('--models_dir', type=str, default='output/models',
                       help='Directory containing trained models')
    parser.add_argument('--keystroke', type=str, 
                       help='Simulate and predict a specific keystroke')
    parser.add_argument('--model', type=str, choices=['random_forest', 'svm', 'cnn'],
                       help='Specific model to use for prediction')
    parser.add_argument('--noise', type=float, default=0.1,
                       help='Noise level for simulation (0.0-1.0)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    print("PhantomStroke Keystroke Inference")
    print("="*40)
    
    # Initialize inference system
    try:
        inference = PhantomStrokeInference(args.models_dir)
    except Exception as e:
        print(f"Error initializing inference system: {e}")
        return
    
    if args.interactive:
        # Interactive mode
        print("\nInteractive mode - Type keystrokes to simulate and predict")
        print("Available keystrokes: a, b, c, d, e, f, g, h, i, j, space, enter")
        print("Type 'quit' to exit")
        
        while True:
            keystroke = input("\nEnter keystroke to simulate: ").strip().lower()
            
            if keystroke == 'quit':
                break
            
            if not keystroke:
                continue
            
            try:
                results = inference.simulate_and_predict(keystroke, args.noise)
                
                print(f"\nTrue keystroke: '{keystroke}'")
                print("-" * 30)
                
                for model_name, result in results.items():
                    if model_name in ['true_keystroke', 'simulated']:
                        continue
                    
                    if 'error' in result:
                        print(f"{model_name}: Error - {result['error']}")
                    else:
                        prediction = result['prediction']
                        confidence = result['confidence']
                        correct = "✓" if prediction == keystroke else "✗"
                        print(f"{model_name}: '{prediction}' (confidence: {confidence:.3f}) {correct}")
                
            except Exception as e:
                print(f"Error during prediction: {e}")
    
    elif args.keystroke:
        # Single keystroke prediction
        print(f"\nSimulating keystroke: '{args.keystroke}'")
        
        try:
            results = inference.simulate_and_predict(args.keystroke, args.noise)
            
            print(f"\nTrue keystroke: '{args.keystroke}'")
            print("="*40)
            
            for model_name, result in results.items():
                if model_name in ['true_keystroke', 'simulated']:
                    continue
                
                if 'error' in result:
                    print(f"\n{model_name.upper()}: Error")
                    print(f"  {result['error']}")
                else:
                    prediction = result['prediction']
                    confidence = result['confidence']
                    correct = "✓ CORRECT" if prediction == args.keystroke else "✗ INCORRECT"
                    
                    print(f"\n{model_name.upper()}: '{prediction}' {correct}")
                    print(f"  Confidence: {confidence:.3f}")
                    
                    if 'probabilities' in result and len(result['probabilities']) <= 10:
                        print("  Top probabilities:")
                        # Get class names from model if available
                        try:
                            if hasattr(inference.models[model_name], 'label_encoder'):
                                classes = inference.models[model_name].label_encoder.classes_
                                probs = result['probabilities']
                                sorted_indices = np.argsort(probs)[::-1][:3]  # Top 3
                                for idx in sorted_indices:
                                    print(f"    '{classes[idx]}': {probs[idx]:.3f}")
                        except:
                            pass
        
        except Exception as e:
            print(f"Error during prediction: {e}")
    
    else:
        # Just show loaded models
        print("\nInference system ready!")
        print(f"Loaded models: {list(inference.models.keys())}")
        print("\nUse --keystroke to test a specific keystroke")
        print("Use --interactive for interactive mode")


if __name__ == "__main__":
    main()