"""
Demo script for PhantomStroke ML training and inference.
Quick demonstration of the complete ML pipeline.
"""

import os
import sys
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from phantom_stroke.data.simulator import GyroscopeDataSimulator
from phantom_stroke.features.extractor import GyroscopeFeatureExtractor
from phantom_stroke.models.classifiers import RandomForestKeystrokeClassifier, SVMKeystrokeClassifier


def run_quick_demo():
    """Run a quick demonstration of the PhantomStroke system."""
    print("PhantomStroke ML Demo")
    print("=" * 40)
    
    # 1. Generate sample data
    print("\n1. Generating sample gyroscope data...")
    simulator = GyroscopeDataSimulator(sampling_rate=100, window_duration=0.2)
    keystrokes = ['a', 'e', 'space']
    raw_data, labels = simulator.generate_dataset(keystrokes, samples_per_key=30)
    print(f"   Generated {raw_data.shape[0]} samples for {len(keystrokes)} keystrokes")
    
    # 2. Extract features
    print("\n2. Extracting features...")
    extractor = GyroscopeFeatureExtractor(sampling_rate=100)
    features, feature_names = extractor.extract_features_batch(raw_data)
    print(f"   Extracted {features.shape[1]} features per sample")
    
    # 3. Train Random Forest model
    print("\n3. Training Random Forest model...")
    rf_model = RandomForestKeystrokeClassifier(n_estimators=50)
    rf_metrics = rf_model.train(features, labels)
    print(f"   RF Test Accuracy: {rf_metrics['test_accuracy']:.3f}")
    
    # 4. Train SVM model
    print("\n4. Training SVM model...")
    svm_model = SVMKeystrokeClassifier()
    svm_metrics = svm_model.train(features, labels)
    print(f"   SVM Test Accuracy: {svm_metrics['test_accuracy']:.3f}")
    
    # 5. Test inference
    print("\n5. Testing inference on new data...")
    test_keystrokes = ['a', 'e', 'space']
    
    for keystroke in test_keystrokes:
        # Generate test sample
        test_signal = simulator.generate_keystroke_signal(keystroke, noise_level=0.15)
        test_features = extractor.extract_all_features(test_signal)
        test_features_array = np.array([list(test_features.values())]).reshape(1, -1)
        
        # Predict with both models
        rf_pred = rf_model.predict(test_features_array)[0]
        svm_pred = svm_model.predict(test_features_array)[0]
        
        rf_correct = "✓" if rf_pred == keystroke else "✗"
        svm_correct = "✓" if svm_pred == keystroke else "✗"
        
        print(f"   True: '{keystroke}' | RF: '{rf_pred}' {rf_correct} | SVM: '{svm_pred}' {svm_correct}")
    
    print("\n" + "=" * 40)
    print("Demo completed successfully!")
    print("\nTo run the full training pipeline:")
    print("  python train_models.py")
    print("\nTo use inference:")
    print("  python inference.py --keystroke a")
    print("  python inference.py --interactive")


if __name__ == "__main__":
    run_quick_demo()