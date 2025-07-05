"""
Main training script for PhantomStroke ML models.
Trains CNN, SVM, and Random Forest models on simulated gyroscope data.
"""

import os
import sys
import argparse
import time
import numpy as np
from typing import Dict, List, Tuple, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from phantom_stroke.data.simulator import GyroscopeDataSimulator, create_default_dataset
from phantom_stroke.features.extractor import GyroscopeFeatureExtractor, create_feature_dataset
from phantom_stroke.models.classifiers import (
    SVMKeystrokeClassifier, 
    RandomForestKeystrokeClassifier,
    CNNKeystrokeClassifier,
    ModelEnsemble
)
from phantom_stroke.utils.visualization import (
    plot_gyroscope_signal,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_training_history,
    plot_keystroke_samples,
    save_model_metrics,
    create_results_summary,
    setup_output_directories
)


def train_all_models(raw_data: np.ndarray, features: np.ndarray, labels: np.ndarray, 
                    feature_names: List[str], output_dirs: Dict[str, str]) -> Dict[str, Any]:
    """
    Train all ML models and return their results.
    
    Args:
        raw_data: Raw gyroscope data
        features: Extracted features
        labels: Target labels
        feature_names: Names of the features
        output_dirs: Output directory paths
        
    Returns:
        Dictionary containing results for all models
    """
    results = {}
    
    # Get unique keystrokes
    unique_keystrokes = list(np.unique(labels))
    n_classes = len(unique_keystrokes)
    
    print(f"Training models on {len(labels)} samples with {n_classes} classes")
    print(f"Classes: {unique_keystrokes}")
    
    # 1. Train Random Forest
    print("\n" + "="*50)
    print("Training Random Forest Classifier")
    print("="*50)
    
    start_time = time.time()
    rf_model = RandomForestKeystrokeClassifier(n_estimators=100, max_depth=20)
    rf_metrics = rf_model.train(features, labels)
    rf_training_time = time.time() - start_time
    
    print(f"Training completed in {rf_training_time:.2f} seconds")
    print(f"Test Accuracy: {rf_metrics['test_accuracy']:.3f}")
    print(f"CV Accuracy: {rf_metrics['cv_mean']:.3f} ± {rf_metrics['cv_std']:.3f}")
    
    # Save Random Forest model and metrics
    rf_model.save(os.path.join(output_dirs['models'], 'random_forest_model.pkl'))
    rf_metrics['training_time'] = rf_training_time
    save_model_metrics(rf_metrics, 'random_forest', output_dirs['metrics'])
    
    # Plot Random Forest feature importance
    importance_dict = rf_model.get_feature_importance(feature_names)
    plot_feature_importance(
        importance_dict, 
        top_n=20, 
        title="Random Forest Feature Importance",
        save_path=os.path.join(output_dirs['plots'], 'rf_feature_importance.png')
    )
    
    # Plot Random Forest confusion matrix
    plot_confusion_matrix(
        rf_metrics['confusion_matrix'],
        unique_keystrokes,
        title="Random Forest Confusion Matrix",
        save_path=os.path.join(output_dirs['plots'], 'rf_confusion_matrix.png')
    )
    
    results['random_forest'] = rf_metrics
    
    # 2. Train SVM
    print("\n" + "="*50)
    print("Training SVM Classifier")
    print("="*50)
    
    start_time = time.time()
    svm_model = SVMKeystrokeClassifier(kernel='rbf', C=10.0)
    svm_metrics = svm_model.train(features, labels)
    svm_training_time = time.time() - start_time
    
    print(f"Training completed in {svm_training_time:.2f} seconds")
    print(f"Test Accuracy: {svm_metrics['test_accuracy']:.3f}")
    print(f"CV Accuracy: {svm_metrics['cv_mean']:.3f} ± {svm_metrics['cv_std']:.3f}")
    
    # Save SVM model and metrics
    svm_model.save(os.path.join(output_dirs['models'], 'svm_model.pkl'))
    svm_metrics['training_time'] = svm_training_time
    save_model_metrics(svm_metrics, 'svm', output_dirs['metrics'])
    
    # Plot SVM confusion matrix
    plot_confusion_matrix(
        svm_metrics['confusion_matrix'],
        unique_keystrokes,
        title="SVM Confusion Matrix",
        save_path=os.path.join(output_dirs['plots'], 'svm_confusion_matrix.png')
    )
    
    results['svm'] = svm_metrics
    
    # 3. Train CNN
    print("\n" + "="*50)
    print("Training CNN Classifier")
    print("="*50)
    
    try:
        start_time = time.time()
        input_shape = (raw_data.shape[1], raw_data.shape[2])  # (window_length, n_channels)
        cnn_model = CNNKeystrokeClassifier(input_shape=input_shape, n_classes=n_classes)
        cnn_metrics = cnn_model.train(raw_data, labels, epochs=30, batch_size=32)
        cnn_training_time = time.time() - start_time
        
        print(f"Training completed in {cnn_training_time:.2f} seconds")
        print(f"Test Accuracy: {cnn_metrics['test_accuracy']:.3f}")
        print(f"Validation Accuracy: {cnn_metrics['val_accuracy']:.3f}")
        
        # Save CNN model and metrics
        cnn_model.save(os.path.join(output_dirs['models'], 'cnn_model'))
        cnn_metrics['training_time'] = cnn_training_time
        save_model_metrics(cnn_metrics, 'cnn', output_dirs['metrics'])
        
        # Plot CNN training history
        plot_training_history(
            cnn_metrics['training_history'],
            title="CNN Training History",
            save_path=os.path.join(output_dirs['plots'], 'cnn_training_history.png')
        )
        
        # Plot CNN confusion matrix
        plot_confusion_matrix(
            cnn_metrics['confusion_matrix'],
            unique_keystrokes,
            title="CNN Confusion Matrix",
            save_path=os.path.join(output_dirs['plots'], 'cnn_confusion_matrix.png')
        )
        
        results['cnn'] = cnn_metrics
        
    except ImportError as e:
        print(f"Skipping CNN training: {e}")
        results['cnn'] = {'error': str(e)}
    
    return results


def create_ensemble_model(raw_data: np.ndarray, features: np.ndarray, labels: np.ndarray,
                         output_dirs: Dict[str, str]) -> Dict[str, Any]:
    """
    Create and train an ensemble model.
    
    Args:
        raw_data: Raw gyroscope data
        features: Extracted features
        labels: Target labels
        output_dirs: Output directory paths
        
    Returns:
        Ensemble model results
    """
    print("\n" + "="*50)
    print("Creating Ensemble Model")
    print("="*50)
    
    # Create ensemble
    ensemble = ModelEnsemble()
    
    # Add Random Forest
    rf_model = RandomForestKeystrokeClassifier(n_estimators=100)
    ensemble.add_model('random_forest', rf_model, weight=1.0)
    
    # Add SVM
    svm_model = SVMKeystrokeClassifier(kernel='rbf')
    ensemble.add_model('svm', svm_model, weight=1.0)
    
    # Add CNN if available
    try:
        input_shape = (raw_data.shape[1], raw_data.shape[2])
        n_classes = len(np.unique(labels))
        cnn_model = CNNKeystrokeClassifier(input_shape=input_shape, n_classes=n_classes)
        ensemble.add_model('cnn', cnn_model, weight=1.2)  # Slightly higher weight for CNN
    except ImportError:
        print("CNN not available for ensemble")
    
    # Train ensemble
    start_time = time.time()
    ensemble.train_ensemble(features, raw_data, labels)
    ensemble_training_time = time.time() - start_time
    
    print(f"Ensemble training completed in {ensemble_training_time:.2f} seconds")
    
    return {
        'training_time': ensemble_training_time,
        'models_count': len(ensemble.models)
    }


def generate_visualizations(raw_data: np.ndarray, labels: np.ndarray, 
                           unique_keystrokes: List[str], output_dirs: Dict[str, str]):
    """
    Generate various visualizations of the data.
    
    Args:
        raw_data: Raw gyroscope data
        labels: Target labels
        unique_keystrokes: List of unique keystrokes
        output_dirs: Output directory paths
    """
    print("\n" + "="*50)
    print("Generating Visualizations")
    print("="*50)
    
    # Plot sample keystrokes
    plot_keystroke_samples(
        raw_data, labels, unique_keystrokes[:6],  # First 6 keystrokes
        samples_per_key=2,
        save_path=os.path.join(output_dirs['plots'], 'keystroke_samples.png')
    )
    
    # Plot individual keystroke examples
    for i, keystroke in enumerate(unique_keystrokes[:3]):  # First 3 keystrokes
        keystroke_indices = np.where(labels == keystroke)[0]
        if len(keystroke_indices) > 0:
            sample_signal = raw_data[keystroke_indices[0]]
            plot_gyroscope_signal(
                sample_signal,
                title=f"Keystroke '{keystroke}' Example",
                save_path=os.path.join(output_dirs['plots'], f'keystroke_{keystroke}_example.png')
            )
    
    print("Visualizations saved to plots directory")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PhantomStroke ML models')
    parser.add_argument('--output_dir', type=str, default='output', 
                       help='Output directory for results')
    parser.add_argument('--n_samples', type=int, default=100, 
                       help='Number of samples per keystroke')
    parser.add_argument('--keystrokes', type=str, 
                       default='a,b,c,d,e,f,g,h,i,j,space,enter',
                       help='Comma-separated list of keystrokes to train on')
    parser.add_argument('--no_plots', action='store_true', 
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Setup output directories
    output_dirs = setup_output_directories(args.output_dir)
    
    # Parse keystrokes
    keystrokes = [k.strip() for k in args.keystrokes.split(',')]
    
    print("PhantomStroke ML Training Pipeline")
    print("="*50)
    print(f"Output directory: {args.output_dir}")
    print(f"Keystrokes: {keystrokes}")
    print(f"Samples per keystroke: {args.n_samples}")
    
    # 1. Generate/Load Data
    print("\n" + "="*50)
    print("Generating Gyroscope Data")
    print("="*50)
    
    simulator = GyroscopeDataSimulator(sampling_rate=100, window_duration=0.2)
    raw_data, labels = simulator.generate_dataset(keystrokes, samples_per_key=args.n_samples)
    
    print(f"Generated dataset: {raw_data.shape}")
    print(f"Data shape: {raw_data.shape[0]} samples × {raw_data.shape[1]} time steps × {raw_data.shape[2]} axes")
    
    # Save raw data
    np.save(os.path.join(output_dirs['data'], 'raw_data.npy'), raw_data)
    np.save(os.path.join(output_dirs['data'], 'labels.npy'), labels)
    
    # 2. Extract Features
    print("\n" + "="*50)
    print("Extracting Features")
    print("="*50)
    
    extractor = GyroscopeFeatureExtractor(sampling_rate=100)
    features, feature_names = extractor.extract_features_batch(raw_data)
    
    print(f"Extracted features: {features.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Save features
    np.save(os.path.join(output_dirs['data'], 'features.npy'), features)
    np.save(os.path.join(output_dirs['data'], 'feature_names.npy'), feature_names)
    
    # 3. Generate visualizations
    if not args.no_plots:
        generate_visualizations(raw_data, labels, keystrokes, output_dirs)
    
    # 4. Train all models
    results = train_all_models(raw_data, features, labels, feature_names, output_dirs)
    
    # 5. Create ensemble (optional)
    try:
        ensemble_results = create_ensemble_model(raw_data, features, labels, output_dirs)
        results['ensemble'] = ensemble_results
    except Exception as e:
        print(f"Ensemble creation failed: {e}")
    
    # 6. Create results summary
    summary = create_results_summary(results, output_dirs['metrics'])
    
    # 7. Print final results
    print("\n" + "="*60)
    print("TRAINING COMPLETED - FINAL RESULTS")
    print("="*60)
    
    for model_name, model_results in results.items():
        if 'test_accuracy' in model_results:
            print(f"{model_name.upper()}: {model_results['test_accuracy']:.3f} accuracy")
        elif 'error' in model_results:
            print(f"{model_name.upper()}: Failed - {model_results['error']}")
    
    if summary['best_model']:
        print(f"\nBest performing model: {summary['best_model'].upper()}")
        print(f"Best accuracy: {summary['best_accuracy']:.3f}")
    
    print(f"\nAll results saved to: {args.output_dir}")
    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()