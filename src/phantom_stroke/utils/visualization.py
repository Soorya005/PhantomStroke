"""
Utilities module for PhantomStroke.
Contains helper functions for visualization, evaluation, and data handling.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import os
import json


def plot_gyroscope_signal(signal: np.ndarray, title: str = "Gyroscope Signal", 
                         sampling_rate: int = 100, save_path: Optional[str] = None):
    """
    Plot gyroscope signal for all three axes.
    
    Args:
        signal: Gyroscope data of shape (n_samples, 3)
        title: Plot title
        sampling_rate: Sampling rate in Hz
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    time = np.arange(signal.shape[0]) / sampling_rate
    axis_names = ['X', 'Y', 'Z']
    
    for i, ax in enumerate(axes):
        ax.plot(time, signal[:, i], label=f'{axis_names[i]}-axis')
        ax.set_ylabel(f'{axis_names[i]} (rad/s)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[0].set_title(title)
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(importance_dict: Dict[str, float], top_n: int = 20,
                          title: str = "Feature Importance", save_path: Optional[str] = None):
    """
    Plot feature importance scores.
    
    Args:
        importance_dict: Dictionary of feature names and importance scores
        top_n: Number of top features to display
        title: Plot title
        save_path: Optional path to save the plot
    """
    # Sort features by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    features, importances = zip(*top_features)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(features)), importances)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, importances)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         title: str = "Confusion Matrix", save_path: Optional[str] = None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(history: Dict[str, List[float]], 
                         title: str = "Training History", save_path: Optional[str] = None):
    """
    Plot training history for neural networks.
    
    Args:
        history: Training history dictionary
        title: Plot title
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    if 'accuracy' in history:
        ax1.plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    if 'loss' in history:
        ax2.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_keystroke_samples(data: np.ndarray, labels: np.ndarray, keystrokes: List[str], 
                          samples_per_key: int = 3, save_path: Optional[str] = None):
    """
    Plot sample gyroscope signals for different keystrokes.
    
    Args:
        data: Gyroscope data of shape (n_samples, window_length, 3)
        labels: Labels for each sample
        keystrokes: List of unique keystrokes
        samples_per_key: Number of samples to plot per keystroke
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(len(keystrokes), samples_per_key, 
                           figsize=(15, 3 * len(keystrokes)))
    
    if len(keystrokes) == 1:
        axes = axes.reshape(1, -1)
    
    for i, keystroke in enumerate(keystrokes):
        # Find samples for this keystroke
        keystroke_indices = np.where(labels == keystroke)[0]
        selected_indices = keystroke_indices[:samples_per_key]
        
        for j, idx in enumerate(selected_indices):
            if j < samples_per_key:
                ax = axes[i, j]
                
                # Plot all three axes
                signal = data[idx]
                time = np.arange(signal.shape[0]) / 100  # Assuming 100Hz
                
                ax.plot(time, signal[:, 0], label='X', alpha=0.7)
                ax.plot(time, signal[:, 1], label='Y', alpha=0.7)
                ax.plot(time, signal[:, 2], label='Z', alpha=0.7)
                
                ax.set_title(f"'{keystroke}' - Sample {j+1}")
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Angular Velocity (rad/s)')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_model_metrics(metrics: Dict[str, Any], model_name: str, output_dir: str):
    """
    Save model training metrics to JSON file.
    
    Args:
        metrics: Dictionary containing model metrics
        model_name: Name of the model
        output_dir: Directory to save metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, dict):
            # Handle nested dictionaries (like classification_report)
            serializable_metrics[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (np.ndarray, np.int64, np.float64)):
                    serializable_metrics[key][sub_key] = float(sub_value) if hasattr(sub_value, 'item') else sub_value.tolist()
                else:
                    serializable_metrics[key][sub_key] = sub_value
        else:
            serializable_metrics[key] = value
    
    filepath = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(filepath, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"Metrics saved to {filepath}")


def evaluate_model_performance(true_labels: np.ndarray, predictions: np.ndarray, 
                             class_names: List[str]) -> Dict[str, Any]:
    """
    Comprehensive evaluation of model performance.
    
    Args:
        true_labels: True labels
        predictions: Model predictions
        class_names: List of class names
        
    Returns:
        Dictionary containing evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, labels=class_names
    )
    
    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': support[i]
        }
    
    # Overall metrics
    overall_precision = np.mean(precision)
    overall_recall = np.mean(recall)
    overall_f1 = np.mean(f1)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=class_names)
    
    return {
        'accuracy': accuracy,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm
    }


def create_results_summary(model_results: Dict[str, Dict[str, Any]], 
                          output_dir: str, filename: str = "results_summary.json"):
    """
    Create a summary of all model results.
    
    Args:
        model_results: Dictionary containing results for each model
        output_dir: Directory to save summary
        filename: Name of the summary file
    """
    summary = {
        'models': {},
        'best_model': None,
        'best_accuracy': 0
    }
    
    for model_name, results in model_results.items():
        test_accuracy = results.get('test_accuracy', 0)
        summary['models'][model_name] = {
            'test_accuracy': test_accuracy,
            'train_accuracy': results.get('train_accuracy', 0),
            'cv_mean': results.get('cv_mean', 0),
            'cv_std': results.get('cv_std', 0)
        }
        
        if test_accuracy > summary['best_accuracy']:
            summary['best_accuracy'] = test_accuracy
            summary['best_model'] = model_name
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results summary saved to {filepath}")
    return summary


def load_and_prepare_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and prepare data from saved files.
    
    Args:
        data_dir: Directory containing saved data files
        
    Returns:
        Tuple of (features, labels, keystroke_mapping)
    """
    features_path = os.path.join(data_dir, 'features.npy')
    labels_path = os.path.join(data_dir, 'labels.npy')
    mapping_path = os.path.join(data_dir, 'keystroke_mapping.npy')
    
    if not all(os.path.exists(path) for path in [features_path, labels_path]):
        raise FileNotFoundError(f"Data files not found in {data_dir}")
    
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    if os.path.exists(mapping_path):
        keystroke_mapping = np.load(mapping_path, allow_pickle=True).item()
        keystrokes = [keystroke_mapping[i] for i in range(len(keystroke_mapping))]
    else:
        keystrokes = list(np.unique(labels))
    
    return features, labels, keystrokes


def setup_output_directories(base_dir: str = "output") -> Dict[str, str]:
    """
    Setup output directories for models, plots, and metrics.
    
    Args:
        base_dir: Base output directory
        
    Returns:
        Dictionary of directory paths
    """
    directories = {
        'base': base_dir,
        'models': os.path.join(base_dir, 'models'),
        'plots': os.path.join(base_dir, 'plots'),
        'metrics': os.path.join(base_dir, 'metrics'),
        'data': os.path.join(base_dir, 'data')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories


if __name__ == "__main__":
    # Demo usage
    from ..data.simulator import GyroscopeDataSimulator
    
    # Generate sample data
    simulator = GyroscopeDataSimulator()
    signal = simulator.generate_keystroke_signal('a')
    
    # Plot sample signal
    plot_gyroscope_signal(signal, "Sample Keystroke 'a'")
    
    print("Utilities module demo completed!")