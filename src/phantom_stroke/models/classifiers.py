"""
ML models module for PhantomStroke.
Implements CNN, SVM, and Random Forest models for keystroke classification.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import os


class SVMKeystrokeClassifier:
    """SVM-based keystroke classifier."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale'):
        """
        Initialize SVM classifier.
        
        Args:
            kernel: SVM kernel type
            C: Regularization parameter
            gamma: Kernel coefficient
        """
        self.svm = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def train(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Train the SVM classifier.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            labels: Labels array
            
        Returns:
            Training metrics dictionary
        """
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
        )
        
        # Train SVM
        self.svm.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.svm.score(X_train, y_train)
        test_score = self.svm.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.svm, features_scaled, labels_encoded, cv=5)
        
        # Predictions for detailed metrics
        y_pred = self.svm.predict(X_test)
        
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(
                y_test, y_pred, target_names=self.label_encoder.classes_, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict keystrokes from features."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        features_scaled = self.scaler.transform(features)
        predictions_encoded = self.svm.predict(features_scaled)
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict keystroke probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        features_scaled = self.scaler.transform(features)
        return self.svm.predict_proba(features_scaled)
    
    def save(self, filepath: str):
        """Save the trained model."""
        model_data = {
            'svm': self.svm,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.svm = model_data['svm']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.is_trained = model_data['is_trained']


class RandomForestKeystrokeClassifier:
    """Random Forest-based keystroke classifier."""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, 
                 min_samples_split: int = 2):
        """
        Initialize Random Forest classifier.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
        """
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def train(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Train the Random Forest classifier.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            labels: Labels array
            
        Returns:
            Training metrics dictionary
        """
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
        )
        
        # Train Random Forest
        self.rf.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.rf.score(X_train, y_train)
        test_score = self.rf.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.rf, features, labels_encoded, cv=5)
        
        # Predictions for detailed metrics
        y_pred = self.rf.predict(X_test)
        
        # Feature importance
        feature_importance = self.rf.feature_importances_
        
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(
                y_test, y_pred, target_names=self.label_encoder.classes_, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': feature_importance
        }
        
        return metrics
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict keystrokes from features."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions_encoded = self.rf.predict(features)
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict keystroke probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.rf.predict_proba(features)
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_dict = {}
        for i, importance in enumerate(self.rf.feature_importances_):
            if i < len(feature_names):
                importance_dict[feature_names[i]] = importance
        
        return importance_dict
    
    def save(self, filepath: str):
        """Save the trained model."""
        model_data = {
            'rf': self.rf,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.rf = model_data['rf']
        self.label_encoder = model_data['label_encoder']
        self.is_trained = model_data['is_trained']


class CNNKeystrokeClassifier:
    """CNN-based keystroke classifier using TensorFlow/Keras."""
    
    def __init__(self, input_shape: Tuple[int, int], n_classes: int):
        """
        Initialize CNN classifier.
        
        Args:
            input_shape: Shape of input data (window_length, n_channels)
            n_classes: Number of keystroke classes
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            self.tf = tf
            self.keras = keras
            self.layers = layers
            
        except ImportError:
            raise ImportError("TensorFlow is required for CNN classifier")
        
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def _build_model(self) -> Any:
        """Build the CNN architecture."""
        model = self.keras.Sequential([
            # First Conv1D block
            self.layers.Conv1D(32, 3, activation='relu', input_shape=self.input_shape),
            self.layers.BatchNormalization(),
            self.layers.MaxPooling1D(2),
            self.layers.Dropout(0.25),
            
            # Second Conv1D block
            self.layers.Conv1D(64, 3, activation='relu'),
            self.layers.BatchNormalization(),
            self.layers.MaxPooling1D(2),
            self.layers.Dropout(0.25),
            
            # Third Conv1D block
            self.layers.Conv1D(128, 3, activation='relu'),
            self.layers.BatchNormalization(),
            self.layers.MaxPooling1D(2),
            self.layers.Dropout(0.25),
            
            # Global pooling and dense layers
            self.layers.GlobalAveragePooling1D(),
            self.layers.Dense(256, activation='relu'),
            self.layers.BatchNormalization(),
            self.layers.Dropout(0.5),
            
            self.layers.Dense(128, activation='relu'),
            self.layers.Dropout(0.3),
            
            # Output layer
            self.layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, raw_data: np.ndarray, labels: np.ndarray, 
              epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the CNN classifier.
        
        Args:
            raw_data: Raw gyroscope data of shape (n_samples, window_length, 3)
            labels: Labels array
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training metrics dictionary
        """
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Update number of classes
        self.n_classes = len(np.unique(labels_encoded))
        
        # Build model
        self.model = self._build_model()
        
        # Callbacks
        callbacks = [
            self.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            self.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5
            )
        ]
        
        # Train model
        history = self.model.fit(
            raw_data, labels_encoded,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Evaluate on test data
        X_train, X_test, y_train, y_test = train_test_split(
            raw_data, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
        )
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Predictions for detailed metrics
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        metrics = {
            'train_accuracy': max(history.history['accuracy']),
            'val_accuracy': max(history.history['val_accuracy']),
            'test_accuracy': test_accuracy,
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'test_loss': test_loss,
            'classification_report': classification_report(
                y_test, y_pred, target_names=self.label_encoder.classes_, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'training_history': history.history
        }
        
        return metrics
    
    def predict(self, raw_data: np.ndarray) -> np.ndarray:
        """Predict keystrokes from raw data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions_proba = self.model.predict(raw_data)
        predictions_encoded = np.argmax(predictions_proba, axis=1)
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def predict_proba(self, raw_data: np.ndarray) -> np.ndarray:
        """Predict keystroke probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(raw_data)
    
    def save(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save model architecture and weights
        self.model.save(f"{filepath}_model.h5")
        
        # Save label encoder
        model_data = {
            'label_encoder': self.label_encoder,
            'input_shape': self.input_shape,
            'n_classes': self.n_classes,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, f"{filepath}_metadata.pkl")
    
    def load(self, filepath: str):
        """Load a trained model."""
        # Load model
        self.model = self.keras.models.load_model(f"{filepath}_model.h5")
        
        # Load metadata
        model_data = joblib.load(f"{filepath}_metadata.pkl")
        self.label_encoder = model_data['label_encoder']
        self.input_shape = model_data['input_shape']
        self.n_classes = model_data['n_classes']
        self.is_trained = model_data['is_trained']


class ModelEnsemble:
    """Ensemble of multiple keystroke classifiers."""
    
    def __init__(self):
        """Initialize the ensemble."""
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models[name] = model
        self.weights[name] = weight
        
    def predict(self, features: np.ndarray = None, raw_data: np.ndarray = None) -> np.ndarray:
        """Predict using ensemble voting."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        predictions = {}
        
        for name, model in self.models.items():
            if isinstance(model, CNNKeystrokeClassifier):
                if raw_data is None:
                    raise ValueError("Raw data required for CNN predictions")
                predictions[name] = model.predict(raw_data)
            else:
                if features is None:
                    raise ValueError("Features required for non-CNN predictions")
                predictions[name] = model.predict(features)
        
        # Weighted voting
        final_predictions = []
        n_samples = len(list(predictions.values())[0])
        
        for i in range(n_samples):
            votes = {}
            for name, preds in predictions.items():
                pred = preds[i]
                weight = self.weights[name]
                votes[pred] = votes.get(pred, 0) + weight
            
            # Get prediction with highest weighted vote
            final_pred = max(votes.keys(), key=lambda k: votes[k])
            final_predictions.append(final_pred)
        
        return np.array(final_predictions)
    
    def train_ensemble(self, features: np.ndarray, raw_data: np.ndarray, labels: np.ndarray):
        """Train all models in the ensemble."""
        for name, model in self.models.items():
            print(f"Training {name}...")
            if isinstance(model, CNNKeystrokeClassifier):
                model.train(raw_data, labels)
            else:
                model.train(features, labels)
        
        self.is_trained = True


if __name__ == "__main__":
    # Demo usage
    from ..data.simulator import create_default_dataset
    from ..features.extractor import create_feature_dataset
    
    print("Creating dataset...")
    raw_data, labels, keystrokes = create_default_dataset()
    
    print("Extracting features...")
    features, _, feature_names = create_feature_dataset(raw_data, labels)
    
    print(f"Dataset: {raw_data.shape[0]} samples, {len(keystrokes)} classes")
    print(f"Features: {features.shape[1]} features")
    
    # Test Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestKeystrokeClassifier(n_estimators=50)
    rf_metrics = rf_model.train(features, labels)
    print(f"RF Test Accuracy: {rf_metrics['test_accuracy']:.3f}")
    
    # Test SVM
    print("\nTraining SVM...")
    svm_model = SVMKeystrokeClassifier()
    svm_metrics = svm_model.train(features, labels)
    print(f"SVM Test Accuracy: {svm_metrics['test_accuracy']:.3f}")