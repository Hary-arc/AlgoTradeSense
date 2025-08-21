import numpy as np
import pandas as pd
import joblib
import os
import sys
from datetime import datetime

# Check if scikit-learn is available
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except ImportError:
    raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow.python.keras.models import Sequential, load_model
    from tensorflow.python.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.python.keras.optimizers import Adam
    from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow is not available. LSTM models will not work.")
    print("Install with: pip install tensorflow")


class BaseModel:
    """Base class for all ML models"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.model_type = "base"
        self.training_history = {}
        self.required_sequence_length = 1  # Default for non-sequence models
    
    def prepare_data(self, X, y, test_size=0.2, shuffle=False, random_state=None):
        """Prepare data for training - simplified version"""
        return train_test_split(
            X, y, 
            test_size=test_size, 
            shuffle=shuffle, 
            random_state=random_state
        )
    
    def create_sequences(self, data, sequence_length):
        """Create sequences from data for time series models"""
        sequences = []
        for i in range(sequence_length, len(data)):
            sequences.append(data[i-sequence_length:i])
        return np.array(sequences)
    
    def save_model(self, filepath):
        """Save the trained model"""
        raise NotImplementedError("Subclasses must implement save_model")
    
    def load_model(self, filepath):
        """Load a trained model"""
        raise NotImplementedError("Subclasses must implement load_model")
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        return None
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance - common implementation"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            predictions = self.predict(X_test)
            
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse),
                'predictions': predictions
            }
        except Exception as e:
            return {'error': f"Evaluation failed: {str(e)}"}


class LSTMModel(BaseModel):
    """LSTM model for time series prediction"""
    
    def __init__(self, input_shape, units=128, dropout_rate=0.2, learning_rate=0.001):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models. Install with: pip install tensorflow")
            
        super().__init__()
        self.input_shape = input_shape
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_type = "LSTM"
        self.required_sequence_length = input_shape[0]  # For sequence models
        
        self._build_model()
    
    def _build_model(self):
        """Build LSTM model architecture"""
        self.model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=self.input_shape),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            
            LSTM(self.units // 2, return_sequences=True),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            
            LSTM(self.units // 4, return_sequences=False),
            Dropout(self.dropout_rate),
            
            Dense(50, activation='relu'),
            Dropout(self.dropout_rate / 2),
            
            Dense(1, activation='linear')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def prepare_data(self, X, y, test_size=0.2, shuffle=False, random_state=None):
        """Prepare sequence data for LSTM"""
        # Create sequences
        X_seq = self.create_sequences(X, self.required_sequence_length)
        y_seq = y[self.required_sequence_length:]
        
        # Split data
        split_idx = int(len(X_seq) * (1 - test_size))
        
        X_train = X_seq[:split_idx]
        X_test = X_seq[split_idx:]
        y_train = y_seq[:split_idx]
        y_test = y_seq[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, validation_data=None, epochs=50, batch_size=32, verbose=1):
        """Train the LSTM model"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=verbose
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False  # Important for time series data
        )
        
        self.is_trained = True
        self.training_history = history.history
        return history
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X, verbose=0).flatten()
    
    def save_model(self, filepath):
        """Save LSTM model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save Keras model
        self.model.save(f"{filepath}.h5")
        
        # Save model metadata
        metadata = {
            'model_type': self.model_type,
            'input_shape': self.input_shape,
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'training_history': self.training_history,
            'required_sequence_length': self.required_sequence_length
        }
        
        joblib.dump(metadata, f"{filepath}_metadata.pkl")
    
    def load_model(self, filepath):
        """Load LSTM model"""
        # Load Keras model
        self.model = load_model(f"{filepath}.h5")
        
        # Load metadata
        metadata = joblib.load(f"{filepath}_metadata.pkl")
        
        self.input_shape = metadata['input_shape']
        self.units = metadata['units']
        self.dropout_rate = metadata['dropout_rate']
        self.learning_rate = metadata['learning_rate']
        self.training_history = metadata.get('training_history', {})
        self.required_sequence_length = metadata.get('required_sequence_length', self.input_shape[0])
        self.is_trained = True


class RandomForestModel(BaseModel):
    """Random Forest model for price prediction"""
    
    def __init__(self, n_estimators=100, max_depth=15, min_samples_split=5, random_state=42):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.model_type = "Random Forest"
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def train(self, X_train, y_train):
        """Train the Random Forest model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Store training info
        self.training_history = {
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'training_score': self.model.score(X_train, y_train)
        }
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if not self.is_trained:
            return None
        
        return self.model.feature_importances_
    
    def save_model(self, filepath):
        """Save Random Forest model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'random_state': self.random_state,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, f"{filepath}.pkl")
    
    def load_model(self, filepath):
        """Load Random Forest model"""
        model_data = joblib.load(f"{filepath}.pkl")
        
        self.model = model_data['model']
        self.n_estimators = model_data['n_estimators']
        self.max_depth = model_data['max_depth']
        self.min_samples_split = model_data['min_samples_split']
        self.random_state = model_data['random_state']
        self.training_history = model_data.get('training_history', {})
        self.is_trained = True


class SVMModel(BaseModel):
    """Support Vector Machine model for price prediction"""
    
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', epsilon=0.1):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.epsilon = epsilon
        self.model_type = "SVM"
        
        self.model = SVR(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            epsilon=self.epsilon
        )
        
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train):
        """Train the SVM model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Store training info
        self.training_history = {
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'n_support_vectors': len(self.model.support_),
            'training_score': self.model.score(X_train_scaled, y_train)
        }
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, filepath):
        """Save SVM model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, f"{filepath}.pkl")
    
    def load_model(self, filepath):
        """Load SVM model"""
        model_data = joblib.load(f"{filepath}.pkl")
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.C = model_data['C']
        self.kernel = model_data['kernel']
        self.gamma = model_data['gamma']
        self.epsilon = model_data['epsilon']
        self.training_history = model_data.get('training_history', {})
        self.is_trained = True


def create_model(model_type, **kwargs):
    """Factory function to create ML models"""
    model_type = model_type.upper()
    
    if model_type == 'LSTM':
        if 'input_shape' not in kwargs:
            raise ValueError("input_shape is required for LSTM model")
        return LSTMModel(**kwargs)
    elif model_type == 'RANDOMFOREST' or model_type == 'RANDOM_FOREST':
        return RandomForestModel(**kwargs)
    elif model_type == 'SVM':
        return SVMModel(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def evaluate_models(models, X_test, y_test):
    """Compare multiple models on test data"""
    results = {}
    
    for name, model in models.items():
        try:
            if model.is_trained:
                results[name] = model.evaluate(X_test, y_test)
            else:
                results[name] = {'error': 'Model not trained'}
        except Exception as e:
            results[name] = {'error': f"Evaluation error: {str(e)}"}
    
    return results


# Example usage
if __name__ == "__main__":
    # Create sample data
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)
    
    # Test Random Forest
    rf_model = RandomForestModel()
    X_train, X_test, y_train, y_test = rf_model.prepare_data(X, y, test_size=0.2)
    rf_model.train(X_train, y_train)
    rf_results = rf_model.evaluate(X_test, y_test)
    print("Random Forest Results:", rf_results)
    
    # Test SVM
    svm_model = SVMModel()
    X_train, X_test, y_train, y_test = svm_model.prepare_data(X, y, test_size=0.2)
    svm_model.train(X_train, y_train)
    svm_results = svm_model.evaluate(X_test, y_test)
    print("SVM Results:", svm_results)
    
    # Test LSTM if TensorFlow is available
    if TENSORFLOW_AVAILABLE:
        # Reshape data for LSTM (samples, timesteps, features)
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
        lstm_model = LSTMModel(input_shape=(1, X.shape[1]), units=64)
        X_train, X_test, y_train, y_test = lstm_model.prepare_data(X_reshaped, y, test_size=0.2)
        lstm_model.train(X_train, y_train, epochs=10, verbose=0)
        lstm_results = lstm_model.evaluate(X_test, y_test)
        print("LSTM Results:", lstm_results)