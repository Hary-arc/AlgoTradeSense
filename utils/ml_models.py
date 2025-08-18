import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os
from datetime import datetime

class BaseModel:
    """Base class for all ML models"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.model_type = "base"
        self.training_history = {}
    
    def prepare_data(self, X, y, test_size=0.2, sequence_length=None):
        """Prepare data for training"""
        if sequence_length is not None:
            # For sequence models like LSTM
            return self._prepare_sequence_data(X, y, sequence_length, test_size)
        else:
            # For traditional ML models
            from sklearn.model_selection import train_test_split
            return train_test_split(X, y, test_size=test_size, shuffle=False)
    
    def _prepare_sequence_data(self, X, y, sequence_length, test_size):
        """Prepare sequence data for LSTM models"""
        sequences_X = []
        sequences_y = []
        
        for i in range(sequence_length, len(X)):
            sequences_X.append(X[i-sequence_length:i])
            sequences_y.append(y[i])
        
        sequences_X = np.array(sequences_X)
        sequences_y = np.array(sequences_y)
        
        # Split data
        split_idx = int(len(sequences_X) * (1 - test_size))
        
        X_train = sequences_X[:split_idx]
        X_test = sequences_X[split_idx:]
        y_train = sequences_y[:split_idx]
        y_test = sequences_y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def save_model(self, filepath):
        """Save the trained model"""
        raise NotImplementedError("Subclasses must implement save_model")
    
    def load_model(self, filepath):
        """Load a trained model"""
        raise NotImplementedError("Subclasses must implement load_model")
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        return None

class LSTMModel(BaseModel):
    """LSTM model for time series prediction"""
    
    def __init__(self, input_shape, units=128, dropout_rate=0.2, learning_rate=0.001):
        super().__init__()
        self.input_shape = input_shape
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_type = "LSTM"
        
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
    
    def train(self, X_train, y_train, validation_data=None, epochs=50, batch_size=32, verbose=1):
        """Train the LSTM model"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        self.training_history = history.history
        return history
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
    
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
            'training_history': self.training_history
        }
        
        joblib.dump(metadata, f"{filepath}_metadata.pkl")
    
    def load_model(self, filepath):
        """Load LSTM model"""
        # Load Keras model
        self.model = tf.keras.models.load_model(f"{filepath}.h5")
        
        # Load metadata
        metadata = joblib.load(f"{filepath}_metadata.pkl")
        
        self.input_shape = metadata['input_shape']
        self.units = metadata['units']
        self.dropout_rate = metadata['dropout_rate']
        self.learning_rate = metadata['learning_rate']
        self.training_history = metadata.get('training_history', {})
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
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
    
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
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
    
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
    if model_type.upper() == 'LSTM':
        if 'input_shape' not in kwargs:
            raise ValueError("input_shape is required for LSTM model")
        return LSTMModel(**kwargs)
    elif model_type.upper() == 'RANDOM FOREST':
        return RandomForestModel(**kwargs)
    elif model_type.upper() == 'SVM':
        return SVMModel(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def evaluate_models(models, X_test, y_test):
    """Compare multiple models on test data"""
    results = {}
    
    for name, model in models.items():
        if model.is_trained:
            results[name] = model.evaluate(X_test, y_test)
        else:
            results[name] = {'error': 'Model not trained'}
    
    return results
