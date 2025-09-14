import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Dict, Tuple, Any
import hashlib
import re
from datetime import datetime

class MLAnomalyDetector:
    def __init__(self, models_folder: str = 'models'):
        self.models_folder = models_folder
        self.contamination = 0.1  # 10% contamination threshold
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        
        # Create models directory if it doesn't exist
        os.makedirs(models_folder, exist_ok=True)
        
        # Try to load existing model
        self.load_model()
    
    def extract_features(self, bids_data: List[Dict]) -> pd.DataFrame:
        """Extract features from bid data for ML processing"""
        if not bids_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(bids_data)
        
        # Basic numerical features
        features = {
            'bid_amount': df['bid_amount'].astype(float),
            'proposal_length': df['proposal'].str.len(),
            'company_name_length': df['company_name'].str.len(),
        }
        
        # Extract time-based features
        if 'submitted_at' in df.columns:
            df['submitted_at'] = pd.to_datetime(df['submitted_at'])
            features['submission_hour'] = df['submitted_at'].dt.hour
            features['submission_day_of_week'] = df['submitted_at'].dt.dayofweek
            features['submission_month'] = df['submitted_at'].dt.month
        else:
            # Default time features if not available
            features['submission_hour'] = 12
            features['submission_day_of_week'] = 0
            features['submission_month'] = 1
        
        # Text-based features
        features['proposal_word_count'] = df['proposal'].str.split().str.len()
        features['proposal_sentence_count'] = df['proposal'].str.count(r'[.!?]') + 1
        features['proposal_avg_word_length'] = df['proposal'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )
        
        # Company name patterns
        features['company_name_has_numbers'] = df['company_name'].str.contains(r'\d', na=False).astype(int)
        features['company_name_has_special_chars'] = df['company_name'].str.contains(r'[^a-zA-Z0-9\s]', na=False).astype(int)
        features['company_name_word_count'] = df['company_name'].str.split().str.len()
        
        # Email domain features
        if 'contact_email' in df.columns:
            features['email_domain_length'] = df['contact_email'].str.split('@').str[1].str.len()
            features['email_is_generic'] = df['contact_email'].str.contains(
                r'@(gmail|yahoo|hotmail|outlook|aol)\.', case=False, na=False
            ).astype(int)
        else:
            features['email_domain_length'] = 10
            features['email_is_generic'] = 0
        
        # Bid amount relative features (if tender info available)
        if 'estimated_value' in df.columns:
            features['bid_to_estimate_ratio'] = df['bid_amount'] / df['estimated_value'].replace(0, 1)
        else:
            features['bid_to_estimate_ratio'] = 1.0
        
        # Advanced text analysis
        features['proposal_caps_ratio'] = df['proposal'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if x else 0
        )
        features['proposal_punctuation_ratio'] = df['proposal'].apply(
            lambda x: sum(1 for c in x if c in '.,!?;:') / len(x) if x else 0
        )
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(features)
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        # Store feature columns for consistency
        self.feature_columns = feature_df.columns.tolist()
        
        return feature_df
    
    def prepare_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model (scaling, encoding)"""
        if features_df.empty:
            return np.array([])
        
        # Ensure we have the same features as training
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0
            features_df = features_df[self.feature_columns]
        
        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(features_df)
        else:
            scaled_features = self.scaler.transform(features_df)
        
        return scaled_features
    
    def train_model(self, bids_data: List[Dict]) -> Dict[str, Any]:
        """Train the Isolation Forest model on bid data"""
        if len(bids_data) < 10:
            return {
                'success': False,
                'message': 'Insufficient data for training. Need at least 10 bids.',
                'model_path': None
            }
        
        try:
            # Extract features
            features_df = self.extract_features(bids_data)
            if features_df.empty:
                return {
                    'success': False,
                    'message': 'No features could be extracted from the data.',
                    'model_path': None
                }
            
            # Prepare features
            X = self.prepare_features(features_df)
            if X.size == 0:
                return {
                    'success': False,
                    'message': 'No valid features after preprocessing.',
                    'model_path': None
                }
            
            # Train Isolation Forest
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.model.fit(X)
            
            # Save model
            model_path = self.save_model()
            
            # Get predictions for training data
            predictions = self.model.predict(X)
            anomaly_scores = self.model.decision_function(X)
            
            # Calculate metrics
            n_outliers = sum(1 for p in predictions if p == -1)
            outlier_percentage = (n_outliers / len(predictions)) * 100
            
            return {
                'success': True,
                'message': f'Model trained successfully on {len(bids_data)} bids.',
                'model_path': model_path,
                'metrics': {
                    'total_samples': len(bids_data),
                    'outliers_detected': n_outliers,
                    'outlier_percentage': round(outlier_percentage, 2),
                    'contamination_threshold': self.contamination,
                    'feature_count': X.shape[1]
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error training model: {str(e)}',
                'model_path': None
            }
    
    def predict_anomalies(self, bids_data: List[Dict]) -> List[Dict]:
        """Predict anomalies in bid data"""
        if self.model is None:
            raise Exception("Model not trained. Please train the model first.")
        
        if not bids_data:
            return []
        
        try:
            # Extract features
            features_df = self.extract_features(bids_data)
            X = self.prepare_features(features_df)
            
            # Get predictions and scores
            predictions = self.model.predict(X)
            anomaly_scores = self.model.decision_function(X)
            
            # Process results
            results = []
            for i, bid in enumerate(bids_data):
                is_anomaly = predictions[i] == -1
                score = float(anomaly_scores[i])
                
                # Normalize score to 0-1 range (higher = more suspicious)
                normalized_score = max(0, min(1, (0.5 - score) / 1.0))
                
                results.append({
                    'bid_id': bid['id'],
                    'is_suspicious': bool(is_anomaly),
                    'anomaly_score': round(normalized_score, 3),
                    'company_name': bid['company_name'],
                    'bid_amount': bid['bid_amount'],
                    'reasons': self._explain_anomaly(bid, features_df.iloc[i] if not features_df.empty else {})
                })
            
            return results
            
        except Exception as e:
            raise Exception(f"Error predicting anomalies: {str(e)}")
    
    def _explain_anomaly(self, bid: Dict, features: pd.Series) -> List[str]:
        """Provide explanations for why a bid might be suspicious"""
        reasons = []
        
        try:
            # Check bid amount anomalies
            if 'bid_to_estimate_ratio' in features and features['bid_to_estimate_ratio'] < 0.5:
                reasons.append("Bid amount significantly lower than estimated value")
            elif 'bid_to_estimate_ratio' in features and features['bid_to_estimate_ratio'] > 2.0:
                reasons.append("Bid amount significantly higher than estimated value")
            
            # Check proposal quality
            if 'proposal_length' in features and features['proposal_length'] < 50:
                reasons.append("Unusually short proposal")
            elif 'proposal_length' in features and features['proposal_length'] > 5000:
                reasons.append("Unusually long proposal")
            
            # Check submission timing
            if 'submission_hour' in features:
                hour = features['submission_hour']
                if hour < 6 or hour > 22:
                    reasons.append("Submitted outside normal business hours")
            
            # Check company name patterns
            if 'company_name_has_numbers' in features and features['company_name_has_numbers']:
                reasons.append("Company name contains numbers")
            
            if 'email_is_generic' in features and features['email_is_generic']:
                reasons.append("Using generic email provider")
            
            # Check text quality
            if 'proposal_caps_ratio' in features and features['proposal_caps_ratio'] > 0.3:
                reasons.append("Excessive use of capital letters in proposal")
            
            if not reasons:
                reasons.append("Multiple statistical anomalies detected")
        
        except Exception:
            reasons.append("Anomaly detection based on statistical analysis")
        
        return reasons
    
    def save_model(self) -> str:
        """Save the trained model to disk"""
        if self.model is None:
            raise Exception("No model to save")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'contamination': self.contamination,
            'timestamp': timestamp
        }
        
        model_path = os.path.join(self.models_folder, f'isolation_forest_{timestamp}.joblib')
        joblib.dump(model_data, model_path)
        
        # Also save as latest
        latest_path = os.path.join(self.models_folder, 'latest_model.joblib')
        joblib.dump(model_data, latest_path)
        
        return model_path
    
    def load_model(self) -> bool:
        """Load the latest trained model"""
        latest_path = os.path.join(self.models_folder, 'latest_model.joblib')
        
        if os.path.exists(latest_path):
            try:
                model_data = joblib.load(latest_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_columns = model_data['feature_columns']
                self.contamination = model_data.get('contamination', 0.1)
                return True
            except Exception:
                return False
        
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if self.model is None:
            return {
                'trained': False,
                'message': 'No model has been trained yet.'
            }
        
        return {
            'trained': True,
            'contamination': self.contamination,
            'feature_count': len(self.feature_columns),
            'features': self.feature_columns,
            'model_type': 'Isolation Forest'
        }
    
    def test_model(self, test_data: List[Dict], known_suspicious_ids: List[int] = None) -> Dict[str, Any]:
        """Test model performance on known data"""
        if self.model is None:
            return {
                'success': False,
                'message': 'No model available for testing.'
            }
        
        if not test_data:
            return {
                'success': False,
                'message': 'No test data provided.'
            }
        
        try:
            predictions = self.predict_anomalies(test_data)
            
            results = {
                'success': True,
                'total_predictions': len(predictions),
                'suspicious_detected': sum(1 for p in predictions if p['is_suspicious']),
                'average_anomaly_score': np.mean([p['anomaly_score'] for p in predictions])
            }
            
            # If we have known suspicious IDs, calculate accuracy metrics
            if known_suspicious_ids:
                y_true = [1 if bid['id'] in known_suspicious_ids else 0 for bid in test_data]
                y_pred = [1 if p['is_suspicious'] else 0 for p in predictions]
                
                if len(set(y_true)) > 1:  # Only if we have both classes
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    results.update({
                        'accuracy': round(accuracy_score(y_true, y_pred), 3),
                        'precision': round(precision_score(y_true, y_pred, zero_division=0), 3),
                        'recall': round(recall_score(y_true, y_pred, zero_division=0), 3),
                        'f1_score': round(f1_score(y_true, y_pred, zero_division=0), 3)
                    })
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error testing model: {str(e)}'
            }