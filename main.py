"""
IRIS Classification Pipeline with DVC Integration
MLOps - Data Versioning with DVC
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os
from datetime import datetime
from tqdm import tqdm
import time

def main():
    """Main training pipeline for IRIS classification with comprehensive logging"""
    
    print("=" * 60)
    print("IRIS Classification Pipeline with DVC")
    print("=" * 60)
    
    steps = [
        "Loading IRIS dataset",
        "Data exploration and validation", 
        "Preparing features and target variables",
        "Splitting data (train/test)",
        "Training Decision Tree model",
        "Making predictions on test set",
        "Calculating performance metrics",
        "Saving detailed metrics",
        "Persisting trained model"
    ]
    
    try:
        with tqdm(total=len(steps), desc="Training Progress") as pbar:
            
            # Step 1: Load IRIS dataset
            pbar.set_description("Loading dataset...")
            data_path = 'data/data.csv'
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset not found at {data_path}")
                
            df = pd.read_csv(data_path)
            
            pbar.set_description("Exploring data...")
            print(f"Dataset Info:")
            print(f"   - Shape: {df.shape}")
            print(f"   - Classes: {df['species'].unique()}")
            print(f"   - Class distribution:\n{df['species'].value_counts().to_dict()}")
            
            # Check for missing values
            missing_values = df.isnull().sum().sum()
            print(f"   - Missing values: {missing_values}")
            time.sleep(0.3)
            pbar.update(1)
            
            # Step 3: Prepare features and target
            pbar.set_description("Preparing features...")
            feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            
            X = df[feature_columns]
            y = df['species']
            
            # Encode target classes to numeric
            unique_classes = y.unique()
            class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
            y_encoded = y.map(class_mapping)
            
            print(f"Features prepared: {X.shape}")
            print(f"Target encoded: {len(unique_classes)} classes")
            time.sleep(0.3)
            pbar.update(1)
            
            # Step 4: Train-test split
            pbar.set_description("Splitting data...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, 
                test_size=0.3, 
                random_state=42,
                stratify=y_encoded
            )
            
            print(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
            time.sleep(0.3)
            pbar.update(1)
            
            # Step 5: Train model
            pbar.set_description("Training model...")
            model = DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            print("Decision Tree model trained successfully")
            time.sleep(0.3)
            pbar.update(1)
            
            # Step 6: Make predictions
            pbar.set_description("Making predictions...")
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Get prediction probabilities for additional metrics
            y_train_proba = model.predict_proba(X_train)
            y_test_proba = model.predict_proba(X_test)
            
            print("Predictions completed")
            time.sleep(0.3)
            pbar.update(1)
            
            # Step 7: Calculate metrics
            pbar.set_description("Calculating metrics...")
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            print(f"Model Performance:")
            print(f"   - Training Accuracy: {train_accuracy:.4f}")
            print(f"   - Test Accuracy: {test_accuracy:.4f}")
            time.sleep(0.3)
            pbar.update(1)
            
            # Step 8: Save comprehensive metrics
            pbar.set_description("Saving metrics...")
            
            # Create metrics dictionary
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "dataset_info": {
                    "total_samples": len(df),
                    "features": feature_columns,
                    "classes": unique_classes.tolist(),
                    "class_distribution": df['species'].value_counts().to_dict()
                },
                "data_split": {
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "test_ratio": 0.3
                },
                "model_config": {
                    "algorithm": "DecisionTreeClassifier",
                    "max_depth": 5,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "random_state": 42
                },
                "performance": {
                    "train_accuracy": float(train_accuracy),
                    "test_accuracy": float(test_accuracy),
                    "accuracy_difference": float(train_accuracy - test_accuracy)
                }
            }
            
            # Save metrics as JSON
            with open("metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Save simple metrics for DVC tracking
            with open("metrics.txt", "w") as f:
                f.write(f"Dataset Size: {len(df)}\n")
                f.write(f"Training Size: {len(X_train)}\n")
                f.write(f"Test Size: {len(X_test)}\n")
                f.write(f"Train Accuracy: {train_accuracy:.4f}\n")
                f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
                f.write(f"Model Overfitting: {(train_accuracy - test_accuracy):.4f}\n")
            
            print("Metrics saved to metrics.json and metrics.txt")
            time.sleep(0.3)
            pbar.update(1)
            
            # Step 9: Save model
            pbar.set_description("🗄Saving model...")
            
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Save model with joblib (better for sklearn models)
            model_path = "models/iris_model.pkl"
            joblib.dump(model, model_path)
            
            # Also save class mapping for inference
            mapping_path = "models/class_mapping.json"
            with open(mapping_path, "w") as f:
                json.dump(class_mapping, f, indent=2)
            
            print(f"Model saved to {model_path}")
            print(f"Class mapping saved to {mapping_path}")
            time.sleep(0.3)
            pbar.update(1)
            
        print("\n" + "=" * 60)
        print("Training Pipeline Completed Successfully!")
        print("=" * 60)
        
        return True
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the iris.csv file is in the data/ directory")
        return False
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nReady for DVC tracking and version control!")
    else:
        print("\nPipeline failed. Please check the errors above.")
        exit(1)
