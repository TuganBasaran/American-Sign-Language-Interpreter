import os
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def main():
    DATA_FILE = "data/landmarks_dataset.pkl"
    MODEL_PATH = "models/asl_landmark_model.pkl"
    
    print("=== ASL Sign Language Recognition System (MediaPipe Landmark) ===")
    
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Data file not found: {DATA_FILE}")
        print("Please run process_dataset.py script first.")
        return
        
    print(f"Loading data: {DATA_FILE} ...")
    with open(DATA_FILE, 'rb') as f:
        dataset = pickle.load(f)
        
    X = dataset['data']
    y = dataset['labels']
    
    print(f"Data shape: {X.shape}")
    print(f"Number of labels: {len(np.unique(y))}")
    
    if np.isnan(X).any():
        print("Warning: Data contains NaN values, cleaning...")
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        print(f"Cleaned data shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

    print("Training classification model (Random Forest)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"Training time: {time.time()-t0:.2f}s")

    print("\n--- Test Results ---")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")

if __name__ == "__main__":
    main()
