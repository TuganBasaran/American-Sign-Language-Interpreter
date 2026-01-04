import pickle
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    DATA_FILE = "data/landmarks_dataset.pkl"
    MODEL_PATH = "models/asl_landmark_model.pkl"
    
    print("=== Data Augmentation and Training for Dual Hand Support ===")
    
    print(f"Loading data: {DATA_FILE} ...")
    try:
        with open(DATA_FILE, 'rb') as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        print("ERROR: Dataset not found. process_dataset.py must be run first.")
        return

    X = dataset['data']
    y = dataset['labels']
    
    print(f"Original Data Shape: {X.shape}")
    
    print("Mirroring data (Left/Right hand simulation)...")
    X_mirrored = X.copy()
    
    X_mirrored[:, 0::3] = X_mirrored[:, 0::3] * -1
    
    X_augmented = np.concatenate([X, X_mirrored], axis=0)
    y_augmented = np.concatenate([y, y], axis=0)
    
    print(f"New Data Shape: {X_augmented.shape}")
    
    if np.isnan(X_augmented).any():
        mask = ~np.isnan(X_augmented).any(axis=1)
        X_augmented = X_augmented[mask]
        y_augmented = y_augmented[mask]
    
    print("Training model (This may take a while as data size doubled)...")
    X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"Training time: {time.time()-t0:.2f}s")
    
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"New Model Accuracy: %{acc*100:.2f}")
    
    joblib.dump(clf, MODEL_PATH)
    print(f"Model updated and saved: {MODEL_PATH}")

if __name__ == "__main__":
    main()
