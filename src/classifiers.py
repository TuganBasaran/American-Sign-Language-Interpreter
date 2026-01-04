from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class SignClassifier:
    def __init__(self, model_type='svm'):
        self.model_type = model_type.lower()
        self.model = self._get_model()
        
    def _get_model(self):
        if self.model_type == 'svm':
            return SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        
        elif self.model_type == 'knn':
            return KNeighborsClassifier(n_neighbors=5)
            
        elif self.model_type == 'rf':
            return RandomForestClassifier(n_estimators=100, random_state=42)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train, y_train):
        print(f"Training started ({self.model_type}). Data shape: {X_train.shape}")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        
        print(f"Model: {self.model_type}")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:\n", report)
        
        return acc, cm

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved: {path}")

    def load_model(self, path):
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Model loaded: {path}")
        else:
            print(f"Model file not found: {path}")
