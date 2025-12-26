from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class SignClassifier:
    def __init__(self, model_type='svm'):
        """
        Sınıflandırıcı modelini başlatır.
        
        Args:
            model_type (str): 'svm', 'knn', veya 'rf'
        """
        self.model_type = model_type.lower()
        self.model = self._get_model()
        
    def _get_model(self):
        if self.model_type == 'svm':
            # Support Vector Machine
            # Probability=True, güven skoru için gerekli olabilir
            return SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        
        elif self.model_type == 'knn':
            # k-Nearest Neighbors
            return KNeighborsClassifier(n_neighbors=5)
            
        elif self.model_type == 'rf':
            # Random Forest
            return RandomForestClassifier(n_estimators=100, random_state=42)
            
        else:
            raise ValueError(f"Bilinmeyen model tipi: {self.model_type}")

    def train(self, X_train, y_train):
        """
        Modeli eğitir.
        """
        print(f"Eğitim başlıyor ({self.model_type}). Veri boyutu: {X_train.shape}")
        self.model.fit(X_train, y_train)
        print("Eğitim tamamlandı.")

    def predict(self, X):
        """
        Tahmin yapar.
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Modeli test eder ve sonuçları döndürür.
        """
        predictions = self.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        
        print(f"Model: {self.model_type}")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:\n", report)
        
        return acc, cm

    def save_model(self, path):
        """
        Modeli diske kaydeder.
        """
        # Klasör yoksa oluştur
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model kaydedildi: {path}")

    def load_model(self, path):
        """
        Modeli diskten yükler.
        """
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Model yüklendi: {path}")
        else:
            print(f"Model dosyası bulunamadı: {path}")
