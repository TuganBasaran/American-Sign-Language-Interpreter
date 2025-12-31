import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from src.dataset import DataLoader
from src.features import FeatureExtractor
from src.classifiers import SignClassifier

def main():
    # Ayarlar
    # data/archive altındaki klasör yapısına göre ayarlandı
    DATA_PATH = os.path.join("data", "archive", "asl_alphabet_train", "asl_alphabet_train")
    # Tam eğitim için bunu None yapın veya büyütün
    SAMPLES_PER_CLASS = None 
    
    print("=== ASL İşaret Dili Tanıma Sistemi (Klasik Yöntemler) Başlatılıyor ===")
    
    # 1. Veri Yükleme
    loader = DataLoader(DATA_PATH, img_size=(64, 64))
    
    # Veri seti yolunu kontrol et
    if not os.path.exists(DATA_PATH):
        # Arşivden çıkmamış olabilir, alternate path dene
        DATA_PATH = os.path.join("data", "archive", "asl_alphabet_train", "asl_alphabet_train")
        loader.data_path = DATA_PATH
        if not os.path.exists(DATA_PATH):
            print(f"HATA: Veri seti bulunamadı: {DATA_PATH}")
            return

    t0 = time.time()
    images, labels = loader.load_images(max_samples_per_class=SAMPLES_PER_CLASS)
    print(f"Veri yükleme tamamlandı: {len(images)} görüntü. Süre: {time.time()-t0:.2f}sn")
    
    if len(images) == 0:
        print("HATA: Hiç görüntü yüklenemedi!")
        return

    # 2. Öznitelik Çıkarımı
    print("Öznitelik çıkarımı yapılıyor (HOG)...")
    extractor = FeatureExtractor()
    features = []
    
    t0 = time.time()
    for img in images:
        # Sadece HOG kullan, contour şimdilik atla (hız ve basitlik için)
        feat = extractor.extract_hog_features(img)
        features.append(feat)
    
    X = np.array(features)
    y = np.array(labels)
    print(f"Öznitelik vektörü boyutu: {X.shape}. Süre: {time.time()-t0:.2f}sn")

    # 3. Eğitim/Test Ayrımı
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    print(f"Eğitim seti: {len(X_train)}, Test seti: {len(X_test)}")

    MODEL_PATH = "models/asl_rf_model.pkl"

    # 4. Sınıflandırma
    # Büyük veri setlerinde SVM (RBF) çok yavaş olabilir. Random Forest daha hızlı sonuç verir.
    # Varsayılan olarak Random Forest (rf) kullanıyoruz.
    classifier = SignClassifier(model_type='rf')
    
    if os.path.exists(MODEL_PATH):
        print(f"Kayıtlı model bulundu, yükleniyor: {MODEL_PATH}")
        classifier.load_model(MODEL_PATH)
    else:
        print("Kayıtlı model bulunamadı. Sınıflandırma modeli eğitiliyor (Bu işlem zaman alabilir)...")
        
        t0 = time.time()
        classifier.train(X_train, y_train)
        print(f"Eğitim süresi: {time.time()-t0:.2f}sn")
        
        # Modeli kaydet
        classifier.save_model(MODEL_PATH)

    # 5. Değerlendirme
    print("\n--- Test Sonuçları ---")
    acc, cm = classifier.evaluate(X_test, y_test)
    


if __name__ == "__main__":
    main()
