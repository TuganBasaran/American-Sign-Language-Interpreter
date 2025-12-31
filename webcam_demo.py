import cv2
import numpy as np
import sys
from src.features import FeatureExtractor
from src.classifiers import SignClassifier

def main():
    # 1. Modeli Yükle
    # Varsayılan olarak RF modelini deniyoruz
    MODEL_PATH = "models/asl_rf_model.pkl"
    
    print(f"Model yükleniyor: {MODEL_PATH} ...")
    classifier = SignClassifier(model_type='rf')
    try:
        classifier.load_model(MODEL_PATH)
    except Exception as e:
        print(f"HATA: Model yüklenemedi. Lütfen önce main.py'yi çalıştırıp eğitimi tamamlayın.\nHata detayı: {e}")
        return

    # 2. Öznitelik Çıkarıcı
    extractor = FeatureExtractor()

    # 3. Kamerayı Başlat
    cap = cv2.VideoCapture(0)
    # Eğer 0 açılmazsa şansımızı 1'de deneyelim (Harici kamera vs.)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("HATA: Kamera açılamadı!")
        print("\n=== ÇÖZÜM ÖNERİLERİ (MacOS) ===")
        print("1. MacOS'ta terminal uygulamasına (VS Code, Terminal, iTerm) KAMERA izni vermeniz gerekir.")
        print("   Adımlar: Sistem Ayarları > Gizlilik ve Güvenlik > Kamera -> Kullandığınız terminali açın.")
        print("2. İzni verdikten sonra terminali tamamen kapatıp yeniden açmanız gerekebilir.")
        return

    print("=== ASL Canlı Tanıma Modu ===")
    print("Çıkış için 'q' tuşuna basın.")
    
    # ROI (Region of Interest) Ayarları
    # Elin ekranda duracağı kare
    ROI_SIZE = 400
    
    # Tahmin filtresi (son n tahmini tutarak titremeyi azalt)
    prediction_history = []
    HISTORY_LEN = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Ayna görüntüsü yap (kullanım kolaylığı için)
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # ROI Koordinatları (Ekranın Tam Ortası)
        x1 = int((w - ROI_SIZE) / 2)
        y1 = int((h - ROI_SIZE) / 2)
        x2 = x1 + ROI_SIZE
        y2 = y1 + ROI_SIZE
        
        # 4. Görüntü İşleme ve Tahmin
        # ROI bölgesini kes
        roi = frame[y1:y2, x1:x2]
        
        if roi.size > 0:
            # Modelimiz raw (işlenmemiş) gri görüntü ile eğitildi (main.py'dan hatırlayalım)
            # Bu nedenle segmentasyon YAPMADAN, sadece resize ve gri dönüşümü ile veriyoruz.
            # Not: Arka planın sade olması başarıyı artırır.
            
            # HOG için özellik çıkar
            try:
                # features.py içindeki extract_hog_features zaten resize ve gray yapıyor
                features = extractor.extract_hog_features(roi)
                
                # Tahmin (reshape gerekli çünkü tek örnek var)
                prediction_index = classifier.predict(features.reshape(1, -1))[0]
                
                # Basit bir yumuşatma (smoothing) uygula
                prediction_history.append(prediction_index)
                if len(prediction_history) > HISTORY_LEN:
                    prediction_history.pop(0)
                
                # En çok tekrar eden tahmini bul
                from collections import Counter
                most_common = Counter(prediction_history).most_common(1)[0][0]
                
                # Ekrana yazdır
                cv2.putText(frame, f"Tahmin: {most_common}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
            except Exception as e:
                pass # Hata olursa (örn. görüntü çok küçük) devam et

        # 5. Görselleştirme
        # ROI karesini çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "Elinizi Kareye Yerlestirin", (x1, y2 + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

        cv2.imshow("ASL Real-time Recognition", frame)
        
        # 'q' ile çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
