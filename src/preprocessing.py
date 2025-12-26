import cv2
import numpy as np

class Preprocessor:
    def __init__(self):
        # Cilt rengi tespiti için HSV aralıkları
        # Bu değerler ortam ışığına göre ayarlanabilir.
        # Genel bir ten rengi aralığı:
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
    def segment_hand(self, img, debug=False):
        """
        Görüntüden el bölgesini segmente eder.
        
        Args:
            img (numpy.ndarray): BGR formatında girdi görüntüsü.
            debug (bool): Ara adımları göstermek için (geliştirme aşamasında).
            
        Returns:
            mask (numpy.ndarray): El bölgesi için binary maske.
            contour (numpy.ndarray): En büyük el konturu.
        """
        # 1. Gürültü azaltma (Gaussian Blur)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # 2. Renk uzayı dönüşümü (BGR -> HSV)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 3. Cilt rengi maskeleme
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # 4. Morfolojik İşlemler (Gürültü temizleme)
        kernel = np.ones((3, 3), np.uint8)
        
        # Opening: Beyaz gürültüyü temizle (Erosion -> Dilation)
        # Closing: Delikleri kapat (Dilation -> Erosion)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Dilation ile eli biraz kalınlaştır (kontur kopukluklarını önlemek için)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 5. Kontur Tespiti
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        main_contour = None
        if contours:
            # Alana göre en büyük konturu seç
            main_contour = max(contours, key=cv2.contourArea)
            
            # Çok küçük gürültüleri el olarak seçme
            if cv2.contourArea(main_contour) < 1000:
                main_contour = None
                
        return mask, main_contour

    def get_segmented_image(self, img):
        """
        Maskelenmiş renkli el görüntüsünü döndürür (Arka plan siyah).
        """
        mask, _ = self.segment_hand(img)
        return cv2.bitwise_and(img, img, mask=mask)
