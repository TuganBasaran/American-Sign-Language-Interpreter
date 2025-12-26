import cv2
import numpy as np
from skimage.feature import hog

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_hog_features(self, img, cell_size=(8, 8), block_size=(2, 2), bins=9):
        """
        Histogram of Oriented Gradients (HOG) özniteliklerini çıkarır.
        
        Args:
            img: Gri seviye veya renkli görüntü.
        
        Returns:
            features: HOG vektörü.
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Görüntüyü standart bir boyuta getir (HOG için önemli)
        # 64x64 genellikle iyi bir başlangıçtır
        img = cv2.resize(img, (64, 64))
        
        features = hog(img, 
                       orientations=bins, 
                       pixels_per_cell=cell_size,
                       cells_per_block=block_size, 
                       visualize=False, 
                       # multichannel=False, # Yeni scikit-image sürümlerinde 'channel_axis' kullanılabilir
                       block_norm='L2-Hys')
        return features

    def extract_hu_moments(self, contour):
        """
        Hu Moments (Şekil değişmezleri) özniteliklerini çıkarır.
        
        Args:
            contour: El konturu.
            
        Returns:
            moments: 7 elemanlı Hu moments vektörü (log scale).
        """
        if contour is None:
            return np.zeros(7)
            
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Log scale dönüşümü (değerler çok küçük olduğu için)
        # İşareti koruyarak log al -> h = -sign(h) * log10(abs(h))
        processed_moments = []
        for i in range(7):
            val = hu_moments[i]
            if val != 0:
                processed_moments.append(-1 * np.sign(val) * np.log10(np.abs(val)))
            else:
                processed_moments.append(0)
                
        return np.array(processed_moments)

    def extract_all(self, img, contour=None):
        """
        Tüm özellikleri birleştirir.
        
        Args:
            img: Görüntü (HOG için)
            contour: Kontur (Hu Moments için) - Opsiyonel
            
        Returns:
            combined_features: Birleştirilmiş öznitelik vektörü
        """
        hog_feats = self.extract_hog_features(img)
        
        # Şimdilik sadece HOG ile başlayalım, contour features opsiyonel
        # Eğer contour verilirse dahil et
        if contour is not None:
            hu_feats = self.extract_hu_moments(contour)
            return np.concatenate([hog_feats, hu_feats])
            
        return hog_feats
