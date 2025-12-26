import os
import cv2
import numpy as np

class DataLoader:
    def __init__(self, data_path, img_size=(64, 64)):
        """
        Veri yükleyici sınıfı.
        
        Args:
            data_path (str): Veri setinin ana dizini (örn: data/archive/asl_alphabet_train)
            img_size (tuple): Görüntülerin yeniden boyutlandırılacağı ebat (width, height)
        """
        self.data_path = data_path
        self.img_size = img_size
        self.classes = sorted(os.listdir(data_path))
        # .DS_Store vb. dosyaları temizle
        self.classes = [c for c in self.classes if not c.startswith('.')]

    def load_images(self, max_samples_per_class=None):
        """
        Görüntüleri ve etiketleri yükler.
        
        Args:
            max_samples_per_class (int): Her sınıftan en fazla kaç örnek alınacağı (None ise hepsi).
                                         Hızlı test için küçük bir sayı verilebilir.
        
        Returns:
            images (list): Yüklenen görüntüler listesi.
            labels (list): Etiket listesi (dizin isimleri).
        """
        images = []
        labels = []
        
        print(f"Veri yükleniyor: {self.data_path}")
        print(f"Toplam sınıf sayısı: {len(self.classes)}")
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_path, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            file_names = os.listdir(class_dir)
            if max_samples_per_class:
                file_names = file_names[:max_samples_per_class]
            
            print(f"Sınıf yükleniyor: {class_name} ({len(file_names)} örnek)")
            
            for file_name in file_names:
                img_path = os.path.join(class_dir, file_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    images.append(img)
                    labels.append(class_name)
                    
        return np.array(images), np.array(labels)
