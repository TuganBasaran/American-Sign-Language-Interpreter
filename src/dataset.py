import os
import cv2
import numpy as np

class DataLoader:
    def __init__(self, data_path, img_size=(64, 64)):
        self.data_path = data_path
        self.img_size = img_size
        self.classes = sorted(os.listdir(data_path))
        self.classes = [c for c in self.classes if not c.startswith('.')]

    def load_images(self, max_samples_per_class=None):
        images = []
        labels = []
        
        print(f"Loading data: {self.data_path}")
        print(f"Total classes: {len(self.classes)}")
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_path, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            file_names = os.listdir(class_dir)
            if max_samples_per_class:
                file_names = file_names[:max_samples_per_class]
            
            print(f"Loading class: {class_name} ({len(file_names)} samples)")
            
            for file_name in file_names:
                img_path = os.path.join(class_dir, file_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    images.append(img)
                    labels.append(class_name)
                    
        return np.array(images), np.array(labels)
