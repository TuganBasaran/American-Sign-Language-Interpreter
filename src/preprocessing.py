import cv2
import numpy as np

class Preprocessor:
    def __init__(self):
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
    def segment_hand(self, img, debug=False):
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        kernel = np.ones((3, 3), np.uint8)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        main_contour = None
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(main_contour) < 1000:
                main_contour = None
                
        return mask, main_contour

    def get_segmented_image(self, img):
        mask, _ = self.segment_hand(img)
        return cv2.bitwise_and(img, img, mask=mask)
