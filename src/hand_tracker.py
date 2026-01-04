import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, mode=False, max_hands=1, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[0]
            
            for id, lm in enumerate(my_hand.landmark):
                lm_list.append([lm.x, lm.y, lm.z])
                
        return lm_list

    def get_normalized_landmarks(self, img):
        self.find_hands(img, draw=False)
        lm_list = self.find_position(img)
        
        if not lm_list:
            return None
            
        data = np.array(lm_list)
        
        base_x, base_y, base_z = data[0][0], data[0][1], data[0][2]
        
        data[:, 0] = data[:, 0] - base_x
        data[:, 1] = data[:, 1] - base_y
        data[:, 2] = data[:, 2] - base_z
        
        max_value = np.max(np.abs(data))
        if max_value > 0:
            data = data / max_value
            
        return data.flatten()
