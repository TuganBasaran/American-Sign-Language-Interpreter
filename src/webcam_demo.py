import cv2
import joblib
import numpy as np
import time
from hand_tracker import HandTracker

def main():
    MODEL_PATH = "models/asl_landmark_model.pkl"
    
    print(f"Loading model: {MODEL_PATH} ...")
    try:
        classifier = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Model could not be loaded ({e}). Please run main.py.")
        return

    tracker = HandTracker(detection_con=0.7, track_con=0.7)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("ERROR: Camera could not be opened!")
        print("MacOS Tip: Ensure specific permission is granted to the terminal.")
        return

    print("=== ASL Real-time Recognition (MediaPipe - Full Screen) ===")
    print("Press 'q' to exit.")
    
    prediction_history = []
    HISTORY_LEN = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        frame = tracker.find_hands(frame, draw=True)
        
        lm_list = tracker.find_position(frame)
        
        prediction_text = ""
        
        if len(lm_list) > 0:
            try:
                data = np.array(lm_list)
                
                base_x, base_y, base_z = data[0][0], data[0][1], data[0][2]
                data[:, 0] = data[:, 0] - base_x
                data[:, 1] = data[:, 1] - base_y
                data[:, 2] = data[:, 2] - base_z
                
                max_value = np.max(np.abs(data))
                if max_value > 0:
                    data = data / max_value
                
                features = data.flatten().reshape(1, -1)
                
                prediction = classifier.predict(features)[0]
                
                prediction_history.append(prediction)
                if len(prediction_history) > HISTORY_LEN:
                    prediction_history.pop(0)
                
                from collections import Counter
                most_common = Counter(prediction_history).most_common(1)[0][0]
                prediction_text = most_common
                
                wrist_x, wrist_y = int(lm_list[0][0] * w), int(lm_list[0][1] * h)
                
                cv2.rectangle(frame, (wrist_x - 20, wrist_y - 60), (wrist_x + 80, wrist_y - 10), (0,0,0), -1)
                cv2.putText(frame, prediction_text, (wrist_x, wrist_y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            except Exception as e:
                print(f"Error: {e}")
        else:
            prediction_history = [] 

        cv2.putText(frame, "Exit: 'q' | Place your hand in front of the camera", (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("ASL Real-time (MediaPipe)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
